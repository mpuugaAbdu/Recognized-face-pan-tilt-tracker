"""SQLite-backed storage for enrolled people, photos, and embeddings."""

from __future__ import annotations

from datetime import UTC, datetime
import re

import numpy as np
from arduino.app_bricks.dbstorage_sqlstore import SQLStore


class FaceDatabase:
    def __init__(self) -> None:
        self.store = SQLStore("face-recognition.db")
        self.store.create_table("people", {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "name": "TEXT NOT NULL UNIQUE",
            "embedding": "BLOB NOT NULL",
            "sample_count": "INTEGER NOT NULL",
            "created_at": "TEXT NOT NULL",
        })
        self.store.create_table("face_photos", {
            "id": "INTEGER PRIMARY KEY AUTOINCREMENT",
            "person_id": "INTEGER NOT NULL",
            "photo": "BLOB NOT NULL",
            "mime_type": "TEXT NOT NULL",
            "created_at": "TEXT NOT NULL",
        })
        existing = {row["name"] for row in self.store.execute_sql("PRAGMA table_info(face_photos)") or []}
        if "embedding" not in existing:
            self.store.execute_sql("ALTER TABLE face_photos ADD COLUMN embedding BLOB")
        if "quality" not in existing:
            self.store.execute_sql("ALTER TABLE face_photos ADD COLUMN quality REAL")

    @staticmethod
    def _clean_name(name: str) -> str:
        cleaned = re.sub(r"\s+", " ", str(name)).strip()
        if not cleaned or len(cleaned) > 80:
            raise ValueError("Name must contain 1 to 80 characters")
        return cleaned

    def list_people(self, include_embeddings: bool = False) -> list[dict]:
        columns = None if include_embeddings else ["id", "name", "sample_count", "created_at"]
        return self.store.read("people", columns=columns, order_by="name COLLATE NOCASE ASC")

    def templates(self) -> list[tuple[int, str, list[np.ndarray], np.ndarray]]:
        result: list[tuple[int, str, list[np.ndarray], np.ndarray]] = []
        for row in self.list_people(include_embeddings=True):
            centroid = np.frombuffer(row["embedding"], dtype=np.float32).copy()
            samples = self.store.execute_sql(
                "SELECT embedding FROM face_photos WHERE person_id = ? AND embedding IS NOT NULL ORDER BY quality DESC",
                (row["id"],),
            ) or []
            vectors = [np.frombuffer(sample["embedding"], dtype=np.float32).copy() for sample in samples]
            result.append((row["id"], row["name"], vectors or [centroid], centroid))
        return result

    def enroll(self, name: str, photos: list[tuple[bytes, str, np.ndarray, float]], embedding: np.ndarray) -> int:
        name = self._clean_name(name)
        now = datetime.now(UTC).isoformat()
        with self.store.conn_lock:
            conn = self.store._get_connection()
            try:
                conn.execute("BEGIN")
                old = conn.execute("SELECT id FROM people WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
                if old:
                    person_id = int(old["id"])
                    conn.execute("DELETE FROM face_photos WHERE person_id = ?", (person_id,))
                    conn.execute(
                        "UPDATE people SET name = ?, embedding = ?, sample_count = ?, created_at = ? WHERE id = ?",
                        (name, embedding.astype(np.float32).tobytes(), len(photos), now, person_id),
                    )
                else:
                    cursor = conn.execute(
                        "INSERT INTO people (name, embedding, sample_count, created_at) VALUES (?, ?, ?, ?)",
                        (name, embedding.astype(np.float32).tobytes(), len(photos), now),
                    )
                    person_id = int(cursor.lastrowid)
                conn.executemany(
                    "INSERT INTO face_photos (person_id, photo, mime_type, created_at, embedding, quality) VALUES (?, ?, ?, ?, ?, ?)",
                    [(person_id, photo, mime, now, vector.astype(np.float32).tobytes(), quality) for photo, mime, vector, quality in photos],
                )
                conn.commit()
                return person_id
            except Exception:
                conn.rollback()
                raise

    def delete_person(self, person_id: int) -> None:
        with self.store.conn_lock:
            conn = self.store._get_connection()
            try:
                conn.execute("BEGIN")
                conn.execute("DELETE FROM face_photos WHERE person_id = ?", (person_id,))
                conn.execute("DELETE FROM people WHERE id = ?", (person_id,))
                conn.commit()
            except Exception:
                conn.rollback()
                raise
