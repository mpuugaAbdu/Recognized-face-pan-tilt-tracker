# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
# SPDX-License-Identifier: MPL-2.0

"""UNO Q face detection, enrollment, and recognition application."""

from __future__ import annotations

import base64
from collections import deque
from datetime import UTC, datetime
import threading
import time

import numpy as np
from arduino.app_utils import App
from arduino.app_bricks.web_ui import WebUI
from arduino.app_bricks.video_objectdetection import VideoObjectDetection

from database import FaceDatabase
from face_tracker import face_tracker
from recognition import FaceEmbedder, mean_embedding, select_consistent_samples


DETECTION_THRESHOLD = 0.5
MATCH_THRESHOLD = 0.48
MIN_IDENTITY_MARGIN = 0.045
RECOGNITION_INTERVAL_SECONDS = 0.14
IDENTITY_HOLD_SECONDS = 0.85
TRACK_ASSOCIATION_DISTANCE = 0.32
SEARCH_TIMEOUT_SECONDS = 45.0
MIN_ENROLLMENT_PHOTOS = 5
MAX_ENROLLMENT_PHOTOS = 20
MAX_PHOTO_BYTES = 2 * 1024 * 1024
CAMERA_ENROLLMENT_SAMPLES = 10
CAMERA_CAPTURE_INTERVAL_SECONDS = 0.55
CAMERA_SAMPLE_MAX_SIMILARITY = 0.995

ui = WebUI()
database = FaceDatabase()
embedder = FaceEmbedder()
detection_stream = VideoObjectDetection(
    confidence=DETECTION_THRESHOLD,
    debounce_sec=0.0,
    camera_preview=True,
)

templates_lock = threading.RLock()
templates = database.templates()
last_recognition_at = 0.0
enrollment_lock = threading.RLock()
enrollment_sessions: dict[str, dict] = {}
recognition_votes: deque[str] = deque(maxlen=3)
tracking_lock = threading.RLock()
tracked_identity: dict | None = None
search_state: dict | None = None
search_thread_stop = threading.Event()


def refresh_templates() -> None:
    global templates
    with templates_lock:
        templates = database.templates()
    recognition_votes.clear()


def recognize(vector: np.ndarray) -> tuple[str, float, int | None]:
    with templates_lock:
        candidates = list(templates)
    if not candidates:
        return "Unknown", 0.0, None
    scored = []
    for person_id, name, samples, centroid in candidates:
        sample_scores = sorted((float(np.dot(vector, sample)) for sample in samples), reverse=True)
        top_count = min(3, len(sample_scores))
        sample_score = float(np.mean(sample_scores[:top_count]))
        centroid_score = float(np.dot(vector, centroid))
        scored.append((0.75 * sample_score + 0.25 * centroid_score, person_id, name))
    scored.sort(reverse=True)
    score, person_id, name = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else -1.0
    if score < MATCH_THRESHOLD or score - second_score < MIN_IDENTITY_MARGIN:
        return "Unknown", score, None
    return name, score, person_id


def stabilize(name: str) -> str:
    """Acquire a recognized identity immediately.

    Matching has already applied both the similarity threshold and the margin
    over the second-best person.  Keeping an additional multi-frame gate here
    only adds visible motor latency after recognition succeeds.
    """
    recognition_votes.append(name)
    return name if name != "Unknown" else "Unknown"


def _box_center(box) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def _expire_tracking_lock() -> None:
    """Convert an expired recognized-face lock into a bounded search."""
    global tracked_identity, search_state
    with tracking_lock:
        if tracked_identity is None:
            return
        if time.monotonic() - tracked_identity["seen_at"] <= IDENTITY_HOLD_SECONDS:
            return
        search_state = {
            "name": tracked_identity["name"],
            "person_id": tracked_identity["person_id"],
            "started_at": time.monotonic(),
        }
        tracked_identity = None
        recognition_votes.clear()


def _search_watchdog() -> None:
    """Run lost-face search independently of detector callbacks.

    Some video-object-detection brick versions do not invoke on_detect_all for
    empty results. Search therefore cannot depend on receiving another frame
    after the face has disappeared.
    """
    global search_state
    while not search_thread_stop.wait(0.02):
        _expire_tracking_lock()
        with tracking_lock:
            active_search = search_state
            if (
                active_search is not None
                and time.monotonic() - active_search["started_at"]
                > SEARCH_TIMEOUT_SECONDS
            ):
                search_state = None
                active_search = None
        if active_search is not None:
            try:
                face_tracker.search_step()
            except Exception as exc:
                # Keep the worker alive after a transient bridge error and make
                # the failure visible in the browser.
                ui.send_message(
                    "app_error", {"message": f"Search movement failed: {exc}"}
                )
                search_thread_stop.wait(0.25)


def _select_tracking_face(faces: list[dict], width: int, height: int) -> dict | None:
    """Associate fresh detector boxes with the last recognized face."""
    global tracked_identity
    with tracking_lock:
        if tracked_identity is None:
            return None
        previous_x, previous_y = _box_center(tracked_identity["box"])

    best, best_distance = None, float("inf")
    diagonal = max((width * width + height * height) ** 0.5, 1.0)
    for face in faces:
        box = face.get("bounding_box_xyxy")
        if not box:
            continue
        center_x, center_y = _box_center(box)
        distance = ((center_x - previous_x) ** 2 + (center_y - previous_y) ** 2) ** 0.5 / diagonal
        if distance < best_distance:
            best, best_distance = face, distance
    return best if best_distance <= TRACK_ASSOCIATION_DISTANCE else None


def _save_camera_enrollment(sid: str, session: dict) -> None:
    """Persist an automatically completed live-camera enrollment."""
    try:
        photos = [
            (photo, "image/jpeg", vector, quality)
            for photo, vector, quality in zip(
                session["photos"], session["vectors"], session["qualities"]
            )
        ]
        person_id = database.enroll(
            session["name"], photos, mean_embedding(session["vectors"])
        )
        refresh_templates()
        ui.send_message("enrollment_result", {
            "ok": True, "person_id": person_id, "name": session["name"],
            "sample_count": len(photos), "rejected_count": 0,
        }, room=sid)
        send_people()
    except Exception as exc:
        ui.send_message(
            "enrollment_result", {"ok": False, "message": str(exc)}, room=sid
        )


def _capture_camera_enrollment(image: np.ndarray, frame: bytes, faces: list[dict]) -> bool:
    """Capture one useful live frame for each active camera enrollment.

    Returns True while a camera enrollment is active. During that short period
    normal recognition/tracking is paused so the user and camera stay still.
    """
    now = time.monotonic()
    with enrollment_lock:
        active = [
            (sid, session) for sid, session in enrollment_sessions.items()
            if session.get("mode") == "camera"
        ]
    if not active:
        return False

    valid_faces = [face for face in faces if face.get("bounding_box_xyxy")]
    if len(valid_faces) != 1:
        reason = "Keep exactly one face in view"
        for sid, session in active:
            if now - session.get("last_notice_at", 0.0) >= 0.8:
                session["last_notice_at"] = now
                ui.send_message("enrollment_progress", {
                    "processed": len(session["vectors"]),
                    "accepted": len(session["vectors"]),
                    "total": session["expected"], "rejected_reason": reason,
                }, room=sid)
        return True

    box = tuple(valid_faces[0]["bounding_box_xyxy"])
    for sid, session in active:
        if now - session.get("last_capture_at", 0.0) < CAMERA_CAPTURE_INTERVAL_SECONDS:
            continue
        session["last_capture_at"] = now
        try:
            crop = embedder.crop_face(image, box)
            quality = embedder.assess_face(crop)
            vector = embedder.embedding(crop)
            if session["vectors"]:
                closest = max(float(np.dot(vector, old)) for old in session["vectors"])
                if closest >= CAMERA_SAMPLE_MAX_SIMILARITY:
                    raise ValueError("Slowly turn or tilt your head for a different view")
            with enrollment_lock:
                session["photos"].append(frame)
                session["vectors"].append(vector)
                session["qualities"].append(quality)
                accepted = len(session["vectors"])
                finished = accepted >= session["expected"]
                if finished:
                    enrollment_sessions.pop(sid, None)
            ui.send_message("enrollment_progress", {
                "processed": accepted, "accepted": accepted,
                "total": session["expected"], "rejected_reason": None,
            }, room=sid)
            if finished:
                _save_camera_enrollment(sid, session)
        except Exception as exc:
            if now - session.get("last_notice_at", 0.0) >= 0.8:
                session["last_notice_at"] = now
                ui.send_message("enrollment_progress", {
                    "processed": len(session["vectors"]),
                    "accepted": len(session["vectors"]),
                    "total": session["expected"], "rejected_reason": str(exc),
                }, room=sid)
    return True


def send_detections_to_ui(detections: dict, frame: bytes | None = None) -> None:
    global last_recognition_at, tracked_identity, search_state
    faces = detections.get("face", [])
    image = embedder.decode_jpeg(frame) if frame is not None else None
    if image is not None and _capture_camera_enrollment(image, frame, faces):
        return
    _expire_tracking_lock()
    if not faces:
        return

    # Once identity has been confirmed, use every detector update for motion.
    # Recognition remains deliberately slower because embedding inference is
    # expensive, but it no longer limits servo command cadence.
    if image is not None:
        tracked_face = _select_tracking_face(faces, int(image.shape[1]), int(image.shape[0]))
        if tracked_face is not None:
            box = tuple(tracked_face["bounding_box_xyxy"])
            command = face_tracker.update(box, int(image.shape[1]), int(image.shape[0]))
            with tracking_lock:
                if tracked_identity is not None:
                    tracked_identity["box"] = box
                    tracked_name = tracked_identity["name"]
                else:
                    tracked_name = "Unknown"
            ui.send_message("tracking", {
                "name": tracked_name, "pan": command.pan, "tilt": command.tilt,
                "horizontal_error": round(command.horizontal_error, 4),
                "vertical_error": round(command.vertical_error, 4),
            })

    now = time.monotonic()
    if frame is None or now - last_recognition_at < RECOGNITION_INTERVAL_SECONDS:
        return
    last_recognition_at = now

    try:
        if image is None:
            image = embedder.decode_jpeg(frame)
        results = []
        # Keep the locked face first; otherwise prioritize a large, central face
        # so callback ordering cannot make identities jump between people.
        locked = _select_tracking_face(faces, int(image.shape[1]), int(image.shape[0]))
        ordered_faces = list(faces)
        if locked is not None:
            ordered_faces.remove(locked)
            ordered_faces.insert(0, locked)
        else:
            ordered_faces.sort(key=lambda f: -(
                (f.get("bounding_box_xyxy", [0, 0, 0, 0])[2] - f.get("bounding_box_xyxy", [0, 0, 0, 0])[0]) *
                (f.get("bounding_box_xyxy", [0, 0, 0, 0])[3] - f.get("bounding_box_xyxy", [0, 0, 0, 0])[1])
            ))
        for face_index, face in enumerate(ordered_faces):
            box = face.get("bounding_box_xyxy")
            if not box:
                continue
            vector = embedder.embedding(embedder.crop_face(image, tuple(box)))
            name, score, person_id = recognize(vector)
            with tracking_lock:
                expected_name = search_state["name"] if search_state is not None else None
            if expected_name is not None and name != expected_name:
                name, person_id = "Unknown", None
            stable_name = stabilize(name) if face_index == 0 else "Unknown"
            if stable_name != "Unknown":
                with tracking_lock:
                    tracked_identity = {
                        "name": stable_name,
                        "person_id": person_id,
                        "box": tuple(box),
                        "seen_at": time.monotonic(),
                    }
                    search_state = None
                face_tracker.stop_search()
            results.append({
                "person_id": person_id if stable_name != "Unknown" else None,
                "name": stable_name,
                "candidate_name": name,
                "similarity": round(score, 4),
                "detection_confidence": face.get("confidence", 0.0),
                "bounding_box": list(box),
                "timestamp": datetime.now(UTC).isoformat(),
            })
        if results:
            ui.send_message("recognition", {
                "faces": results,
                "frame_width": int(image.shape[1]),
                "frame_height": int(image.shape[0]),
            })

    except Exception as exc:
        ui.send_message("app_error", {"message": f"Recognition failed: {exc}"})


def start_enrollment(sid, payload: dict) -> None:
    try:
        name = str(payload.get("name", "")).strip()
        expected = int(payload.get("photo_count", 0))
        if not name:
            raise ValueError("Enter the person's name")
        if not MIN_ENROLLMENT_PHOTOS <= expected <= MAX_ENROLLMENT_PHOTOS:
            raise ValueError(f"Choose between {MIN_ENROLLMENT_PHOTOS} and {MAX_ENROLLMENT_PHOTOS} face photos")
        with enrollment_lock:
            enrollment_sessions[sid] = {
                "name": name, "expected": expected, "processed": 0,
                "photos": [], "vectors": [], "qualities": [], "rejected": [],
            }
        ui.send_message("enrollment_progress", {"processed": 0, "accepted": 0, "total": expected}, room=sid)
    except Exception as exc:
        ui.send_message("enrollment_result", {"ok": False, "message": str(exc)}, room=sid)


def start_camera_enrollment(sid, payload: dict) -> None:
    """Begin automatic enrollment from frames produced by the camera brick."""
    global tracked_identity, search_state
    try:
        name = str(payload.get("name", "")).strip()
        expected = int(payload.get("sample_count", CAMERA_ENROLLMENT_SAMPLES))
        if not name:
            raise ValueError("Enter the person's name")
        if not MIN_ENROLLMENT_PHOTOS <= expected <= 15:
            raise ValueError("Choose between 5 and 15 camera samples")
        with enrollment_lock:
            # Only one live enrollment can safely own the shared camera at once.
            enrollment_sessions.clear()
            enrollment_sessions[sid] = {
                "mode": "camera", "name": name, "expected": expected,
                "processed": 0, "photos": [], "vectors": [],
                "qualities": [], "rejected": [], "last_capture_at": 0.0,
                "last_notice_at": 0.0,
            }
        with tracking_lock:
            tracked_identity = None
            search_state = None
        recognition_votes.clear()
        ui.send_message("enrollment_progress", {
            "processed": 0, "accepted": 0, "total": expected,
            "rejected_reason": "Look at the camera, then slowly turn and tilt your head",
        }, room=sid)
    except Exception as exc:
        ui.send_message(
            "enrollment_result", {"ok": False, "message": str(exc)}, room=sid
        )


def cancel_camera_enrollment(sid, _payload=None) -> None:
    with enrollment_lock:
        enrollment_sessions.pop(sid, None)
    ui.send_message("enrollment_result", {
        "ok": False, "cancelled": True, "message": "Camera enrollment cancelled",
    }, room=sid)


def add_enrollment_photo(sid, item: dict) -> None:
    try:
        with enrollment_lock:
            session = enrollment_sessions.get(sid)
        if session is None:
            raise ValueError("Enrollment session expired; please try again")
        if session["processed"] >= session["expected"]:
            raise ValueError("More photos were received than expected")
        encoded = str(item.get("data", ""))
        mime = str(item.get("type", "image/jpeg"))
        if "," in encoded:
            encoded = encoded.split(",", 1)[1]
        photo = base64.b64decode(encoded, validate=True)
        if not photo or len(photo) > MAX_PHOTO_BYTES:
            raise ValueError("A photo is empty or larger than 2 MB after resizing")
        try:
            vector, quality = embedder.embedding_and_quality_from_photo(photo)
        except Exception as exc:
            # A single weak upload should not discard an otherwise useful
            # enrollment. Count it, explain it, and let the browser continue.
            with enrollment_lock:
                session["processed"] += 1
                session["rejected"].append(str(exc))
                processed = session["processed"]
                accepted = len(session["photos"])
            ui.send_message(
                "enrollment_progress",
                {
                    "processed": processed, "accepted": accepted,
                    "total": session["expected"], "rejected_reason": str(exc),
                },
                room=sid,
            )
            return
        with enrollment_lock:
            session["processed"] += 1
            session["photos"].append((photo, mime))
            session["vectors"].append(vector)
            session["qualities"].append(quality)
            processed = session["processed"]
            accepted = len(session["photos"])
        ui.send_message(
            "enrollment_progress",
            {"processed": processed, "accepted": accepted, "total": session["expected"], "rejected_reason": None},
            room=sid,
        )
    except Exception as exc:
        with enrollment_lock:
            enrollment_sessions.pop(sid, None)
        ui.send_message("enrollment_result", {"ok": False, "message": str(exc)}, room=sid)


def finish_enrollment(sid, _payload: dict) -> None:
    try:
        with enrollment_lock:
            session = enrollment_sessions.pop(sid, None)
        if session is None:
            raise ValueError("Enrollment session expired; please try again")
        if session["processed"] != session["expected"]:
            raise ValueError(f"Only {session['processed']} of {session['expected']} photos reached the board")
        if len(session["vectors"]) < MIN_ENROLLMENT_PHOTOS:
            raise ValueError(
                f"Only {len(session['vectors'])} photos could be decoded; upload at least {MIN_ENROLLMENT_PHOTOS} valid image files"
            )
        selected = select_consistent_samples(session["vectors"], session["qualities"])
        photos = [
            (session["photos"][i][0], session["photos"][i][1], session["vectors"][i], session["qualities"][i])
            for i in selected
        ]
        person_id = database.enroll(session["name"], photos, mean_embedding([session["vectors"][i] for i in selected]))
        refresh_templates()
        ui.send_message("enrollment_result", {
            "ok": True,
            "person_id": person_id,
            "name": session["name"],
            "sample_count": len(photos),
            "rejected_count": session["expected"] - len(photos),
        }, room=sid)
        send_people()
    except Exception as exc:
        ui.send_message("enrollment_result", {"ok": False, "message": str(exc)}, room=sid)


def send_people(_sid=None, _payload=None) -> None:
    ui.send_message("people", {"people": database.list_people()})


def delete_person(_sid, payload: dict) -> None:
    try:
        database.delete_person(int(payload["person_id"]))
        refresh_templates()
        send_people()
    except Exception as exc:
        ui.send_message("app_error", {"message": f"Could not delete person: {exc}"})


ui.on_message("override_th", lambda _sid, threshold: detection_stream.override_threshold(float(threshold)))
ui.on_message("camera_enrollment_start", start_camera_enrollment)
ui.on_message("camera_enrollment_cancel", cancel_camera_enrollment)
ui.on_message("enrollment_start", start_enrollment)
ui.on_message("enrollment_photo", add_enrollment_photo)
ui.on_message("enrollment_finish", finish_enrollment)
ui.on_message("list_people", send_people)
ui.on_message("delete_person", delete_person)
detection_stream.on_detect_all(send_detections_to_ui)

threading.Thread(
    target=_search_watchdog, name="lost-face-search", daemon=True
).start()
App.run()
