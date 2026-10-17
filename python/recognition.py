"""Face embedding helpers using ONNX Runtime on the UNO Q."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import threading

import numpy as np
import onnxruntime as ort
from PIL import Image


MODEL_PATH = Path(__file__).with_name("face_recognition_sface_2021dec.onnx")


class FaceEmbedder:
    """Create normalized SFace embeddings without relying on OpenCV DNN."""

    def __init__(self, model_path: Path = MODEL_PATH) -> None:
        if not model_path.exists():
            raise FileNotFoundError(f"Face recognition model is missing: {model_path}")
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._session_lock = threading.Lock()

    @staticmethod
    def decode_jpeg(data: bytes) -> np.ndarray:
        """Decode an image as an RGB uint8 NumPy array."""
        return np.asarray(Image.open(BytesIO(data)).convert("RGB"))

    @staticmethod
    def crop_face(frame: np.ndarray, box: tuple[int, int, int, int], margin: float = 0.22) -> np.ndarray:
        """Return a square, undistorted face crop around a detector box.

        SFace is quite sensitive to geometric distortion.  Expanding the raw
        rectangular detector box and forcing it to 112x112 made narrow boxes
        look unnaturally wide.  A square crop also makes live-camera and
        enrollment preprocessing consistent.
        """
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = (int(v) for v in box)
        face_width, face_height = x2 - x1, y2 - y1
        if face_width <= 0 or face_height <= 0:
            raise ValueError("Invalid face bounding box")
        side = max(face_width, face_height) * (1.0 + 2.0 * margin)
        center_x = (x1 + x2) / 2.0
        # A small upward shift includes the forehead without adding too much
        # clothing below the chin.
        center_y = (y1 + y2) / 2.0 - 0.04 * face_height
        side = min(side, float(width), float(height))
        x1 = int(round(center_x - side / 2.0))
        y1 = int(round(center_y - side / 2.0))
        x1 = max(0, min(width - int(side), x1))
        y1 = max(0, min(height - int(side), y1))
        x2, y2 = min(width, x1 + int(side)), min(height, y1 + int(side))
        if x2 <= x1 or y2 <= y1:
            raise ValueError("Invalid face bounding box")
        return frame[y1:y2, x1:x2]

    def embedding(self, face: np.ndarray) -> np.ndarray:
        if face is None or face.size == 0:
            raise ValueError("The face image is empty")

        # SFace was trained with OpenCV's BGR preprocessing. PIL resizes our
        # RGB array, then the final axis is reversed to produce BGR.
        resized_rgb = np.asarray(Image.fromarray(face).resize((112, 112), Image.Resampling.BILINEAR))
        bgr = resized_rgb[:, :, ::-1].astype(np.float32)
        normalized = (bgr - 127.5) / 128.0
        blob = np.transpose(normalized, (2, 0, 1))[None, ...]

        with self._session_lock:
            output = self._session.run(None, {self._input_name: blob})[0]
        vector = np.asarray(output, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if norm < 1e-8:
            raise ValueError("The recognition model returned an empty embedding")
        return vector / norm

    def embedding_from_photo(self, data: bytes) -> np.ndarray:
        """Embed an uploaded close-up photo containing one centered face."""
        frame = self.decode_jpeg(data)
        height, width = frame.shape[:2]
        side = min(height, width)
        x1, y1 = (width - side) // 2, (height - side) // 2
        return self.embedding(frame[y1 : y1 + side, x1 : x1 + side])

    def assess_photo(self, data: bytes) -> tuple[np.ndarray, float]:
        """Return the decoded image and a 0..1 quality score, or reject it."""
        frame = self.decode_jpeg(data)
        return frame, self.assess_face(frame)

    @staticmethod
    def assess_face(frame: np.ndarray) -> float:
        """Score a decoded face crop and reject frames unsuitable for enrollment."""
        height, width = frame.shape[:2]
        if min(height, width) < 96:
            raise ValueError("Move closer to the camera")
        gray = np.mean(frame.astype(np.float32), axis=2)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        edge_strength = float(
            (np.mean(np.abs(np.diff(gray, axis=0))) + np.mean(np.abs(np.diff(gray, axis=1)))) / 2.0
        )
        if brightness < 35:
            raise ValueError("The image is too dark; add light in front of you")
        if brightness > 225:
            raise ValueError("The image is overexposed; reduce the light")
        if contrast < 14:
            raise ValueError("The face has too little contrast")
        if edge_strength < 2.5:
            raise ValueError("Hold still; the image is blurry")
        brightness_score = max(0.0, 1.0 - abs(brightness - 128.0) / 128.0)
        quality = min(1.0, 0.45 * brightness_score + 0.25 * min(contrast / 55.0, 1.0) + 0.30 * min(edge_strength / 14.0, 1.0))
        return quality

    def embedding_and_quality_from_photo(self, data: bytes) -> tuple[np.ndarray, float]:
        frame, quality = self.assess_photo(data)
        height, width = frame.shape[:2]
        side = min(height, width)
        x1, y1 = (width - side) // 2, (height - side) // 2
        return self.embedding(frame[y1 : y1 + side, x1 : x1 + side]), quality


def mean_embedding(vectors: list[np.ndarray]) -> np.ndarray:
    if not vectors:
        raise ValueError("At least one face photo is required")
    result = np.mean(np.stack(vectors), axis=0).astype(np.float32)
    return result / max(float(np.linalg.norm(result)), 1e-8)


def select_consistent_samples(
    vectors: list[np.ndarray], qualities: list[float], maximum: int = 12
) -> list[int]:
    """Remove near-duplicates and embedding outliers, retaining varied good samples."""
    if len(vectors) != len(qualities):
        raise ValueError("Sample vectors and quality scores do not match")
    candidates = sorted(range(len(vectors)), key=lambda i: qualities[i], reverse=True)
    unique: list[int] = []
    for index in candidates:
        if all(float(np.dot(vectors[index], vectors[kept])) < 0.995 for kept in unique):
            unique.append(index)
    if len(unique) < 5:
        raise ValueError("Too many enrollment photos are duplicates; use more varied head angles")
    center = mean_embedding([vectors[i] for i in unique])
    similarities = np.array([float(np.dot(vectors[i], center)) for i in unique])
    cutoff = max(0.20, float(np.median(similarities) - 0.18))
    consistent = [i for i, score in zip(unique, similarities) if score >= cutoff]
    consistent.sort(key=lambda i: qualities[i], reverse=True)
    if len(consistent) < 5:
        raise ValueError("Enrollment photos are not consistent enough to describe one person")
    return consistent[:maximum]
