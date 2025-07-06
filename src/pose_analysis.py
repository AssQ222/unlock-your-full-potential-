import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
from typing import Tuple, Dict

mp_pose = mp.solutions.pose

# ── 1. detekcja punktów ───────────────────────────────────────────────────
def detect_keypoints(image: np.ndarray) -> Tuple[np.ndarray, Dict[int, Tuple[int, int]]]:
    """
    Zwraca: (obraz z nakładką, dict {id: (x, y)})
    """
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            raise ValueError("Nie wykryto sylwetki – spróbuj lepszego oświetlenia / całej postaci.")
        h, w = image.shape[:2]
        pts = {}
        for i, lm in enumerate(results.pose_landmarks.landmark):
            pts[i] = (int(lm.x * w), int(lm.y * h))
        annotated = image.copy()
        mp.solutions.drawing_utils.draw_landmarks(
            annotated, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
        return annotated, pts

# ── 2. obliczenia prostych metryk ─────────────────────────────────────────
def compute_metrics(pts: Dict[int, Tuple[int, int]]) -> Dict[str, float]:
    """
    Przykładowe metryki (kąt barków, asymetria barków, talia/biodra).
    """
    def distance(p, q):  # euklides
        return np.hypot(p[0]-q[0], p[1]-q[1])

    # id 11 = L shoulder, 12 = R shoulder, 23 = L hip, 24 = R hip
    shoulder_line = distance(pts[11], pts[12])
    hip_line      = distance(pts[23], pts[24])
    waist_to_hip  = hip_line / shoulder_line if shoulder_line else np.nan
    asym_shoulder = abs(pts[11][1]-pts[12][1])  # różnica wysokości barków w pikselach

    return {
        "shoulder_width_px": shoulder_line,
        "hip_width_px": hip_line,
        "waist_hip_ratio": waist_to_hip,
        "shoulder_asym_px": asym_shoulder,
    }
