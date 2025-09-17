"""Face and person detection utilities (OpenCV-based)."""
from __future__ import annotations
from typing import List, Tuple
import cv2
import numpy as np


def _load_face_cascade():
    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(face_xml)


def detect_faces(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) face boxes in BGR image."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    face_cascade = _load_face_cascade()
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_people(image_bgr: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """Return list of (x, y, w, h) person boxes using HOG + SVM people detector."""
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    rects, _ = hog.detectMultiScale(image_bgr, winStride=(8, 8), padding=(8, 8), scale=1.05)
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]


def detect_faces_and_people(pil_image) -> tuple[List[Tuple[int,int,int,int]], List[Tuple[int,int,int,int]]]:
    """Detect faces and people from a PIL Image; returns (faces, persons)."""
    image_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    faces = detect_faces(image_bgr)
    people = detect_people(image_bgr)
    return faces, people
