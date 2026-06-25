import cv2
import os
import numpy as np
import pickle

KNOWN_FACES_DIR = "known_faces"
MODEL_PATH      = "face_model.yml"
LABELS_PATH     = "face_labels.pkl"
THRESHOLD       = 80   # LBPH confidence — lower = more similar (< 80 is a good match)

os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# ── Face detector (Haar Cascade — built into OpenCV) ─────────────────────────
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def detect_face(frame):
    """Return (face_crop, x, y, w, h) for the largest face, or None."""
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(80, 80))
    if len(faces) == 0:
        return None
    # Pick largest face
    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
    return face_crop, x, y, w, h


# ── Save enrolled face image ──────────────────────────────────────────────────
def save_face_image(name, frame):
    result = detect_face(frame)
    if result is None:
        return False, "No face detected. Please look directly at the camera."

    safe_name = name.strip().replace(" ", "_")
    path = os.path.join(KNOWN_FACES_DIR, f"{safe_name}.jpg")
    cv2.imwrite(path, frame)
    return True, path


# ── Train LBPH model from all enrolled faces ─────────────────────────────────
def train_model():
    """Re-train the LBPH recognizer from known_faces/ and save model."""
    faces, labels, label_map = [], [], {}
    label_id = 0

    for filename in os.listdir(KNOWN_FACES_DIR):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        name = os.path.splitext(filename)[0].replace("_", " ")
        path = os.path.join(KNOWN_FACES_DIR, filename)
        img  = cv2.imread(path)
        if img is None:
            continue
        result = detect_face(img)
        if result is None:
            continue
        face_crop = result[0]
        if name not in label_map.values():
            label_map[label_id] = name
            current_id = label_id
            label_id += 1
        else:
            current_id = [k for k, v in label_map.items() if v == name][0]

        faces.append(face_crop)
        labels.append(current_id)

    if not faces:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(MODEL_PATH)

    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    return recognizer, label_map


# ── Load trained model ────────────────────────────────────────────────────────
def load_known_faces():
    """Load or train LBPH model. Returns (recognizer, label_map)."""
    if not os.listdir(KNOWN_FACES_DIR):
        return None, {}

    if os.path.exists(MODEL_PATH) and os.path.exists(LABELS_PATH):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(MODEL_PATH)
        with open(LABELS_PATH, "rb") as f:
            label_map = pickle.load(f)
        return recognizer, label_map

    return train_model()


# ── Recognise faces in a frame ───────────────────────────────────────────────
def recognize_faces(frame, known_faces):
    """
    known_faces = (recognizer, label_map) tuple from load_known_faces()
    Returns list of dicts: [{name, confidence, x, y, w, h}]
    """
    recognizer, label_map = known_faces
    results = []

    if recognizer is None or not label_map:
        return results

    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
        label_id, confidence = recognizer.predict(face_crop)

        # LBPH: lower confidence = better match
        if confidence < THRESHOLD:
            name = label_map.get(label_id, "Unknown")
            confidence_pct = round((1 - confidence / THRESHOLD) * 100, 1)
        else:
            name = "Unknown"
            confidence_pct = 0.0

        results.append({
            "name":       name,
            "confidence": confidence_pct,
            "x": x, "y": y, "w": w, "h": h
        })

    return results