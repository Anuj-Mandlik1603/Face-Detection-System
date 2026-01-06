import os
import pickle
import cv2 
import face_recognition
from datetime import datetime
import csv

# -----------------------------
# PATH HELPERS
# -----------------------------

ENCODINGS_PATH = "encodings/face_encodings.pickle"
ATTENDANCE_FILE = "attendance.csv"


# -----------------------------
# LOAD FACE ENCODINGS
# -----------------------------

def load_encodings():
    """
    Load face encodings from pickle file
    """
    if not os.path.exists(ENCODINGS_PATH):
        raise FileNotFoundError("❌ face_encodings.pickle not found. Run encode_faces.py first.")

    with open(ENCODINGS_PATH, "rb") as f:
        data = pickle.load(f)

    return data


# -----------------------------
# FACE RECOGNITION LOGIC
# -----------------------------

def recognize_face(known_data, face_encoding):
    """
    Compare a face encoding with known encodings
    Returns name or 'Unknown'
    """
    matches = face_recognition.compare_faces(known_data["encodings"], face_encoding)
    name = "Unknown"

    if True in matches:
        matched_idxs = [i for i, m in enumerate(matches) if m]
        counts = {}

        for i in matched_idxs:
            person = known_data["names"][i]
            counts[person] = counts.get(person, 0) + 1

        name = max(counts, key=counts.get)

    return name


# -----------------------------
# DRAW FACE BOX & NAME
# -----------------------------

def draw_face_box(frame, box, name):
    """
    Draw rectangle and name on frame
    """
    top, right, bottom, left = box
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(
        frame,
        name,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )


# -----------------------------
# ATTENDANCE FUNCTIONS
# -----------------------------

def init_attendance_file():
    """
    Create attendance CSV file if not exists
    """
    if not os.path.exists(ATTENDANCE_FILE):
        with open(ATTENDANCE_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Date", "Time"])


def mark_attendance(name):
    """
    Mark attendance once per day per person
    """
    init_attendance_file()

    today = datetime.now().strftime("%Y-%m-%d")

    with open(ATTENDANCE_FILE, "r", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    for row in rows[1:]:
        if row[0] == name and row[1] == today:
            return  # already marked

    now = datetime.now().strftime("%H:%M:%S")
    with open(ATTENDANCE_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now])
