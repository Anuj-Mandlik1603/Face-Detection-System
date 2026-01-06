import cv2 
import face_recognition
import pickle
import csv
import os
from datetime import datetime

# Load face encodings
with open("encodings/face_encodings.pickle", "rb") as f:
    data = pickle.load(f)

attendance_file = "attendance.csv"

# Create attendance file if not exists
if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")

    with open(attendance_file, "r", newline="") as f:
        reader = csv.reader(f)
        entries = list(reader)

    for row in entries[1:]:
        if row[0] == name and row[1] == today:
            return  # already marked today

    now = datetime.now()
    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, today, now.strftime("%H:%M:%S")])

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("[INFO] Attendance system started. Press Q to exit.")

while True:
    ret, frame = video.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, boxes)

    for (top, right, bottom, left), encoding in zip(boxes, encodings):
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"

        if True in matches:
            matched_idxs = [i for i, m in enumerate(matches) if m]
            counts = {}

            for i in matched_idxs:
                person = data["names"][i]
                counts[person] = counts.get(person, 0) + 1

            name = max(counts, key=counts.get)
            mark_attendance(name)

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()
