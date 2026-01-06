import cv2 
import face_recognition
import pickle
import os
import numpy as np

# Path to encodings file
ENCODINGS_PATH = "encodings/face_encodings.pickle"

# Load trained face encodings
if not os.path.exists(ENCODINGS_PATH):
    print("❌ Encodings file not found. Run encode_faces.py first.")
    exit()

with open(ENCODINGS_PATH, "rb") as f:
    data = pickle.load(f)

# Image path (put image in project root)
IMAGE_PATH = "test.jpg"

if not os.path.exists(IMAGE_PATH):
    print("❌ test.jpg not found in project folder.")
    exit()


# Load image using face_recognition (recommended method)
rgb = face_recognition.load_image_file(IMAGE_PATH)

# Also load for OpenCV display
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Failed to load image. Check if test.jpg is a valid image file.")
    exit()

# Detect faces
face_locations = face_recognition.face_locations(rgb)
face_encodings = face_recognition.face_encodings(rgb, face_locations)


# Loop through detected faces
for (top, right, bottom, left), encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces(data["encodings"], encoding)
    name = "Unknown"

    if True in matches:
        matched_idxs = [i for i, m in enumerate(matches) if m]
        counts = {}

        for i in matched_idxs:
            person_name = data["names"][i]
            counts[person_name] = counts.get(person_name, 0) + 1

        name = max(counts, key=counts.get)

    # Draw box and name
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(
        image,
        name,
        (left, top - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

# Show result
cv2.imshow("Image Face Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
