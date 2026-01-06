import face_recognition 
import os
import pickle

# Path to dataset folder
DATASET_DIR = "dataset"
ENCODINGS_DIR = "encodings"
ENCODINGS_FILE = os.path.join(ENCODINGS_DIR, "face_encodings.pickle")

known_encodings = []
known_names = []

print("[INFO] Starting face encoding...")

# Loop through each person folder
for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing person: {person_name}")

    # Loop through each image of the person
    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        print(f"  -> Encoding image: {image_name}")

        image = face_recognition.load_image_file(image_path)
        face_locations = face_recognition.face_locations(image)
        face_encodings = face_recognition.face_encodings(image, face_locations)

        # Ensure exactly one face is present
        if len(face_encodings) == 1:
            known_encodings.append(face_encodings[0])
            known_names.append(person_name)
        else:
            print(f"  ⚠ Skipped {image_name} (No face or multiple faces detected)")

# Save encodings
os.makedirs(ENCODINGS_DIR, exist_ok=True)

data = {
    "encodings": known_encodings,
    "names": known_names
}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("[SUCCESS] Face encoding completed!")
print(f"[INFO] Total faces encoded: {len(known_encodings)}")
