import face_recognition
import cv2 
import pickle

# Load encodings
with open("encodings/face_encodings.pickle", "rb") as f:
    data = pickle.load(f)

# Image to recognize
image_path = "test.jpg"   # put an image in project root
image = cv2.imread(image_path)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, name, (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

cv2.imshow("Image Recognition", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
