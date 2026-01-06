import cv2
import pickle
import os

# Path to encodings file
ENCODINGS_PATH = "encodings/face_encodings.pickle"

# Load trained face names (we'll use simple template matching since face_recognition has issues)
if not os.path.exists(ENCODINGS_PATH):
    print("❌ Encodings file not found. This is a simplified version using OpenCV.")
    print("⚠️  Note: This version does face DETECTION, not recognition.")

# Image path
IMAGE_PATH = "test.jpg"

if not os.path.exists(IMAGE_PATH):
    print("❌ test.jpg not found in project folder.")
    exit()

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load image
print("[INFO] Loading image...")
image = cv2.imread(IMAGE_PATH)
if image is None:
    print("❌ Failed to load image.")
    exit()

# Convert to grayscale for detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
print("[INFO] Detecting faces...")
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

print(f"[SUCCESS] Found {len(faces)} face(s)!")

# Draw rectangles around faces
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(
        image,
        "Face Detected",
        (x, y - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

# Show result
cv2.imshow("Face Detection (OpenCV)", image)
print("[INFO] Press any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("[SUCCESS] Done!")
