import cv2

print("=" * 60)
print("🎥 REAL-TIME FACE DETECTION USING OPENCV")
print("=" * 60)
print("[INFO] Starting webcam...")
print("[INFO] Press 'q' to quit")
print("=" * 60)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

face_count = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Error: Failed to grab frame")
        break
    
    # Convert to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Face {face_count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    
    # Display face count
    cv2.putText(
        frame,
        f"Faces Detected: {len(faces)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )
    
    # Display instructions
    cv2.putText(
        frame,
        "Press 'q' to quit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Show the frame
    cv2.imshow('Real-Time Face Detection', frame)
    
    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\n[INFO] Quitting...")
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
print("[SUCCESS] Webcam closed successfully!")
