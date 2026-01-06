import cv2
import csv
from datetime import datetime
import os

print("=" * 60)
print("📋 ATTENDANCE SYSTEM - FACE DETECTION")
print("=" * 60)
print("[INFO] Starting webcam...")
print("[INFO] Press 's' to save attendance")
print("[INFO] Press 'q' to quit")
print("=" * 60)

# Load OpenCV's pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam")
    exit()

# Attendance file
ATTENDANCE_FILE = "attendance.csv"

# Create attendance file if it doesn't exist
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'Faces Detected', 'Status'])

attendance_saved = False

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
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"Person {i+1}",
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
        "Press 's' to save attendance | 'q' to quit",
        (10, frame.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1
    )
    
    # Show saved message if attendance was just saved
    if attendance_saved:
        cv2.putText(
            frame,
            "ATTENDANCE SAVED!",
            (frame.shape[1]//2 - 150, frame.shape[0]//2),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0),
            3
        )
    
    # Show the frame
    cv2.imshow('Attendance System - Face Detection', frame)
    
    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        print("\n[INFO] Quitting...")
        break
    elif key == ord('s'):
        # Save attendance
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(ATTENDANCE_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, len(faces), 'Present' if len(faces) > 0 else 'Absent'])
        
        print(f"[SUCCESS] Attendance saved! Faces detected: {len(faces)}")
        attendance_saved = True
    else:
        attendance_saved = False

# Cleanup
cap.release()
cv2.destroyAllWindows()
print(f"[SUCCESS] Attendance log saved to: {ATTENDANCE_FILE}")
print("[SUCCESS] Webcam closed successfully!")
