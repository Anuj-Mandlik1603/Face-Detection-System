import face_recognition
import numpy as np

print("Testing face recognition image loading...")

try:
    # Try loading the image
    image = face_recognition.load_image_file("test.jpg")
    print(f"Image loaded successfully!")
    print(f"Image shape: {image.shape}")
    print(f"Image dtype: {image.dtype}")
    print(f"Image type: {type(image)}")
    
    # Try detecting faces
    print("\nTrying to detect faces...")
    face_locations = face_recognition.face_locations(image)
    print(f"Found {len(face_locations)} face(s)")
    
    if len(face_locations) > 0:
        print("Face locations:", face_locations)
        face_encodings = face_recognition.face_encodings(image, face_locations)
        print(f"Successfully encoded {len(face_encodings)} face(s)")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
