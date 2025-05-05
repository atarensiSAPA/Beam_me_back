import face_recognition
import cv2
import os
from PIL import Image
import controller.validate_img as validate_img

def load_known_faces(known_path):
    known_encodings = []
    known_names = []
    for filename in os.listdir(known_path):
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'): # We can't use the function validate_img.validate_image() because we are going to use this code on azure app functions
            image_path = os.path.join(known_path, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_encodings.append(encoding[0])
                known_names.append(os.path.splitext(filename)[0])
    return known_encodings, known_names

def process_image(unkown_path, output_path, known_face_encodings, known_face_names):
    target_image = face_recognition.load_image_file(unkown_path)
    face_locations = face_recognition.face_locations(target_image)
    face_encodings = face_recognition.face_encodings(target_image, face_locations)

    if not face_locations:
        return False  # No faces found

    for i, (top, right, bottom, left) in enumerate(face_locations):
        face_encoding = face_encodings[i]
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        else:
            name = f"Unknown_{i + 1}"  # Add a unique number for each unknown face

        face_image = target_image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        final_output_path = os.path.join(output_path, f"{name}_{i}_detected.jpg")
        pil_image.save(final_output_path)
        print(f"Saved face: {final_output_path}")
        

    return True
