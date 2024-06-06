import face_recognition
import os
import pickle

def load_images_and_encode(image_dir):
    known_face_encodings = []
    known_face_names = []
    for person_name in os.listdir(image_dir):
        person_dir = os.path.join(image_dir, person_name)
        if os.path.isdir(person_dir):
            for filename in os.listdir(person_dir):
                if filename.endswith(".jpg") or filename.endswith(".png"):
                    img_path = os.path.join(person_dir, filename)
                    print("Processing:", img_path)
                    image = face_recognition.load_image_file(img_path)
                    face_encoding = face_recognition.face_encodings(image)
                    print("Face encoding:", face_encoding)
                    if face_encoding:
                        known_face_encodings.append(face_encoding[0])
                        known_face_names.append(person_name)
    return known_face_encodings, known_face_names



if __name__ == "__main__":
    image_dir = input("Enter your image or directory: ")
    known_face_encodings, known_face_names = load_images_and_encode(image_dir)
    with open("trained_model.pkl_2", "wb") as f:
        pickle.dump((known_face_encodings, known_face_names), f)
    print("Model trained")

