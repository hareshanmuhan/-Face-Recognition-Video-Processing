import cv2
import face_recognition
import pickle
import os
import numpy as np

with open('trained_model.pkl_2', 'rb') as f:
    known_face_encodings, known_face_names = pickle.load(f)

video_capture_input = input("Enter the path: ")
if video_capture_input == '0':
    video_capture = cv2.VideoCapture(0)
else:
    video_capture = cv2.VideoCapture(video_capture_input)
    if not video_capture.isOpened():
        print("Error: Unable to open video capture.")
        exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_video_path = 'enhanced_video.avi'
out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (480, 360))

cropped_dir = 'cropped_images'
if not os.path.exists(cropped_dir):
    os.makedirs(cropped_dir)

image_counts = {name: 0 for name in known_face_names}

frame_count = 0
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    frame_resize = cv2.resize(frame, (480, 360))

    enhanced_frame_path = f'temp_frame_{frame_count}.png'
    cv2.imwrite(enhanced_frame_path, frame_resize, [cv2.IMWRITE_JPEG_QUALITY, 95])
    enhanced_frame = cv2.imread(enhanced_frame_path)
    out.write(enhanced_frame)

    rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])

    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

            if True in matches:
                face_image = frame[top:bottom, left:right]
                person_dir = os.path.join(cropped_dir, name)
                if not os.path.exists(person_dir):
                    os.makedirs(person_dir)
                face_image_path = os.path.join(person_dir, f'{frame_count}_{image_counts[name]}.png')
                cv2.imwrite(face_image_path, face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])

                image_counts[name] += 1

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 5), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()


for i in range(frame_count):
    os.remove(f'temp_frame_{i}.png')
