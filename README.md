# -Face-Recognition-Video-Processing
This project performs face recognition on a video input stream, detecting and recognizing faces in real-time. It utilizes the face_recognition library for face detection and recognition and OpenCV for video processing.
Imports: The necessary libraries are imported, including OpenCV (cv2), face_recognition, pickle, os, and numpy.

Load Trained Model: The script loads a pre-trained face recognition model from a pickle file (trained_model.pkl_2) containing known face encodings and corresponding names.

Video Capture: The user is prompted to enter the path of the video file or '0' to capture video from the webcam. If the specified path is invalid, an error message is displayed, and the script exits.

Output Video Configuration: The script sets up an output video file to save the processed frames.

Cropped Images Directory: It creates a directory named cropped_images to store cropped images of detected faces.

Face Recognition Loop: The script continuously captures frames from the video input stream, resizes each frame, saves the enhanced frame to a temporary file, and then performs face recognition on the frame.

Face Detection and Recognition: For each frame, it detects faces using the face_recognition library, compares the detected face encodings with the known face encodings, and draws rectangles around the detected faces. If a match is found, it saves the cropped face image to the corresponding directory within cropped_images.

Displaying and Saving Results: The script displays the processed frame with bounding boxes and recognized names. It also writes the processed frames to the output video file.

Cleanup: After processing all frames, the temporary enhanced frame files are deleted.

User Termination: The script terminates when the user presses the 'q' key.
