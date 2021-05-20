import face_recognition
import os
import cv2
import numpy as np  # Numpy module will be used for horizontal stacking of two frames

KnownFacesDir = 'known_faces'
UnknownFacesDir = 'unknown_faces'

process_this_frame = True
n = 0
tolerance = 0.57

print("loading known faces")

known_faces_encodings = []
known_faces_names = []

face_names = []
face_locations = []
face_encodings = []

video_capture = cv2.VideoCapture(0)

for file in os.listdir(f"../{KnownFacesDir}"):
    # Load pictures and learn how to recognize them
    image = face_recognition.load_image_file(f"../{KnownFacesDir}/{file}")
    known_faces_encodings.append(face_recognition.face_encodings(image)[0])
    known_faces_names.append(file[0:-4])

print("known faces loaded")
print("Video on")
print("\n\n" + "Faces found:\n" + "-" * 50)


while True:
    # Grab a single frame of video and mirror it
    ret, mirror_frame = video_capture.read()
    frame = cv2.flip(mirror_frame, 1)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame and n % 20 == 0:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            results = face_recognition.compare_faces(known_faces_encodings, face_encoding, tolerance=tolerance)
            name = "Unknown"

            for index, val in enumerate(results):
                if val:
                    name = known_faces_names[index]

            print(name)
            face_names.append(name)

    n += 1

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        if name == "Unknown":
            cv2.rectangle(frame, (left - 10, top - 10 - 20), (right + 10, bottom + 20), (0, 0, 255), 2)

            # Draw a label for a name below the face
            cv2.rectangle(frame, (left - 10, bottom + 20 - 35), (right + 10, bottom + 20), (0, 0, 255), cv2.FILLED)
        else:
            cv2.rectangle(frame, (left - 10, top - 10 - 20), (right + 10, bottom + 20), (0, 255, 0), 2)

            # Draw a label for a name below the face
            cv2.rectangle(frame, (left - 10, bottom + 20 - 35), (right + 10, bottom + 20), (0, 255, 0), cv2.FILLED)

        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left - 10 + 6, bottom + 20 - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Press "q" to quit!', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("-" * 50)
print("Completed")

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()

