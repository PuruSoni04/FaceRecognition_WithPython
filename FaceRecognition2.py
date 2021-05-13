import face_recognition
import os
import cv2

KnownFacesDir = 'known_faces'
UnknownFacesDir = 'unknown_faces'

tolerance = 0.5

print("loading known faces")

known_faces_encodings = []
known_faces_names = []

for file in os.listdir(KnownFacesDir):
    # Load a second picture and learn how to recognize it.
    image = face_recognition.load_image_file(f"{KnownFacesDir}/{file}")
    known_faces_encodings.append(face_recognition.face_encodings(image)[0])
    known_faces_names.append(file[0:-4])

print("known faces loaded")

for UnknownImage in os.listdir(f"{UnknownFacesDir}"):

    cv2_image = cv2.imread(f"{UnknownFacesDir}/{UnknownImage}")

    u_image = face_recognition.load_image_file(f"{UnknownFacesDir}/{UnknownImage}")
    u_face_locations = face_recognition.face_locations(u_image)
    u_face_encodings = face_recognition.face_encodings(u_image, u_face_locations)

    print("\n\n" + "-" * 50)

    for (top, right, bottom, left), encoding in zip(u_face_locations, u_face_encodings):
        results = face_recognition.compare_faces(known_faces_encodings, encoding, tolerance=tolerance)

        name = "Unknown"
        for index, val in enumerate(results):
            if val:
                name = known_faces_names[index]
        print(name)

        # Draw a box around the face
        cv2.rectangle(cv2_image, (left - 20, top - 20), (right + 20, bottom + 40), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(cv2_image, (left - 20, bottom + 40 - 25), (right + 20, bottom + 40), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(cv2_image, name, (left - 20 + 6, bottom + 40 - 6), font, 0.4, (255, 255, 255), 1)

    print("-" * 50 + "\n\n")

    cv2.imshow(UnknownImage, cv2_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print('Completed')
