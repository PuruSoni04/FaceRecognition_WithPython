import face_recognition
import os
import cv2
import keyword


KnownFacesDir = 'known_faces'
UnknownFacesDir = 'unknown_faces'
UnknownImage = "5.jpg"

tolerance = 0.6
frame_thickness = 3
font_thickness = 2
MODEL = "cnn"

print("loading known faces")

known_faces_encodings = []
known_faces_names = []

for file in os.listdir(KnownFacesDir):
    # Load a second picture and learn how to recognize it.
    image = face_recognition.load_image_file(f"{KnownFacesDir}/{file}")
    known_faces_encodings.append(face_recognition.face_encodings(image)[0])
    known_faces_names.append(file[0:-4])

print("known faces loaded")
print("loading unknown faces")

u_image = face_recognition.load_image_file(f"{UnknownFacesDir}/{UnknownImage}")
u_face_locations = face_recognition.face_locations(u_image)
u_face_encodings = face_recognition.face_encodings(u_image, u_face_locations)

print("unknown faces loaded")
print("comparing faces")

print("\n\n"+"-"*50)

for encoding in u_face_encodings:
    # results is an array of True/False telling if the unknown face matched anyone in the known_faces array
    results = face_recognition.compare_faces(known_faces_encodings, encoding)

    for index, val in enumerate(results):
        if val:
            print(f"Match found: {known_faces_names[index]}")

print("-"*50+"\n\n")
print('Completed')













