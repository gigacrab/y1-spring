import face_recognition
import cv2
import numpy as np

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.


# Load a sample picture and learn how to recognize it.
bryan_image = face_recognition.load_image_file("./Project-Week-3/extra-task/faces/bryan.jpeg")
bryan_face_encoding = face_recognition.face_encodings(bryan_image)[0]

# Load a second sample picture and learn how to recognize it.
jayden_image = face_recognition.load_image_file("./Project-Week-3/extra-task/faces/jayden.jpeg")
jayden_face_encoding = face_recognition.face_encodings(jayden_image)[0]

hermawan_image = face_recognition.load_image_file("./Project-Week-3/extra-task/faces/dr-hermawan.jpeg")
hermawan_face_encoding = face_recognition.face_encodings(jayden_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    bryan_face_encoding,
    jayden_face_encoding,
    hermawan_face_encoding
]
known_face_names = [
    "Bryan",
    "Jayden",
    "Dr Hermawan"
]

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

def recognize_face(frame):
    
    global face_locations, face_encodings, face_names
    # Only process every other frame of video to save time

    # Resize frame of video to 1/4 size for faster face recognition processing
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"

        

        # If a match was found in known_face_encodings, just use the first one.
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        # face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        # best_match_index = np.argmin(face_distances)
        # if matches[best_match_index]:
        #     name = known_face_names[best_match_index]

        face_names.append(name)
    
    print(face_names)
    print(face_locations)

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    # Display the resulting image
    cv2.imshow('Facial Recognition', frame)

    if cv2.waitKey(1) == ord('q'):
        return True
    else:
        return False
