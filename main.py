import face_recognition
import cv2
import numpy as np
from datetime import datetime
import csv

# Initialize webcam
video_capture = cv2.VideoCapture(0)  # Use 0 for default webcam

# Load known faces
Monika_image = face_recognition.load_image_file("faces/PHOTOGRAPH_MONIKA.jpg")
Monika_encoding = face_recognition.face_encodings(Monika_image)[0]

Saif_image = face_recognition.load_image_file("faces/Saif.jpg")
Saif_encoding = face_recognition.face_encodings(Saif_image)[0]

known_face_encoding = [Monika_encoding, Saif_encoding]
known_face_names = ["Monika", "Saif"]

# List of expected studentszzxx
students = known_face_names.copy()
marked_students = set()  # Track already marked students

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)
lnwriter.writerow(["Name", "Time"])  # CSV Header

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)

    if face_locations:
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    else:
        face_encodings = []

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encoding, face_encoding)
        face_distances = face_recognition.face_distance(known_face_encoding, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

        name = "Unknown"

        if best_match_index != -1 and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Determine if already marked
        if name in marked_students:
            display_text = f"{name} - Already Marked"
        else:
            display_text = f"{name} - Present"
            if name in students:
                students.remove(name)
                marked_students.add(name)
                current_time = now.strftime("%H:%M:%S")
                lnwriter.writerow([name, current_time])

        # Display text on screen
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 100)
        fontScale = 1.5
        fontColor = (0, 255, 0) if name in marked_students else (255, 0, 0)
        thickness = 3
        lineType = 2

        cv2.putText(frame, display_text, bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()