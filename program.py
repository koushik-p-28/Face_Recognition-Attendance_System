import face_recognition
import cv2
import numpy as np
import csv
import os
from datetime import datetime

# Safely access the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Unable to access the camera.")
    exit()

# Load images and encode
def load_and_encode(path):
    image = face_recognition.load_image_file(path)
    encoding = face_recognition.face_encodings(image)
    if encoding:
        return encoding[0]
    else:
        raise Exception(f"No face found in image: {path}")

known_face_encodings = [
    load_and_encode(r"C:\Users\palak\OneDrive\Documents\Python\Student attandence\photos\jobs.png"),
    load_and_encode(r"C:\python\Student attandence\photos\stata.png"),
    load_and_encode(r"C:\python\Student attandence\photos\sadmona.png"),
    load_and_encode(r"C:\python\Student attandence\photos\stesla.png")
]

known_faces_names = [
    "jobs",
    "ratan tata",
    "sadmona",
    "tesla"
]

students = known_faces_names.copy()

# Prepare CSV file for today's attendance
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")
f = open(current_date + '.csv', 'w+', newline='')
lnwriter = csv.writer(f)

# Main loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read from the camera.")
        break

    # Resize frame for faster recognition
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces and compute encodings
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = ""

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)

        if matches[best_match_index]:
            name = known_faces_names[best_match_index]

        face_names.append(name)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H:%M:%S")
            lnwriter.writerow([name, current_time])
            print(f"{name} marked present at {current_time}")

    # Display the frame
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up since the frame we used was scaled down
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a rectangle and name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    cv2.imshow("Attendance System", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
f.close()
