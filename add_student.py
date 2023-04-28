import cv2
import os
import json

images_per_person = 100
face_finder = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def extract_face(image, i, roll, name):
    """Extract a face from an image and send it to get preprocessed"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_finder.detectMultiScale(gray)

    if len(faces) != 0:
        x, y, w, h = faces[0]
        face = image.copy()[y:y+h, x:x+w]
        cv2.rectangle(image, (x, y), (x+w, y+h), color=(0, 255, 0), thickness=3)
        face = cv2.resize(face, (160, 160))
        face = cv2.GaussianBlur(face, (3, 3), 0)
        cv2.imwrite("students/"+str(roll).zfill(2)+"-"+name+"/"+str(i)+".jpg", face)
        return i+1
    return i


def add_student_data(name, roll):
    with open("student_list.json", "r", encoding="utf-8") as jsonfile:
        student_dict = json.load(jsonfile)
    with open("student_list.json", "w", encoding="utf-8") as jsonfile:
        if student_dict.get(str(roll).zfill(2)) is None:
            student_dict[str(roll).zfill(2)] = name
            keys = list(student_dict.keys())
            keys.sort()
            student_dict_sorted = {i: student_dict[i] for i in keys}
            json.dump(student_dict_sorted, jsonfile)
            return True
        else:
            return False


def add_student_images(roll, name):
    os.mkdir(os.path.join("students/", f"{str(roll).zfill(2)}-{name}"))
    cap = cv2.VideoCapture(0)
    i = 0
    while i <= images_per_person:
        ret, frame = cap.read()
        i = extract_face(frame, i, roll, name)
        frame = cv2.putText(frame,
                            f"capturing photos: {i}/{images_per_person}",
                            (5, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def add_student():
    name = input("What is your name?")
    roll = int(input("What is your roll number?"))
    if add_student_data(name, roll):
        add_student_images(roll, name)
    else:
        print("Roll Number already exists...")
