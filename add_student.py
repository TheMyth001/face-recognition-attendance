from facenet_pytorch import MTCNN
import os
import cv2
import json


def add_student():
    name = input("What is your name? ")
    roll = int(input("What is your roll number? "))
    if add_student_data(name, roll):
        capture_images(roll, name)
    else:
        print("Roll Number already exists...")


def capture_images(roll, name):
    """
    Capture images, and save them to the required folder
    """
    try:
        os.mkdir(os.path.join("students/", f"{str(roll).zfill(2)}-{name}"))
    finally:
        pass
    cap = cv2.VideoCapture(0)
    i = 0
    images = 0
    while images < 100:
        ret, frame = cap.read()
        face, box = get_face(frame)
        if face is not None:
            i += 1
            x0, y0, x1, y1 = box
            cv2.rectangle(frame, (x0, y0), (x1, y1), color=(0, 255, 0), thickness=3)
            if i % 5 == 0:
                cv2.imwrite("students/"+str(roll).zfill(2)+"-"+name+"/"+str(i/5)+".jpg", face)
                images += 1
        frame = cv2.putText(frame,
                            f"capturing photos: {images}/{100}",
                            (5, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 0),
                            2,
                            cv2.LINE_AA)
        cv2.imshow("Capturing Images", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


def get_face(image):
    """
    Get the most prominent face in the frame and the coordinates of its bounding box.
    Also resize the face and apply blur.
    If no face is present, returns None, None.
    """
    mtcnn = MTCNN()
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    face = None
    x0 = y0 = x1 = y1 = None
    if boxes is not None:
        try:
            main_face_box = boxes[0]
            x0, y0, x1, y1 = main_face_box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
            face = image.copy()[y0:y1, x0:x1]
            face = cv2.resize(face, (224, 224))
        finally:
            pass
    return face, tuple([x0, y0, x1, y1])


def add_student_data(name, roll):
    """
    If student exists, returns `None`
    Else adds their data to `student_list.json`
    """
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
