from model import Net
import csv
import cv2
import json
import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from datetime import datetime


def inference(model, sample):
    """
    Returns the predicted class along with the confidence.
    """
    softmax = nn.Softmax(dim=-1)
    model.eval()
    with torch.no_grad():
        inputs = sample
        logits = model(inputs)
        outputs = softmax(logits)
    predictions = outputs.detach().cpu().numpy()[0]
    return np.argmax(predictions), max(predictions)


def get_face(image):
    """
    Get the most prominent face in the frame and the coordinates of its bounding box.
    Also resize the face and apply blur.
    If no face is present, returns None, None.
    """
    mtcnn = MTCNN()
    boxes, probs, points = mtcnn.detect(image, landmarks=True)
    if boxes is None:
        face = None
        x0 = y0 = x1 = y1 = None
    else:
        main_face_box = boxes[0]
        x0, y0, x1, y1 = main_face_box
        x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
        face = image.copy()[y0:y1, x0:x1]
        face = cv2.resize(face, (160, 160))
    return face, tuple([x0, y0, x1, y1])


def get_roll_name(class_number):
    """
    Get the roll number and name of the student,
    based on the class predicted by our model
    """
    with open("student_list.json", "r", encoding="utf-8") as jsonfile:
        dictionary = json.load(jsonfile)
        roll = int(list(dictionary.keys())[class_number])
        name = dictionary[list(dictionary.keys())[class_number]]
    return roll, name


def mark_attendance(roll: int, name: str):
    """
    Append your attendance to attendance.csv
    """
    attendance_list = []
    with open("attendance.csv", "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            attendance_list.append(row)
    with open("attendance.csv", "w", encoding="utf-8", newline="") as csv_file:
        for item in attendance_list:
            print(item[0] + "," + item[1] + "," + item[2] + "," + item[3], file=csv_file)
        now = str(datetime.now())
        print(str(roll) + "," + name + "," + now[11:19] + "," + now[:10], file=csv_file)


def add_attendance():
    """
    The whole process of marking your attendance
    """
    pretrained_model = InceptionResnetV1(pretrained='vggface2').eval()
    my_model = Net()
    my_model.load_state_dict(torch.load("output/clssnn.pth"))
    cap = cv2.VideoCapture(0)
    recognized = False
    for i in range(100):
        ret, frame = cap.read()
        face, box = get_face(frame)
        if face is None:
            pass
        else:
            face_tensor = torch.from_numpy(face)
            face_tensor = torch.permute(face_tensor, (2, 0, 1))
            face_tensor = face_tensor.reshape(1, 3, 160, 160)
            embeddings = pretrained_model(face_tensor.float())
            student, confidence = inference(my_model, embeddings)
            if confidence >= 0.95:
                recognized = True
                cv2.imshow("Detected Face", face)
                cv2.waitKey(0)
                roll, name = get_roll_name(student)
                print(f"student: {name} | prediction confidence: {confidence*100:.2f}")
                print(f"Marked attendance for {name}")
                mark_attendance(roll, name)
                break
    if not recognized:
        print("Could not recognize face with enough confidence...")
    cap.release()
    cv2.destroyAllWindows()


def delete_attendance():
    """
    Convenience function to clear all previous attendances
    """
    with open("attendance.csv", "w", encoding="utf-8", newline="") as csv_file:
        print("roll,name,time,date", file=csv_file)


def view_attendance():
    """
    View previous attendances
    """
    attendance_list = []
    with open("attendance.csv", "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            attendance_list.append(row)
    for item in attendance_list:
        if item[0] == "roll":
            print(f"     {item[3]}\t\t  {item[2]}\t    | \t{item[0]}")
            print("-"*38)
        else:
            print(f"  {item[3]}\t{item[2]}\t| \t {str(item[0]).zfill(2)}")
