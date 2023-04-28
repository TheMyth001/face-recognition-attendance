import csv
import cv2
import json
import torch
from torch import nn
from facenet_pytorch import InceptionResnetV1
import numpy as np
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor
from model import Net, hyperparams, class_strength
from datetime import datetime


face_finder = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
pretrained_model = InceptionResnetV1(pretrained='vggface2').eval()
model = Net()
model.load_state_dict(torch.load("output/clssnn.pth"))


def inference(model, sample):
    softmax = nn.Softmax(dim=-1)
    model.eval()
    with torch.no_grad():
        inputs = sample
        logits = model(inputs)
        outputs = softmax(logits)
    preds = outputs.detach().cpu().numpy()[0]
    logits = logits.detach().cpu().numpy()[0]
    return np.argmax(preds), max(preds)


def extract_face(image):
    """Extract a face from an image and send it to get preprocessed"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_finder.detectMultiScale(gray)
    if len(faces) != 0:
        x, y, w, h = faces[0]
        face = image.copy()[y:y+h, x:x+w]
        face = cv2.resize(face, (160, 160))
        face = cv2.GaussianBlur(face, (3, 3), 0)
        facetensor = Image.fromarray(face)
        facetensor = pil_to_tensor(facetensor)
        facetensor = facetensor.reshape(1, 3, 160, 160)
        return facetensor, face
    else:
        return None


def get_roll_name(class_number):
    with open("student_list.json", "r", encoding="utf-8") as jsonfile:
        dictionary = json.load(jsonfile)
        roll = int(list(dictionary.keys())[class_number])
        name = dictionary[list(dictionary.keys())[class_number]]
    return roll, name


def mark_attendance(roll: int, name: str):
    list = []
    with open("attendance.csv", "r", encoding="utf-8", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            list.append(row)
    with open("attendance.csv", "w", encoding="utf-8", newline="") as csv_file:
        for item in list:
            print(item[0] + "," + item[1] + "," + item[2] + "," + item[3], file=csv_file)
        now = str(datetime.now())
        print(str(roll) + "," + name + "," + now[11:19] + "," + now[:10], file=csv_file)


def add_attendance():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        face = extract_face(frame)
        if face is None:
            pass
        else:
            embeddings = pretrained_model(face[0].float())
            student, confidence = inference(model, embeddings)
            if confidence >= 0.8:
                cv2.imshow("Detected Face", face[1])
                cv2.waitKey(0)
                roll, name = get_roll_name(student)
                print(student)
                print(f"student is: {name}\twith prediction confidence: {confidence}")
                print(f"Marked attendance for {name}")
                mark_attendance(roll, name)
                break
    cap.release()
    cv2.destroyAllWindows()
