import torch
from torch import nn
import json


with open("student_list.json", "r", encoding="utf-8") as jsonfile:
    student_dict = json.load(jsonfile)
    class_strength = len(student_dict)


hyperparams = {
    "layer1": 512,
    "layer2": 32*class_strength,
    "layer3": 256,
    "activation1": nn.ReLU(),
    "activation2": nn.ReLU(),
    "activation3": nn.ReLU(),
    "train_batch_size": 64,
    "train_epochs": 10*class_strength,
    "optimizer": torch.optim.AdamW,
    "learning_rate": 0.01,
    "adam_epsilon": pow(10, -8),
    "eval_batch_size": 1024,
    "save_steps": 1,
}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, hyperparams["layer1"])
        self.fc2 = nn.Linear(hyperparams["layer1"], hyperparams["layer2"])
        self.fc3 = nn.Linear(hyperparams["layer2"], hyperparams["layer3"])
        self.fc4 = nn.Linear(hyperparams["layer3"], class_strength)

    def forward(self, x):
        x = hyperparams["activation1"](self.fc1(x))
        x = hyperparams["activation2"](self.fc2(x))
        x = hyperparams["activation3"](self.fc3(x))
        x = hyperparams["activation3"](self.fc4(x))
        return x
