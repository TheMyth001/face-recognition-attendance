import torch
from torch import nn
import json

with open("student_list.json", "r", encoding="utf-8") as jsonfile:
    dictionary = json.load(jsonfile)
    class_strength = len(dictionary)

hyperparams = {
    "layer1": 64,
    "layer2": 16*class_strength,
    "activation1": nn.ReLU(),
    "activation2": nn.ReLU(),
    "train_batch_size": 32,
    "train_epochs": 10,
    "optimizer": torch.optim.AdamW,
    "learning_rate": 0.01,
    "adam_epsilon": pow(10, -8),
    "max_grad_norm": 1.0,
    "eval_batch_size": 1024,
    "save_steps": 1
}


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(512, hyperparams["layer1"])
        self.fc2 = nn.Linear(hyperparams["layer1"], hyperparams["layer2"])
        self.fc3 = nn.Linear(hyperparams["layer2"], class_strength)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = hyperparams["activation1"](self.fc1(x))
        x = hyperparams["activation1"](self.fc2(x))
        x = self.fc3(x)
        return x
