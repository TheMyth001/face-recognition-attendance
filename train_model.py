from facenet_pytorch import InceptionResnetV1
from model import hyperparams, Net
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
import torch
import numpy as np
import transformers
import os
from torch import nn
from sklearn.metrics import f1_score, recall_score, precision_score
from matplotlib import pyplot as plt


def train(train_dataset, val_dataset, model):

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  sampler=train_sampler,
                                  batch_size=hyperparams["train_batch_size"]
                                  )
    train_batch_size = hyperparams["train_batch_size"]

    t_total = len(train_dataloader) * hyperparams["train_epochs"]
    optimizer = hyperparams["optimizer"](model.parameters(),
                                         lr=hyperparams["learning_rate"],
                                         eps=hyperparams["adam_epsilon"],
                                         )
    scheduler = transformers.get_cosine_schedule_with_warmup(optimizer,
                                                             num_warmup_steps=t_total // 20,
                                                             num_training_steps=t_total
                                                             )
    criterion = nn.CrossEntropyLoss()

    print("\t> Number of Train Samples: ", len(train_dataset))
    print("\t> Number of Epochs: ", hyperparams["train_epochs"])
    print("\t> Instantaneous batch size per GPU = ", train_batch_size)

    train_losses, val_losses = [], []
    train_acc, val_acc = [], []
    train_f1, val_f1 = [], []
    tr_loss, logging_loss = 0.0, 0.0

    model.zero_grad()

    global_step = 0
    best_f1_score = 0
    patience = 5
    last_best_epoch = -1

    if not os.path.exists("output"):
        os.makedirs("output")

    for epoch in range(int(hyperparams["train_epochs"])):
        print(f"Epoch {epoch+1}/{hyperparams['train_epochs']}")

        for step, batch in enumerate(train_dataloader):
            model.train()
            input_, labels_ = batch
            outputs = model(input_)
            loss = criterion(outputs, labels_.long())
            loss.backward()
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            global_step += 1

        results = evaluate(train_dataset, model)
        print("\t> Train loss: {:.4f}".format(tr_loss/global_step))
        train_losses.append(tr_loss/global_step)
        print("\t> Train accuracy: {:.2f}%".format(100*results["acc"]))
        print("\t> Train f1-score: {:.4f}".format(results["f1"]))
        train_acc.append(results["acc"])
        train_f1.append(results["f1"])

        results = evaluate(val_dataset, model)
        print("\t> Validation loss: {:.4f}".format(results["eval_loss"]))
        print("\t> Validation accuracy: {:.2f}%".format(100*results["acc"]))
        print("\t> Validation f1-score: {:.4f}".format(results["f1"]))
        val_losses.append(results["eval_loss"])
        val_acc.append(results["acc"])
        val_f1.append(results["f1"])

        if (results.get("f1") > best_f1_score) and (hyperparams["save_steps"] > 0):
            best_f1_score = results.get("f1")
            model_to_save = model.module if hasattr(model, "module") else model
            torch.save(model_to_save.state_dict(), "output/clssnn2.pth")
            torch.save(hyperparams, os.path.join("output", "training_args2.bin"))
            last_best_epoch = epoch
            print("\t> Last best epoch is {}".format(last_best_epoch))
        elif epoch - last_best_epoch > patience:
            print("Early stopped at epoch {}".format(epoch))
            break

    return train_losses, train_acc, train_f1, val_losses, val_acc, val_f1


def evaluate(val_dataset, model):
    """
    Evaluate a models performance on the given dataset.
    """
    eval_sampler = SequentialSampler(val_dataset)
    eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=hyperparams["eval_batch_size"])

    results = {}
    criterion = nn.CrossEntropyLoss()

    eval_loss = 0.0
    nb_eval_steps = 0
    predictions = None
    out_label_ids = None

    for batch in eval_dataloader:
        model.eval()
        with torch.no_grad():
            inputs, labels_ = batch
            outputs = model(inputs)
            logits = outputs
            loss = criterion(outputs, labels_.long())
            eval_loss += loss.mean().item()
        nb_eval_steps += 1

        if predictions is None:
            predictions = logits.detach().cpu().numpy()
            out_label_ids = labels_.detach().cpu().numpy()
        else:
            predictions = np.append(predictions, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, labels_.detach().cpu().numpy(), axis=0)

    eval_loss = eval_loss / nb_eval_steps
    predictions = np.argmax(predictions, axis=1)
    result = acc_and_f1(predictions, out_label_ids)
    results.update(result)
    results["eval_loss"] = eval_loss
    return results


def simple_accuracy(predictions, labels):
    return (predictions == labels).mean()


def acc_and_f1(predictions, labels):
    acc = simple_accuracy(predictions, labels)
    f1 = f1_score(y_true=labels, y_pred=predictions, average="weighted", zero_division=0)
    precision = precision_score(y_true=labels, y_pred=predictions, average="weighted", zero_division=0)
    recall = recall_score(y_true=labels, y_pred=predictions, average="weighted", zero_division=0)
    return {"acc": acc,
            "f1": f1,
            "acc_and_f1": (acc + f1) / 2,
            "precision": precision,
            "recall": recall
            }


def get_embeddings(image_dataset):
    dataloader = DataLoader(image_dataset)
    embeddings = None
    targets = None
    for image, target in dataloader:
        image = image.float()
        if embeddings is None:
            embeddings = torch.Tensor(pretrained_model(image))
            targets = torch.Tensor([target])
        else:
            embeddings = torch.cat((embeddings, pretrained_model(image)))
            targets = torch.cat((targets, torch.tensor([target])))
    print(embeddings.shape)
    print(targets.shape)
    print("="*72)
    return embeddings, targets


print("-"*72)
print("Defining Models...")
pretrained_model = InceptionResnetV1(pretrained='vggface2').eval()
for param in pretrained_model.parameters():
    param.requires_grad = False
my_model = Net()

print("-"*72)
print("Creating Dataset...")
transform = transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.1, hue=0.04),
                                transforms.RandomRotation(10),
                                transforms.PILToTensor(),
                                ])

dataset1 = datasets.ImageFolder("students", transform=transform)
dataset2 = datasets.ImageFolder("students", transform=transform)
dataset3 = datasets.ImageFolder("students", transform=transform)
dataset = ConcatDataset([dataset1, dataset2, dataset3])

print("-"*72)
print("Extracting Face Embeddings...")
images_tensor, targets_tensor = get_embeddings(dataset)

print("-"*72)
print("Preparing to pass through second model...")
dataset = TensorDataset(images_tensor, targets_tensor)
train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])

print("-"*72)
print("Training started...")
train_loss_list, train_acc_list, train_f1_list, val_loss_list, val_acc_list, val_f1_list\
    = train(train_set, test_set, my_model)
print("-"*72)
print(train_loss_list, train_acc_list, train_f1_list, val_loss_list, val_acc_list, val_f1_list, sep="\n")

plt.plot(val_acc_list)
plt.show()
plt.plot(train_acc_list)
plt.show()
plt.plot(val_f1_list)
plt.show()
plt.plot(train_f1_list)
plt.show()
