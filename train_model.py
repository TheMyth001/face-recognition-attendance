import torch
from facenet_pytorch import InceptionResnetV1
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader, random_split, RandomSampler, SequentialSampler, TensorDataset, ConcatDataset
from torchvision import transforms
import transformers
from tqdm import trange, tqdm
import os
import numpy as np
from sklearn.metrics import f1_score, recall_score, precision_score
from model import hyperparams, Net, class_strength


def retrain():
    pretrained_model = InceptionResnetV1(pretrained='vggface2').eval()
    for param in pretrained_model.parameters():
        param.requires_grad = False

    transform = transforms.Compose([transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
                                    transforms.PILToTensor()
                                    ])

    dataset0 = datasets.ImageFolder("students", transform=transform)
    dataset1 = datasets.ImageFolder("students", transform=transform)
    dataset = ConcatDataset([dataset0, dataset1])


    def get_embeddings(dataset):
        print("="*72)
        print("Getting face embeddings...")
        dataloader = DataLoader(dataset)
        embeddings = None
        targets = None
        for image, target in dataloader:
            # print(image.shape)
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


    images, targets = get_embeddings(dataset)
    dataset = TensorDataset(images, targets)
    train_set, test_set = random_split(dataset, [int(len(dataset) * 0.8), len(dataset) - int(len(dataset) * 0.8)])


    class Net(nn.Module):

        def __init__(self):
            super(Net, self).__init__()
            self.fc1 = nn.Linear(512, hyperparams["layer1"])
            self.fc2 = nn.Linear(hyperparams["layer1"], hyperparams["layer2"])
            self.fc3 = nn.Linear(hyperparams["layer2"], class_strength)

        def forward(self, x):
            x = hyperparams["activation1"](self.fc1(x))
            x = hyperparams["activation1"](self.fc2(x))
            x = self.fc3(x)
            return x


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
                                             eps=hyperparams["adam_epsilon"]
                                             )
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=t_total // 10,
                                                                 num_training_steps=t_total
                                                                 )
        criterion = nn.CrossEntropyLoss()

        print("Running training...")
        print("Num examples = ", len(train_dataset))
        print("Num Epochs = ", hyperparams["train_epochs"])
        print("Instantaneous batch size per GPU = ", train_batch_size)

        global_step = 0
        train_losses, val_losses = [], []

        train_acc, val_acc = [], []
        tr_loss, logging_loss = 0.0, 0.0
        model.zero_grad()

        train_iterator = trange(int(hyperparams["train_epochs"]), desc="Epoch")

        best_f1_score = 0
        if not os.path.exists("output"):
            os.makedirs("output")

        patience = 3
        last_best_epoch = -1

        for epoch in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")

            for step, batch in enumerate(epoch_iterator):
                model.train()
                input_, labels_ = batch
                outputs = model(input_)

                loss = criterion(outputs, labels_.long())
                loss.backward()

                nn.utils.clip_grad_norm_(model.parameters(), hyperparams["max_grad_norm"])

                tr_loss += loss.item()
                optimizer.step()
                scheduler.step()
                model.zero_grad()
                optimizer.zero_grad()
                global_step += 1

            print("Train loss: {}".format(tr_loss/global_step))
            train_losses.append(tr_loss/global_step)

            print("Train accuracy stats: ")
            results = evaluate(train_dataset, model)
            print("Train accuracy: {}".format(results["acc"]))
            train_acc.append(results["acc"])

            results = evaluate(val_dataset, model)
            print("Validation accuracy: {}".format(results["acc"]))
            print("Validation loss: {}".format(results["eval_loss"]))
            val_losses.append(results["eval_loss"])
            val_acc.append(results["acc"])

            if (results.get("f1") > best_f1_score) and (hyperparams["save_steps"] > 0):
                best_f1_score = results.get("f1")
                model_to_save = model.module if hasattr(model, "module") else model
                torch.save(model_to_save.state_dict(), "output/clssnn.pth")
                torch.save(hyperparams, os.path.join("output", "training_args.bin"))
                last_best_epoch = epoch
                print("Last best epoch is {}".format(last_best_epoch))
            elif epoch - last_best_epoch > patience:
                print("Early stopped at epoch {}".format(epoch))
                break
        return train_losses, train_acc, val_losses, val_acc


    def evaluate(val_dataset, model):
        eval_sampler = SequentialSampler(val_dataset)
        eval_dataloader = DataLoader(val_dataset, sampler=eval_sampler, batch_size=hyperparams["eval_batch_size"])

        results = {}
        criterion = nn.CrossEntropyLoss()
        print("Num examples = ", len(val_dataset))
        print("Batch size = ", hyperparams["eval_batch_size"])

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            with torch.no_grad():
                inputs, labels_ = batch
                outputs = model(inputs)
                logits = outputs
                loss = criterion(outputs, labels_.long())
                eval_loss += loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = labels_.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, labels_.detach().cpu().numpy(), axis = 0)

        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        result = acc_and_f1(preds, out_label_ids)
        results.update(result)
        results["eval_loss"] = eval_loss
        return results


    def simple_accuracy(preds, labels):
        return (preds == labels).mean()


    def acc_and_f1(preds, labels):
        acc = simple_accuracy(preds, labels)
        f1 = f1_score(y_true=labels, y_pred=preds, average="weighted", zero_division=0)
        precision = precision_score(y_true=labels, y_pred=preds, average="weighted", zero_division=0)
        recall = recall_score(y_true=labels, y_pred=preds, average="weighted", zero_division=0)
        return {"acc": acc,
                "f1": f1,
                "acc_and_f1": (acc + f1) / 2,
                "precision": precision,
                "recall": recall
                }


    my_model = Net()
    train_losses, train_acc, val_losses, val_acc = train(train_set, test_set, my_model)
