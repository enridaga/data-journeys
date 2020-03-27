
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path
from PIL import Image
import random
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch
import torchvision
from tqdm import tqdm
cfg = {
    # Batch Size for Training and Varidation
        "batch_size": 1024,
    # CUDA:0 or CPU
        "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
    # Epoch Size for Training and Validation
        "epoch_size": 10,
    # Ratio of Filling Noise on Training and Validation Images
        "noise_ratio": 0.25,
    # Sigma Parameter of Gauss Deviation for Transform
        "noise_sigma": 0.1,
    # Path to Dig-MNIST.csv
        "path_Dig-MNIST_csv": Path("../input/Kannada-MNIST/Dig-MNIST.csv"),
    # Path to test.csv
        "path_test_csv": Path("../input/Kannada-MNIST/test.csv"),
    # Path to train.csv
        "path_train_csv": Path("../input/Kannada-MNIST/train.csv"),
    # Range of Degrees Rotated by RandomRotation
        "pil_trans_degree": (-10, 10),
    # Range of Aspect Ratio of the Origin Aspect Ratio Cropped by RandomResizedCrop
        "pil_trans_ratio": (0.8*0.8, 1.25*1.25),
    # Range of Size of the Origin Size Cropped by RandomResizedCrop
        "pil_trans_scale": (0.75*0.75, 1.0),
    # Random Seed
        "seed": 17122019,
    # Ratio of Training Dataset against Overall One
        "train_dataset_ratio": 0.9,
}
random.seed(cfg["seed"])
np.random.seed(cfg["seed"])
torch.manual_seed(cfg["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed(cfg["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
sns.set(style="darkgrid", context="notebook", palette="muted")
class KannadaMNISTDataset(torch.utils.data.Dataset):
    def __init__(self,
                 path_csv: Path,
                 cfg: dict,
                 transform=None):
        df_csv = pd.read_csv(path_csv)
        self.imgs = df_csv.drop(["label"], axis=1).values.astype(np.int32)
        # Reshape Image from (data_size, 784) to (data_size, 1, 28, 28)
        self.imgs = self.imgs.reshape(-1, 1, 28, 28)
        self.labels = torch.tensor(df_csv["label"],
                                   dtype=torch.int64,
                                   device=cfg["device"])
        self.transform = transform
        self.device = cfg["device"]

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.labels[idx]
        if self.transform is None:
            # Scale Image from [0, 255] to [0.0, 1.0]
            img = torch.tensor(img/255.0,
                               dtype=torch.float32,
                               device=self.device)
        else:
            img = self.transform(img)
        return img, label
class KannadaMNISTTransform():
    def __init__(self, cfg: dict):
        self.device = cfg["device"]
        self.noise_ratio = cfg["noise_ratio"]
        self.noise_sigma = cfg["noise_sigma"]
        self.pil_trans = torchvision.transforms.Compose([
            torchvision.transforms.ToPILImage(),
            torchvision.transforms.RandomResizedCrop(28,
                                                     scale=cfg["pil_trans_scale"],
                                                     ratio=cfg["pil_trans_ratio"]),
            torchvision.transforms.RandomRotation(degrees=cfg["pil_trans_degree"]),
            torchvision.transforms.ToTensor()
        ])

    def __call__(self, img: np.ndarray):
        # Add Noise on Images
        mask = np.random.random(img.shape)>self.noise_ratio
        noise = np.random.normal(0.0,
                                 self.noise_sigma,
                                 size=img.shape)
        noise[mask] = 0.0
        noise *= 255.0
        noise = noise.astype(np.int32)
        img += noise

        # Execute Pillow's Transforms
        img = self.pil_trans(img[0])

        # Scale Image from [0, 255] to [0.0, 1.0]
        img = img.to(torch.float32)/255.0

        return img.to(self.device)
def create_training_datasets(cfg: dict,
                             transform: KannadaMNISTTransform):
    # Create Overall Dataset Setting KannadaMNISTTransform
    overall_dataset = KannadaMNISTDataset(cfg["path_train_csv"], cfg, transform)
    # Split Overall Dataset into Training and Validation Ones
    train_size = int(len(overall_dataset) * cfg["train_dataset_ratio"])
    valid_size = len(overall_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(overall_dataset,
                                                                 [train_size, valid_size])
    return train_dataset, valid_dataset
%%time
# Training Datasets
train_dataset, valid_dataset = create_training_datasets(cfg,
                                                        KannadaMNISTTransform(cfg))
# Learning Dataset
lrn_dataset = KannadaMNISTDataset(cfg["path_train_csv"], cfg, None)
# Investigation Dataset
inv_dataset = KannadaMNISTDataset(cfg["path_Dig-MNIST_csv"], cfg, None)
class ThisNetwork(torch.nn.Module):
    def __init__(self):
        super(ThisNetwork, self).__init__()
        self.features = torch.nn.Sequential(
            # (batch,1,28,28) -> (batch,64,28,28)
            torch.nn.Conv2d(in_channels=1,
                            out_channels=64,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=64),
            torch.nn.ReLU(inplace=True),
            # (batch,64,28,28) -> (batch,64,14,14)
            torch.nn.MaxPool2d(kernel_size=2,
                               stride=2),
            # (batch,64,14,14) -> (batch,128,14,14)
            torch.nn.Conv2d(in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            padding=1),
            torch.nn.BatchNorm2d(num_features=128),
            torch.nn.ReLU(inplace=True),
            # (batch,128,14,14) -> (batch,128,7,7)
            torch.nn.MaxPool2d(kernel_size=2,
                               stride=2),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d(output_size=3)
        self.classifier = torch.nn.Sequential(
            # (batch,1152) -> (batch,256)
            torch.nn.Linear(in_features=1152,
                            out_features=256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            # (batch,256) -> (batch,256)
            torch.nn.Linear(in_features=256,
                            out_features=256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(p=0.5),
            # (batch,256) -> (batch,10)
            torch.nn.Linear(in_features=256,
                            out_features=10),
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # (batch,1,28,28) -> (batch,128,7,7)
        x = self.features(x)
        # (batch,128,7,7) -> (batch,128,3,3)
        x = self.avgpool(x)
        # (batch,128,3,3) -> (batch,1152)
        x = x.view(x.size(0), -1)
        # (batch,1152) -> (batch,10)
        x = self.classifier(x)
        # (batch,10) -> (batch,10)
        x = self.log_softmax(x)
        return x
network = ThisNetwork().to(cfg["device"])
def learn(network: torch.nn.Module,
          train_dataset: KannadaMNISTDataset,
          valid_dataset: KannadaMNISTDataset,
          cfg: dict):
    result = {"Epoch" : [],
              "Type" : [],
              "Average Loss" : [],
              "Accuracy" : []}
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(network.parameters())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg["batch_size"],
                                               shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=cfg["batch_size"],
                                               shuffle=True)

    # Start
    for epoch in range(1, cfg["epoch_size"]+1):
        # Training
        sum_loss = 0.0
        sum_correct = 0
        for imgs, true_labels in tqdm(train_loader):
            network.zero_grad()
            pred_probs = network(imgs)
            pred_labels = torch.argmax(pred_probs, dim=1)
            loss = criterion(pred_probs, true_labels)
            loss.backward()
            optimizer.step()
            sum_loss += loss.item() * imgs.shape[0]
            sum_correct += int(torch.sum(pred_labels == true_labels))
        ave_loss = sum_loss / len(train_dataset)
        accuracy = 100.0 * sum_correct / len(train_dataset)
        result["Epoch"].append(epoch)
        result["Type"].append("Training")
        result["Average Loss"].append(ave_loss)
        result["Accuracy"].append(accuracy)
        args = (epoch, cfg["epoch_size"], ave_loss, accuracy)
        print_str = "[Training]Epoch:%d/%d,Average Loss:%.3f,Accuracy:%.2f%%"
        print(print_str % args)

        # Validation
        sum_loss = 0.0
        sum_correct = 0
        for imgs, true_labels in tqdm(valid_loader):
            pred_probs = network(imgs)
            pred_labels = torch.argmax(pred_probs, dim=1)
            loss = criterion(pred_probs, true_labels)
            sum_loss += loss.item() * imgs.shape[0]
            sum_correct += int(torch.sum(pred_labels == true_labels))
        ave_loss = sum_loss / len(valid_dataset)
        accuracy = 100.0 * sum_correct / len(valid_dataset)
        result["Epoch"].append(epoch)
        result["Type"].append("Validation")
        result["Average Loss"].append(ave_loss)
        result["Accuracy"].append(accuracy)
        args = (epoch, cfg["epoch_size"], ave_loss, accuracy)
        print_str = "[Validation]Epoch:%d/%d,Average Loss:%.3f,Accuracy:%.2f%%"
        print(print_str % args)

    return result
%%time
result = learn(network,
               train_dataset,
               valid_dataset,
               cfg)
sns.relplot(x="Epoch",
            y="Average Loss",
            hue="Type",
            kind="line",
            data=pd.DataFrame(result))
sns.relplot(x="Epoch",
            y="Accuracy",
            hue="Type",
            kind="line",
            data=pd.DataFrame(result))
def invest(inv_dataset: KannadaMNISTDataset,
           network: torch.nn.Module,
           cfg: dict):
    inv_true_labels = np.array([])
    inv_pred_labels = np.array([])
    inv_loader = torch.utils.data.DataLoader(inv_dataset,
                                             batch_size=cfg["batch_size"])

    # Prediction
    for imgs, true_labels in tqdm(inv_loader):
        pred_probs = network(imgs)
        pred_labels = torch.argmax(pred_probs, dim=1)
        inv_true_labels = np.concatenate([inv_true_labels,
                                          true_labels.cpu().numpy()])
        inv_pred_labels = np.concatenate([inv_pred_labels,
                                          pred_labels.cpu().numpy()])
    return inv_true_labels, inv_pred_labels
%%time
inv_true_labels, inv_pred_labels = invest(inv_dataset, network, cfg)
target_str = ["Image No.%d" % num for num in range(10)]
report_str = classification_report(inv_true_labels,
                                   inv_pred_labels,
                                   target_names=target_str,
                                   digits=3)
print(report_str)
cm = pd.DataFrame(confusion_matrix(inv_true_labels, inv_pred_labels),
                  columns=np.unique(inv_true_labels),
                  index=np.unique(inv_pred_labels))
cm.index.name = "True Image No."
cm.columns.name = "Predicted Image No."
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
def show_30_imgs(dataset: KannadaMNISTDataset,
                 label: int,
                 title: str):
    # Mask Images
    mask = (dataset.labels == label).cpu()
    imgs = dataset[mask][0].cpu()

    # Show Top 30 Masked Images
    fig, ax = plt.subplots(5, 6, sharex=True, sharey=True)
    fig.suptitle(title)
    for row in range(5):
        for col in range(6):
            idx = 6 * row + col
            ax[row][col].set_xticklabels([]) 
            ax[row][col].set_yticklabels([]) 
            ax[row][col].imshow(imgs[idx][0])
show_30_imgs(lrn_dataset, 0, "[Learning] True Image No.0")
show_30_imgs(inv_dataset, 0, "[Investigation] True Image No.0")
show_30_imgs(lrn_dataset, 6, "[Learning] True Image No.6")
show_30_imgs(inv_dataset, 6, "[Investigation] True Image No.6")
def show_inv_error_30_imgs(inv_dataset: KannadaMNISTDataset,
                           true_labels: np.ndarray,
                           pred_labels: np.ndarray,
                           true_num: int,
                           pred_num: int):
    # Mask Images
    mask = (true_labels == true_num)
    mask *= (pred_labels == pred_num)
    err_pred_labels = pred_labels[mask]
    err_true_labels = true_labels[mask]
    err_imgs = inv_dataset[mask][0].cpu()

    # Show Top 30 Masked Images
    fig, ax = plt.subplots(5, 6, sharex=True, sharey=True)
    args = (true_num, pred_num)
    title = "[Investigation] True Image No.%d, Predict Image No.%d" % args
    fig.suptitle(title)
    for row in range(5):
        for col in range(6):
            idx = 6 * row + col
            ax[row][col].set_xticklabels([]) 
            ax[row][col].set_yticklabels([]) 
            ax[row][col].imshow(err_imgs[idx][0])
show_30_imgs(lrn_dataset, 7, "[Learning] True Image No.7")
show_30_imgs(lrn_dataset, 6, "[Learning] True Image No.6")
show_inv_error_30_imgs(inv_dataset,
                       inv_true_labels,
                       inv_pred_labels,
                       7,
                       6)
show_30_imgs(lrn_dataset, 1, "[Learning] True Image No.1")
show_30_imgs(lrn_dataset, 0, "[Learning] True Image No.0")
show_inv_error_30_imgs(inv_dataset,
                       inv_true_labels,
                       inv_pred_labels,
                       1,
                       0)
def test(network: torch.nn.Module,
         cfg: dict):
    labels = []
    df_csv = pd.read_csv(cfg["path_test_csv"])
    ids = df_csv["id"]
    imgs = df_csv.drop(["id"], axis=1).values.astype(np.int32)
    # Reshape Image from (data_size, 784) to (data_size, 1, 1, 28, 28)
    # Where Batch Size is 1
    imgs = imgs.reshape(-1, 1, 1, 28, 28)

    # Prediction
    for id, img in zip(tqdm(ids), imgs):
        # Scale Image from [0, 255] to [0.0, 1.0]
        img = torch.tensor(img/255.0,
                           dtype=torch.float32,
                           device=cfg["device"])
        pred_probs = network(img)
        pred_labels = torch.argmax(pred_probs, dim=1)
        labels.append(pred_labels.cpu().numpy()[0])

    result = pd.DataFrame({"id" : ids,
                           "label" : labels})
    result.to_csv("submission.csv", index=False)
%%time
test(network, cfg)