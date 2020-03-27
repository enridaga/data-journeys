
import sys
package_dir = "../input/pretrained-models/pretrained-models/pretrained-models.pytorch-master/"
sys.path.insert(0, package_dir)
import numpy as np
import pandas as pd
import torchvision
import torch.nn as nn
from tqdm import tqdm
from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import os
import pretrainedmodels

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
class RetinopathyDatasetTest(Dataset):
    def __init__(self, csv_file, transform):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/test_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = self.transform(image)
        return {'image': image}
model = pretrainedmodels.__dict__['resnet101'](pretrained=None)

model.avg_pool = nn.AdaptiveAvgPool2d(1)
model.last_linear = nn.Sequential(
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.25),
                          nn.Linear(in_features=2048, out_features=2048, bias=True),
                          nn.ReLU(),
                          nn.BatchNorm1d(2048, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                          nn.Dropout(p=0.5),
                          nn.Linear(in_features=2048, out_features=1, bias=True),
                         )
model.load_state_dict(torch.load("../input/mmmodel/model.bin"))
model = model.to(device)
for param in model.parameters():
    param.requires_grad = False

model.eval()
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = RetinopathyDatasetTest(csv_file='../input/aptos2019-blindness-detection/sample_submission.csv',
                                      transform=test_transform)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds1 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds1[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds2 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds2[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds3 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds3[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds4 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds4[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds5 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds5[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds6 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds6[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds7 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds7[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds8 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds8[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds9 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds9[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_data_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
test_preds10 = np.zeros((len(test_dataset), 1))
tk0 = tqdm(test_data_loader)
for i, x_batch in enumerate(tk0):
    x_batch = x_batch["image"]
    pred = model(x_batch.to(device))
    test_preds10[i * 32:(i + 1) * 32] = pred.detach().cpu().squeeze().numpy().ravel().reshape(-1, 1)
test_preds = (test_preds1 + test_preds2 + test_preds3 + test_preds4 + test_preds5
             + test_preds6 + test_preds7 + test_preds8 + test_preds9 + test_preds10) / 10.0
coef = [0.5, 1.5, 2.5, 3.5]

for i, pred in enumerate(test_preds):
    if pred < coef[0]:
        test_preds[i] = 0
    elif pred >= coef[0] and pred < coef[1]:
        test_preds[i] = 1
    elif pred >= coef[1] and pred < coef[2]:
        test_preds[i] = 2
    elif pred >= coef[2] and pred < coef[3]:
        test_preds[i] = 3
    else:
        test_preds[i] = 4


sample = pd.read_csv("../input/aptos2019-blindness-detection/sample_submission.csv")
sample.diagnosis = test_preds.astype(int)
sample.to_csv("submission.csv", index=False)
sample
test_preds4
