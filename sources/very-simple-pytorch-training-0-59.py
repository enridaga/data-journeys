
import pandas as pd

import time
import torchvision
import torch.nn as nn
from tqdm import tqdm_notebook as tqdm

from PIL import Image, ImageFile
from torch.utils.data import Dataset
import torch
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os

device = torch.device("cuda:0")
ImageFile.LOAD_TRUNCATED_IMAGES = True
class RetinopathyDatasetTrain(Dataset):

    def __init__(self, csv_file):

        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join('../input/aptos2019-blindness-detection/train_images', self.data.loc[idx, 'id_code'] + '.png')
        image = Image.open(img_name)
        image = image.resize((256, 256), resample=Image.BILINEAR)
        label = torch.tensor(self.data.loc[idx, 'diagnosis'])
        return {'image': transforms.ToTensor()(image),
                'labels': label
                }
model = torchvision.models.resnet101(pretrained=False)
model.load_state_dict(torch.load("../input/pytorch-pretrained-models/resnet101-5d3b4d8f.pth"))
num_features = model.fc.in_features
model.fc = nn.Linear(2048, 1)

model = model.to(device)
train_dataset = RetinopathyDatasetTrain(csv_file='../input/aptos2019-blindness-detection/train.csv')
data_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

plist = [
         {'params': model.layer4.parameters(), 'lr': 1e-4, 'weight': 0.001},
         {'params': model.fc.parameters(), 'lr': 1e-3}
         ]

optimizer = optim.Adam(plist, lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=10)
since = time.time()
criterion = nn.MSELoss()
num_epochs = 15
for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch, num_epochs - 1))
    print('-' * 10)
    scheduler.step()
    model.train()
    running_loss = 0.0
    tk0 = tqdm(data_loader, total=int(len(data_loader)))
    counter = 0
    for bi, d in enumerate(tk0):
        inputs = d["image"]
        labels = d["labels"].view(-1, 1)
        inputs = inputs.to(device, dtype=torch.float)
        labels = labels.to(device, dtype=torch.float)
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        counter += 1
        tk0.set_postfix(loss=(running_loss / (counter * data_loader.batch_size)))
    epoch_loss = running_loss / len(data_loader)
    print('Training Loss: {:.4f}'.format(epoch_loss))

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
torch.save(model.state_dict(), "model.bin")
