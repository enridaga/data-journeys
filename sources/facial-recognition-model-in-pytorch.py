
%%capture
# Install facenet-pytorch
!pip install /kaggle/input/facenet-pytorch-vggface2/facenet_pytorch-1.0.1-py3-none-any.whl

# Copy model checkpoints to torch cache so they are loaded automatically by the package
!mkdir -p /tmp/.cache/torch/checkpoints/
!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-logits.pth /tmp/.cache/torch/checkpoints/vggface2_DG3kwML46X.pt
!cp /kaggle/input/facenet-pytorch-vggface2/20180402-114759-vggface2-features.pth /tmp/.cache/torch/checkpoints/vggface2_G5aNV2VSMn.pt
import os
import glob
import torch
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# See github.com/timesler/facenet-pytorch:
from facenet_pytorch import MTCNN, InceptionResnetV1

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Running on device: {device}')
# Load face detector
mtcnn = MTCNN(device=device).eval()

# Load facial recognition model
resnet = InceptionResnetV1(pretrained='vggface2', num_classes=2, device=device).eval()
# Get all test videos
filenames = glob.glob('/kaggle/input/deepfake-detection-challenge/test_videos/*.mp4')

n_frames = 10
X = []
with torch.no_grad():
    for i, filename in enumerate(filenames):
        print(f'Processing {i+1:5n} of {len(filenames):5n} videos\r', end='')
        try:
            v_cap = cv2.VideoCapture(filename)
            v_len = int(v_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            sample = np.linspace(0, v_len - 1, n_frames).round().astype(int)
            imgs = []
            for j in range(v_len):
                success, vframe = v_cap.read()
                vframe = cv2.cvtColor(vframe, cv2.COLOR_BGR2RGB)
                if j in sample:
                    imgs.append(Image.fromarray(vframe))
            v_cap.release()
            faces = mtcnn(imgs)
            faces = [f for f in faces if f is not None]
            faces = torch.stack(faces).to(device)
            embeddings = resnet(faces)
            centroid = embeddings.mean(dim=0)
            X.append((embeddings - centroid).norm(dim=1).cpu().numpy())
        except:
            X.append(None)
bias = -0.31777231
weights = np.array([
    0.13553764,
    0.3220863,
    0.03451057,
    0.20375968,
    0.13409478,
    -0.16768392,
    0.30386854,
    0.02896564,
    0.3549986,
    -0.66778037
])

submission = []
for filename, x_i in zip(filenames, X):
    if x_i is not None and len(x_i) == 10:
        prob = 1 / (1 + np.exp(-(bias + np.matmul(weights, x_i))))
    else:
        prob = 0.5
    submission.append([os.path.basename(filename), prob])
submission = pd.DataFrame(submission, columns=['filename', 'label'])
submission.sort_values('filename').to_csv('submission.csv', index=False)
submission