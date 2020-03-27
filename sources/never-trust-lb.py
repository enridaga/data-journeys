
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.metrics import accuracy_score, cohen_kappa_score
import random
from tqdm.notebook import tqdm

random.seed(42)
np.random.seed(42)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data_dir = "/kaggle/input/data-science-bowl-2019/"
df_labels = pd.read_csv(os.path.join(data_dir, "train_labels.csv"))
# Any results you write to the current directory are saved as output.
probas = df_labels.accuracy_group.value_counts(normalize=True)
probas.head()
df_gt_simulated = np.random.choice(probas.index.values, 7000, p=probas.values)
pd.Series(df_gt_simulated).value_counts(normalize=True)
df_pred_simulated = df_gt_simulated.copy()
cohen_kappa_score(df_gt_simulated, df_pred_simulated, weights='quadratic')
inds_to_shuffle = np.random.choice(range(len(df_gt_simulated)), int(len(df_gt_simulated) * 0.6))
df_pred_simulated[inds_to_shuffle] = np.random.permutation(df_pred_simulated[inds_to_shuffle])
cohen_kappa_score(df_gt_simulated, df_pred_simulated, weights='quadratic'), accuracy_score(df_gt_simulated, df_pred_simulated)
scores = []
for t, p in zip(np.split(df_gt_simulated, 7), np.split(df_pred_simulated, 7)):
    score = cohen_kappa_score(t, p, weights='quadratic')
    print("fold score: ", score)
    scores.append(score)
print("Mean score: ", np.mean(scores))
min(scores), max(scores)
real_kappas = []
fold_kappas = []
for t in tqdm(range(100, 1, -1)):
    df_pred_simulated = df_gt_simulated.copy()
    inds_to_shuffle = np.random.choice(range(len(df_gt_simulated)), int(len(df_gt_simulated) * t / 100))
    df_pred_simulated[inds_to_shuffle] = np.random.permutation(df_pred_simulated[inds_to_shuffle])
    real_kappa = cohen_kappa_score(df_gt_simulated, df_pred_simulated, weights='quadratic')
    scores = []
    for t, p in zip(np.split(df_gt_simulated, 7), np.split(df_pred_simulated, 7)):
        scores.append(cohen_kappa_score(t, p, weights='quadratic'))
    for s in scores:
        real_kappas.append(real_kappa)
        fold_kappas.append(s)
pd.DataFrame({"real_kappas": real_kappas, "fold_kappas": fold_kappas}).plot.scatter(x="real_kappas", y="fold_kappas", grid=True)
pd.DataFrame({"real_kappas": real_kappas, "fold_kappas": fold_kappas}).groupby("real_kappas").agg(['min', 'max']).plot()