
import pandas as pd
import numpy as np

import json

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier
# from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

import keras

# import pprint
import gc
import matplotlib.pyplot as plt
import seaborn as sns

# pandas display option
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_row', 1500)
pd.set_option('max_colwidth', 150)
pd.set_option('display.float_format', '{:.2f}'.format)
train = pd.read_csv("/kaggle/input/data-science-bowl-2019/train.csv")
# test = pd.read_csv("/kaggle/input/data-science-bowl-2019/test.csv")
label = pd.read_csv("/kaggle/input/data-science-bowl-2019/train_labels.csv")
# sample = pd.read_csv("/kaggle/input/data-science-bowl-2019/sample_submission.csv")
specs = pd.read_csv("/kaggle/input/data-science-bowl-2019/specs.csv")
# df = train.merge(label, how="left", on=["installation_id", "game_session", "title"])
# df.timestamp = pd.to_datetime(df.timestamp)
# df.sort_values(["timestamp", "event_count"], ascending=True, inplace=True)
# del train, label
# gc.collect()
train.timestamp = pd.to_datetime(train.timestamp)
df = train.sort_values(["timestamp", "event_count"], ascending=True)
del train
gc.collect()
# ['Scrub-A-Dub', 'Bubble Bath', 'All Star Sorting', 'Chow Time',
#  'Dino Dive', 'Happy Camel', 'Leaf Leader', 'Pan Balance',
#  'Dino Drink', 'Crystals Rule', 'Air Show']
df.head()
title_select = "Scrub-A-Dub"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])
_["levels"] = _.event_data.apply(lambda x:json.loads(x)["level"])

bins_rounds = len(np.unique(_.groupby("game_session").rounds.max()))
bins_levels = len(np.unique(_.groupby("game_session").levels.max()))

plt.figure(figsize=(25, 7))
plt.subplot(1, 2, 1)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)
plt.subplot(1, 2, 2)
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins_rounds, label="rounds")
sns.distplot(_.groupby("game_session").levels.max(), bins=bins_levels, label="levels")
plt.legend()
plt.title(title_select)

plt.show()
title_select = "Bubble Bath"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "All Star Sorting"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Chow Time"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Dino Dive"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Happy Camel"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Leaf Leader"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Pan Balance"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Dino Drink"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Crystals Rule"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
title_select = "Air Show"
_ = df.query("type=='Game' and title==@title_select").reset_index(drop=True)
_ = _.loc[:, ["installation_id", "game_session", "event_data"]]
_["rounds"] = _.event_data.apply(lambda x:json.loads(x)["round"])

bins = len(np.unique(_.groupby("game_session").rounds.max()))
plt.figure(figsize=(20, 10))
sns.distplot(_.groupby("game_session").rounds.max(), bins=bins)
plt.title(title_select)
plt.show()
# accuracy = _.query("((event_code==2000) or (event_code==4100 and title!='Bird Measurer (Assessment)') or \
#                      (event_code==4110 and title=='Bird Measurer (Assessment)')) and (type=='Assessment')").reset_index(drop=True)

# accuracy["event_data_json"] = accuracy["event_data"].apply(lambda x: json.loads(x))

# accuracy["num_incorrect"] = accuracy["event_data_json"].apply(lambda x: (0 if x["correct"] else 1) if "correct" in x  else 0)
# accuracy["num_correct"] = accuracy["event_data_json"].apply(lambda x: (1 if x["correct"] else 0)  if "correct" in x  else 0)

# accuracy = accuracy.groupby(["installation_id", "game_session"]).agg(num_correct_pred = ("num_correct", "max"), num_incorrect_pred = ("num_incorrect", "sum"), ).reset_index()
# accuracy["accuracy_group_pred"] = accuracy["num_incorrect_pred"].apply(lambda x: 3 if x == 0 else (2 if x == 1 else 1)) * accuracy["num_correct_pred"]

# accuracy = accuracy.groupby(["installation_id"]).last().reset_index()
# accuracy.drop("game_session", axis=1, inplace=True)