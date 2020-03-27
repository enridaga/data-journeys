
import os
from collections import Counter, defaultdict
from pathlib import Path

from tqdm.notebook import tqdm
import json
import numpy as np
import pandas as pd
from fastai.tabular import * 

pd.set_option('display.max_colwidth', 200)
pd.set_option('display.max_columns', None)
pd.set_option('display.min_rows', 100)
pd.set_option('display.max_rows', 100)
home = Path("/kaggle/input/data-science-bowl-2019/")
specs = pd.read_csv(home/"specs.csv"); len(specs)
specs.head()
train_labels = pd.read_csv(home/"train_labels.csv"); len(train_labels)
train_labels.head(5)
pd.read_csv(home/"sample_submission.csv").head()
%%time
types = {"event_code": np.int16, "event_count": np.int16, "game_time": np.int32}
raw_train = pd.read_csv(home/"train.csv", dtype=types)
raw_train["timestamp"] = pd.to_datetime(raw_train["timestamp"]); len(raw_train)
raw_test = pd.read_csv(home/"test.csv", dtype=types)
raw_test["timestamp"] = pd.to_datetime(raw_test["timestamp"])
raw_test.head(5)
raw_train.sample(5)
# raw_train[raw_train["game_session"] == "969a6c0d56aa4683"].tail()
# TODO keep them and see how it affects score
ids_with_subms = raw_train[raw_train.type == "Assessment"][['installation_id']].drop_duplicates()
raw_train = pd.merge(raw_train, ids_with_subms, on="installation_id", how="inner"); len(raw_train)
def get_accuracy(correct_data):
    # Rounding correct > 1 to 1 lowers the score. Why?
    correct = len(correct_data.loc[correct_data])
    wrong = len(correct_data.loc[~correct_data])
    accuracy = correct/(correct + wrong) if correct + wrong else 0
    return accuracy, correct, wrong

def get_group(accuracy):
    if not accuracy:
        return 0
    elif accuracy == 1:
        return 3
    elif accuracy >= 0.5:
        return 2
    return 1
# I prefer this over calculating average
def lin_comb(v1, v2, beta): return beta*v1 + (1-beta)*v2
# TRY
# do not remove assessments without attempts in train
# add number of passed assessments
# add time spent in each activity
# remove sessions with 1 row
# clear installation_id not found in train_labels
# event_data

def prepare(data: pd.DataFrame, one_hot: List[str], test=False) -> pd.DataFrame:
    one_hot_dict = defaultdict(int)

    prepared = []
    for id_, g in tqdm(data.groupby("installation_id", sort=False)):
        features = process_id(g, one_hot, one_hot_dict.copy(), test)
        if not features:
            continue
        if test:
            features[-1]["is_test"] = 1
        prepared.extend(features)
    return pd.DataFrame(prepared).fillna(0).sort_index(axis=1)
def process_id(id_data: pd.DataFrame, one_hot_cols, one_hot_dict, test: bool) -> pd.DataFrame:
    a_accuracy, a_group, a_correct, a_wrong, counter, accumulated_duration_mean = 0, 0, 0, 0, 0, 0
    a_groups = {"0":0, "1":0, "2":0, "3":0}
    # accumulated one_hot features per id for a given session, e.g. Bird Measurer: 50
    features = []

    for s, gs in id_data.groupby("game_session", sort=False):
        def update_counter(counter: dict, column: str):
            session_counter = Counter(gs[column])
            for value in session_counter.keys():
                counter[f"{column}_{value}"] += session_counter[value]
            return counter

        def process_session(gs):
            # share state with parent process_id()
            nonlocal one_hot_dict, a_groups, a_accuracy, a_group, a_correct, a_wrong, counter, accumulated_duration_mean
            # increment one hot columns for session, e.g. Bird Measurer: 50
            for c in one_hot_cols:
                one_hot_dict.update(update_counter(one_hot_dict, c))
    
            # an accumulated session duration mean
            duration = (gs["timestamp"].iloc[-1] - gs["timestamp"].iloc[0]).seconds
            accumulated_duration_mean = lin_comb(accumulated_duration_mean or duration, duration, beta=0.9)
            if gs["type"].iloc[0] != "Assessment":
                return

            guess_mask = ((gs["event_data"].str.contains("correct")) & 
             (((gs["event_code"] == 4100) &(~gs["title"].str.startswith("Bird")) | 
               ((gs["event_code"] == 4110) & (gs["title"].str.startswith("Bird"))))))
            answers = gs.loc[guess_mask].event_data.apply(lambda x: json.loads(x).get("correct"))

            # skip assessments without attempts in train
            if answers.empty and not test:
                return
            accuracy, correct, wrong = get_accuracy(answers)
            assert accuracy <= 1
            group = get_group(accuracy)
            processed = {"installation_id": id_data.installation_id.iloc[0], #"game_session": s,
                         "title": gs.title.iloc[0],
                         "last_timestamp": gs.timestamp.iloc[-1], "accumulated_duration_mean": accumulated_duration_mean,
                         "accumulated_correct": a_correct, "accumulated_incorrect": a_wrong,
                         "accumulated_accuracy_mean": a_accuracy/counter if counter > 0 else 0,
                         "accumulated_accuracy_group_mean": a_group/counter if counter > 0 else 0, 
                         "accuracy_group": group}
            processed.update(a_groups)
            counter += 1
            a_accuracy += accuracy
            a_correct += correct
            a_wrong += wrong
            a_group += group
            a_groups[str(group)] += 1
            processed.update(one_hot_dict)
            return processed
        
        gs.reset_index(inplace=True, drop=True)

        if (gs["timestamp"].iloc[-1] - gs["timestamp"].iloc[0]).seconds > 1800:
            gs.loc[:, "passed"] = gs.loc[:, "timestamp"].diff().apply(lambda x: x.seconds)
            id_max = gs.loc[:, "passed"].idxmax()
            if gs.loc[:, "passed"].max() > 1800:
                session = gs.iloc[:id_max]
                continued_session = gs.iloc[id_max:]
                fs = process_session(session)
                c_fs = process_session(continued_session)
                if fs:
                    features.append(fs)
                if c_fs:
                    features.append(c_fs)
                continue

        session_features = process_session(gs)
        if session_features:
            features.append(session_features)
        
    return features
# import gc; gc.collect()
one_hot_counters=["title", "type", "event_code", "event_id"]
train = prepare(raw_train, one_hot=one_hot_counters)
# train = prepare(raw_train.iloc[:1_000_000], one_hot=one_hot_counters)
add_datepart(train, "last_timestamp", prefix="last_", time=True)
train.tail()
test = prepare(raw_test, one_hot=one_hot_counters, test=True)
# for the case when one hot encoded columns don't match between datasets
add_datepart(test, "last_timestamp", prefix="last_", time=True);
# diff = train.columns.difference(test.columns)
# display(f"Test doesn't contain {diff}")
# for c in diff:
#     test[c] = 0
# why discard good data from test, let's use all the taken assessments for train!
train = (pd.concat([train, test[test["is_test"] == 0].drop(columns=["is_test"])],
                   ignore_index=True, sort=False)).fillna(0)
train.tail()
test = test.loc[test["is_test"] == 1]
test.drop(columns=["accuracy_group", "is_test"], inplace=True)
test.head()
del_cols = []
for col in train.columns.values:
    counts = train[col].value_counts().iloc[0]
    if (counts / train.shape[0]) >= 0.99:
        del_cols.append(col)
train.drop(columns=del_cols, inplace=True)
test.drop(columns=del_cols, inplace=True, errors="ignore")
display(f"Dropped {del_cols}")
procs = [FillMissing, Categorify, Normalize]
np.random.seed(42)
# grab the last assessments per id
valid_idx = [g.iloc[-1].name for i, g in train.groupby("installation_id", sort=False)]
threes = train.loc[valid_idx].query("accuracy_group == 3").index
zeroes = train.loc[valid_idx].query("accuracy_group == 0").index
others = train.loc[valid_idx].query("accuracy_group == 1 or accuracy_group == 2").index
valid_idx = sorted(pd.Series(threes).sample(len(others) // 2, random_state=42).to_list() +
             pd.Series(zeroes).sample(len(others) // 2, random_state=42).to_list() +
             others.to_list()
            )
len(valid_idx)
# let's randomly leave some data for the train
# valid_idx = np.random.choice(valid_idx, int(len(valid_idx) * 0.5), replace=False)
# valid_idx = pd.Series(valid_idx).sample(int(len(valid_idx) * 0.5), random_state=42).sort_values().values
# len(valid_idx)
# ids = train["installation_id"].unique()
# sampled_ids = np.random.choice(ids, int(len(ids) * 0.2))
# valid_idx = train[train["installation_id"].isin(sampled_ids)].drop_duplicates(["installation_id"],
#                                                                               keep="last").index
# len(valid_idx)
train.accuracy_group.value_counts(normalize=True)
train.loc[valid_idx].accuracy_group.value_counts(normalize=True)
date_cols = train.columns[train.columns.str.startswith("last", na=False)].to_list()
dep_var = "accuracy_group"
cat_names = list(filter(lambda x: x not in [dep_var, "last_Elapsed"], date_cols)) + ["title"]
cont_names = list(filter(lambda x: x not in ["installation_id", "game_session", dep_var] + cat_names,
                         train.columns.to_list()))
data = (TabularList.from_df(train, path=home, cat_names=cat_names, cont_names=cont_names, procs=procs)
        .split_by_idx(valid_idx=valid_idx)
        .label_from_df(cols=dep_var, label_cls=CategoryList)
        .add_test(TabularList.from_df(test, path=home, cat_names=cat_names, cont_names=cont_names, procs=procs))
        .databunch()
)
# data.show_batch()
def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction=='mean' else loss.sum() if reduction=='sum' else loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, ε:float=0.1, reduction='mean'):
        super().__init__()
        self.ε,self.reduction = ε,reduction
    
    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return lin_comb(loss/c, nll, self.ε)

## TODO - update kappa to regression
learn = tabular_learner(data, layers=[1000,500],
#                         metrics=[mean_absolute_error, explained_variance],
#                         y_range=[0, 3],
                        metrics=[KappaScore("quadratic")],
                        loss_func=LabelSmoothingCrossEntropy(),
#                         emb_drop=0.04,
#                         use_bn=False,
                       )
# learn.model_dir = "/kaggle/working"
# learn.lr_find()
# learn.recorder.plot()
learn.fit_one_cycle(5, 1e-03)
# for lr in [1e-03, 5e-03, 1e-02, 3e-02]:
#     learn = tabular_learner(data, layers=[200,100],
#                             metrics=[KappaScore("quadratic")],
#                             loss_func=LabelSmoothingCrossEntropy(),
#                            )
#     display(lr)
#     learn.fit_one_cycle(20, lr)
# preds, y = learn.get_preds(ds_type=DatasetType.Test)
# labels = preds.flatten()
# display(labels[:10])
# pd.Series(labels.tolist()).hist(bins=100); 
# def rounder(preds):
#     y = preds.clone()
#     y[y < 0.7] = 0
#     y[(y >= 0.7) & (y < 1.4)] = 1
#     y[y >= 1.8] = 3
#     y[(y >= 1.4) & (y < 1.8)] = 2
#     return y.type(torch.IntTensor)
# labels = rounder(labels)
# # labels = labels.round().type(torch.IntTensor)
# pd.Series(labels.tolist()).hist(bins=4);
# pd.Series(preds.type(torch.IntTensor).flatten()).value_counts()
preds, y = learn.get_preds(ds_type=DatasetType.Test)
labels = np.argmax(preds, 1)
submission = pd.DataFrame({"installation_id": test.installation_id, "accuracy_group": labels})
submission.to_csv("submission.csv", index=False)
len(submission), submission.accuracy_group.value_counts(normalize=True)
