
import os
import pandas as pd
from kaggle.competitions import nflrush
import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
import random
from sklearn.model_selection import KFold
import lightgbm as lgb
import gc
import pickle
import tqdm
env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
unused_columns = ["GameId","PlayId","Team","Yards","TimeHandoff","TimeSnap"]
unique_columns = []
for c in train_df.columns:
    if c not in unused_columns+["PlayerBirthDate"] and len(set(train_df[c][:11]))!= 1:
        unique_columns.append(c)
        print(c," is unique")
# unique_columns+=["BirthY"]
ok = True
for i in range(0,509762,22):
    p=train_df["PlayId"][i]
    for j in range(1,22):
        if(p!=train_df["PlayId"][i+j]):
            ok=False
            break
print("train data is sorted by PlayId." if ok else "train data is not sorted by PlayId.")
ok = True
for i in range(0,509762,11):
    p=train_df["Team"][i]
    for j in range(1,11):
        if(p!=train_df["Team"][i+j]):
            ok=False
            break
print("train data is sorted by Team." if ok else "train data is not sorted by Team.")
all_columns = []
for c in train_df.columns:
    if c not in unique_columns + unused_columns+["DefensePersonnel","GameClock","PlayerBirthDate"]:
        all_columns.append(c)
all_columns.append("DL")
all_columns.append("LB")    
all_columns.append("DB")
all_columns.append("GameHour")   
for c in unique_columns:
    for i in range(22):
        all_columns.append(c+str(i))
lbl_dict = {}
for c in train_df.columns:
    if c == "DefensePersonnel":
        arr = [[int(s[0]) for s in t.split(", ")] for t in train_df["DefensePersonnel"]]
        train_df["DL"] = np.array([a[0] for a in arr])
        train_df["LB"] = np.array([a[1] for a in arr])
        train_df["DB"] = np.array([a[2] for a in arr])
    elif c == "GameClock":
        arr = [[int(s) for s in t.split(":")] for t in train_df["GameClock"]]
        train_df["GameHour"] = pd.Series([a[0] for a in arr])
    elif c == "PlayerBirthDate":
        arr = [[int(s) for s in t.split("/")] for t in train_df["PlayerBirthDate"]]
        train_df["BirthY"] = pd.Series([a[2] for a in arr])
    # elif c == "PlayerHeight":
    #     arr = [float(s.split("-")[0]) * 30.48 + float(s.split("-")[1]) * 2.54
    #         for s in list(train_df["PlayerHeight"])]
    #     train_df["PlayerHeight"] = pd.Series(arr)
    elif train_df[c].dtype=='object' and c not in unused_columns: 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_df[c].values))
        lbl_dict[c] = lbl
        train_df[c] = lbl.transform(list(train_df[c].values))
train_data=np.zeros((509762//22,len(all_columns)))
for i in tqdm.tqdm(range(0,509762,22)):
    count=0
    for c in all_columns:
        if c in train_df:
            train_data[i//22][count] = train_df[c][i]
            count+=1
    for c in unique_columns:
        for j in range(22):
            train_data[i//22][count] = train_df[c][i+j]
            count+=1        
y_train_ = np.array([train_df["Yards"][i] for i in range(0,509762,22)])
X_train = pd.DataFrame(data=train_data,columns=all_columns)
data = [0 for i in range(199)]
for y in y_train_:
    data[int(y+99)]+=1
plt.plot([i-99 for i in range(199)],data)
# scaler = preprocessing.StandardScaler()
# scaler.fit([[y] for y in y_train_])
# y_train = np.array([y[0] for y in scaler.transform([[y] for y in y_train_])])
scaler = preprocessing.StandardScaler()
scaler.fit(y_train_.reshape(-1, 1))
y_train = scaler.transform(y_train_.reshape(-1, 1)).flatten()
folds = 10
seed = 222
kf = KFold(n_splits = folds, shuffle = True, random_state=seed)
y_valid_pred = np.zeros(X_train.shape[0])
models = []

for tr_idx, val_idx in kf.split(X_train, y_train):
    tr_x, tr_y = X_train.iloc[tr_idx,:], y_train[tr_idx]
    vl_x, vl_y = X_train.iloc[val_idx,:], y_train[val_idx]
            
    print(len(tr_x),len(vl_x))
    tr_data = lgb.Dataset(tr_x, label=tr_y)
    vl_data = lgb.Dataset(vl_x, label=vl_y)  
    clf = lgb.LGBMRegressor(n_estimators=200,learning_rate=0.01)
    clf.fit(tr_x, tr_y,
        eval_set=[(vl_x, vl_y)],
        early_stopping_rounds=20,
        verbose=False)
    y_valid_pred[val_idx] += clf.predict(vl_x, num_iteration=clf.best_iteration_)
    models.append(clf)

gc.collect()
y_pred = np.zeros((509762//22,199))
y_ans = np.zeros((509762//22,199))

for i,p in enumerate(np.round(scaler.inverse_transform(y_valid_pred))):
    p+=99
    for j in range(199):
        if j>=p+10:
            y_pred[i][j]=1.0
        elif j>=p-10:
            y_pred[i][j]=(j+10-p)*0.05

for i,p in enumerate(scaler.inverse_transform(y_train)):
    p+=99
    for j in range(199):
        if j>=p:
            y_ans[i][j]=1.0

print("validation score:",np.sum(np.power(y_pred-y_ans,2))/(199*(509762//22)))
index = 0
for (test_df, sample_prediction_df) in tqdm.tqdm(env.iter_test()):
    for c in test_df.columns:
        if c == "DefensePersonnel":
            try:
                arr = [[int(s[0]) for s in t.split(", ")] for t in test_df["DefensePersonnel"]]
                test_df["DL"] = [a[0] for a in arr]
                test_df["LB"] = [a[1] for a in arr]
                test_df["DB"] = [a[2] for a in arr]
            except:
                test_df["DL"] = [np.nan for i in range(22)]
                test_df["LB"] = [np.nan for i in range(22)]
                test_df["DB"] = [np.nan for i in range(22)]
        elif c == "GameClock":
            try:
                arr = [[int(s) for s in t.split(":")] for t in test_df["GameClock"]]
                test_df["GameHour"] = pd.Series([a[0] for a in arr])
            except:
                test_df["GameHour"] = [np.nan for i in range(22)]
        elif c == "PlayerBirthDate":
            try:
                arr = [[int(s) for s in t.split("/")] for t in test_df["PlayerBirthDate"]]
                test_df["BirthY"] = pd.Series([a[2] for a in arr])
            except:
                test_df["BirthY"] = [np.nan for i in range(22)]
        # elif c == "PlayerHeight":
        #     try:
        #         arr = [float(s.split("-")[0]) * 30.48 + float(s.split("-")[1]) * 2.54
        #             for s in list(test_df["PlayerHeight"])]
        #         test_df["PlayerHeight"] = pd.Series(arr)
        #     except:
        #         test_df["PlayerHeight"] = [np.nan for i in range(22)]
        elif c in lbl_dict and test_df[c].dtype=='object'and c not in unused_columns\
            and not pd.isnull(test_df[c]).any():
            try:
                test_df[c] = lbl_dict[c].transform(list(test_df[c].values))
            except:
                test_df[c] = [np.nan for i in range(22)]
    count=0
    test_data = np.zeros((1,len(all_columns)))

    for c in all_columns:
        if c in test_df:
            try:
                test_data[0][count] = test_df[c][index]
            except:
                test_data[0][count] = np.nan
            count+=1
    for c in unique_columns:
        for j in range(22):
            try:
                test_data[0][count] = test_df[c][index + j]
            except:
                test_data[0][count] = np.nan
            count+=1        
    y_pred = np.zeros(199)        
    y_pred_p = np.sum(np.round(scaler.inverse_transform(
        [model.predict(test_data)[0] for model in models])))/folds
    y_pred_p += 99
    for j in range(199):
        if j>=y_pred_p+10:
            y_pred[j]=1.0
        elif j>=y_pred_p-10:
            y_pred[j]=(j+10-y_pred_p)*0.05
    env.predict(pd.DataFrame(data=[y_pred],columns=sample_prediction_df.columns))
    index += 22
env.write_submission_file()