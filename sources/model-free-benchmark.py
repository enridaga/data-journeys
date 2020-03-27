
import pandas as pd
import numpy as np
train = pd.read_csv("../input/nfl-big-data-bowl-2020/train.csv")
train = train.drop_duplicates(subset="PlayId")
dist = train["Yards"].hist(density = True, cumulative = True, bins = 200)
train_own = train[train["FieldPosition"] == train["PossessionTeam"]]
train_other = train[train["FieldPosition"] != train["PossessionTeam"]]
import matplotlib.pyplot as plt
own_cdf = np.histogram(train_own["Yards"], bins=199,
                 range=(-99,100), density=True)[0].cumsum()
other_cdf = np.histogram(train_other["Yards"], bins=199,
                 range=(-99,100), density=True)[0].cumsum()
own_cdf
other_cdf
y_train = train["Yards"].values
y_ans = np.zeros((len(train),199))

for i,p in enumerate(y_train):
    for j in range(199):
        if j-99>=p:
            y_ans[i][j]=1.0
print("validation score own half:",np.sum(np.power(own_cdf-y_ans,2))/(199*(len(train))))
print("validation score other half:",np.sum(np.power(other_cdf-y_ans,2))/(199*(len(train))))
import matplotlib.pyplot as plt
plt.plot(own_cdf)
plt.plot(other_cdf)
from kaggle.competitions import nflrush
env = nflrush.make_env()
for (test_df, sample_prediction_df) in env.iter_test():
    if test_df["FieldPosition"].iloc[0] != test_df["PossessionTeam"].iloc[0]:
        #when they are in the opponents half
        cdf = np.copy(other_cdf)
        cdf[-test_df["YardLine"].iloc[0]:] = 1
        sample_prediction_df.iloc[0, :] = cdf
    else:
        #when they are in their own half
        cdf = np.copy(own_cdf)
        cdf[-(100 - (test_df["YardLine"].iloc[0] + 50)):] = 1
        sample_prediction_df.iloc[0, :] = cdf
    env.predict(sample_prediction_df)

env.write_submission_file()
print(sample_prediction_df)
