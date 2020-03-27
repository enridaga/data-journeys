
from kaggle.competitions import nflrush
import pandas as pd

# You can only call make_env() once, so don't lose it!
env = nflrush.make_env()
train_df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
train_df
# You can only iterate through a result from `env.iter_test()` once
# so be careful not to lose it once you start iterating.
iter_test = env.iter_test()
(test_df, sample_prediction_df) = next(iter_test)
test_df
sample_prediction_df
sample_prediction_df[sample_prediction_df.columns[98:108]]
next(iter_test)
env.predict(sample_prediction_df)
for (test_df, sample_prediction_df) in iter_test:
    env.predict(sample_prediction_df)
env.write_submission_file()
# We've got a submission file!
import os
print([filename for filename in os.listdir('/kaggle/working') if '.csv' in filename])