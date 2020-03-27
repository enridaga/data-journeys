
from fastai.core import *
Path.read_csv = lambda o: pd.read_csv(o)
input_path = Path("/kaggle/input/data-science-bowl-2019")
pd.options.display.max_columns=200
pd.options.display.max_rows=200
input_path.ls()
sample_subdf = (input_path/'sample_submission.csv').read_csv()
specs_df = (input_path/"specs.csv").read_csv()
train_df = (input_path/"train.csv").read_csv()
train_labels_df = (input_path/"train_labels.csv").read_csv()
test_df = (input_path/"test.csv").read_csv()
train_labels_df.shape, train_df.shape, test_df.shape, specs_df.shape, sample_subdf.shape
# example submission
sample_subdf.head(2)
# training labels - how target: accuracy_group is created
train_labels_df.head(2)
train_labels_df[['num_correct', 'num_incorrect', 'accuracy', 'accuracy_group']].corr()
train_labels_df.pivot_table(values= "installation_id",index="accuracy_group", columns="accuracy", aggfunc=np.count_nonzero)
train_df.head(2)
test_df.head(2)
specs_df.head(2)
# Get last assessment start for each installation_id - it should have 'event_code' == 2000, we have exactly 1000 test samples that we need predictions of
test_assessments_df = test_df.sort_values("timestamp").query("type == 'Assessment' & event_code == 2000").groupby("installation_id").tail(1).reset_index(drop=True)
# event_data, installation_id, event_count, event_code, game_time is constant for any assessment start
# for extarcting similar rows we can look at event_code==2000 and type==Assessment combination for each installation_id
test_assessments_df = test_assessments_df.drop(['event_data', 'installation_id', 'event_count', 'event_code', 'game_time'],1); test_assessments_df
# there is unique event_id for each assesment
test_assessments_df.pivot_table(values=None, index="event_id", columns="title", aggfunc=np.count_nonzero)['game_session']
# there are common worlds among different assessments
test_assessments_df.pivot_table(values=None, index="world", columns="title", aggfunc=np.count_nonzero)['game_session']
test_assessments_df.describe(include='all')
def get_assessment_start_idxs(df): return listify(df.query("type == 'Assessment' & event_code == 2000").index)
# drop installation ids without at least 1 completed assessment
_train_df = train_df[train_df.installation_id.isin((train_labels_df.installation_id).unique())].reset_index(drop=True)
# join training labels to game starts by game sessions  
_trn_str_idxs = get_assessment_start_idxs(_train_df)
_label_df = _train_df.iloc[_trn_str_idxs]
_label_df = _label_df.merge(train_labels_df[['game_session', 'num_correct','num_incorrect','accuracy','accuracy_group']], "left", on="game_session")
_label_df = _label_df[["event_id", "installation_id", 'game_session', 'num_correct','num_incorrect','accuracy','accuracy_group']]
_label_df.head()
_label_df['accuracy_group'].value_counts(dropna=False).sort_index()
# join labels to train by event_id, game_session, installation_id
train_with_labels_df = _train_df.merge(_label_df, "left", on=["event_id", "game_session", "installation_id"])
train_with_labels_df['accuracy_group'].value_counts(dropna=False).sort_index()
train_with_labels_df.shape
# success statistics per game
(train_with_labels_df.query("type == 'Assessment'")
                     .groupby(["title", "world"])['accuracy']
                     .agg({np.mean, np.median, np.max, np.min})
                     .sort_values("mean"))
def count_nonnan(l): return np.sum([0 if np.isnan(o) else 1 for o in l])
# verify that all training installation ids have at least one assesment with non NaN label
assert not any(train_with_labels_df.groupby("installation_id")['accuracy'].agg(count_nonnan) == 0) 
# save dataframe train with labels
train_with_labels_df.to_csv("train_with_labels.csv", index=False)
# save MEM space
del _label_df
del _train_df
gc.collect()
from fastai.tabular.transform import add_datepart
# set filtered and labels added df
train_df = train_with_labels_df
def get_assessment_start_idxs_with_labels(df):
    "return indexes that will be used for supervised learning"
    df = df[~df.accuracy.isna()]
    return listify(df.query("type == 'Assessment' & event_code == 2000").index)
def get_sorted_user_df(df, ins_id):
    "extract sorted data for a given installation id and add datetime features"
    _df = df[df.installation_id == ins_id].sort_values("timestamp").reset_index(drop=True)
    add_datepart(_df, "timestamp", time=True)
    return _df
# pick installation_id and get data until an assessment_start
rand_id = np.random.choice(train_df.installation_id)
user_df = get_sorted_user_df(train_df, rand_id)
start_idxs = get_assessment_start_idxs_with_labels(user_df)
print(f"Assessment start idxs in user df: {start_idxs}")
# we would like to get and create features for each assessment start for supervised learning
str_idx = start_idxs[1]
user_assessment_df = user_df[:str_idx+1]; user_assessment_df
from fastai.tabular import *
import types

stats = ["median","mean","sum","min","max"]

UNIQUE_COL_VALS = types.SimpleNamespace(
    event_ids = np.unique(train_df.event_id),
    media_types = np.unique(train_df.type),
    titles = np.unique(train_df.title),
    worlds = np.unique(train_df.world),
    event_codes = np.unique(train_df.event_code),
)
def array_output(f):
    def inner(*args, **kwargs): return array(listify(f(*args, **kwargs))).flatten()
    return inner
feature_funcs = []
@array_output
def time_elapsed_since_hist_begin(df):
    "total time passed until assessment begin"
    return df['timestampElapsed'].max() - df['timestampElapsed'].min()

feature_funcs.append(time_elapsed_since_hist_begin)
time_elapsed_since_hist_begin(user_assessment_df)
@array_output
def time_elapsed_since_each(df, types, dfcol):
    "time since last occurence of each types, if type not seen then time since history begin"
    types = UNIQUE_COL_VALS.__dict__[types]
    last_elapsed = df['timestampElapsed'].max()
    _d = dict(df.iloc[:-1].groupby(dfcol)['timestampElapsed'].max())
    return [last_elapsed - _d[t] if t in _d else time_elapsed_since_hist_begin(df)[0] for t in types]
feature_funcs.append(partial(time_elapsed_since_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(time_elapsed_since_each, types="titles", dfcol="title"))
feature_funcs.append(partial(time_elapsed_since_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(time_elapsed_since_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(time_elapsed_since_each, types="event_codes", dfcol="event_code"))
def countfreqhist(df, types, dfcol, freq=False):
    "count or freq of types until assessment begin"
    types = UNIQUE_COL_VALS.__dict__[types]
    _d = dict(df[dfcol].value_counts(normalize=(True if freq else False)))
    return [_d[t] if t in _d else 0 for t in types]
feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=False))
feature_funcs.append(partial(countfreqhist, types="media_types", dfcol="type", freq=True))

feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=False))
feature_funcs.append(partial(countfreqhist, types="titles", dfcol="title", freq=True))

feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=False))
feature_funcs.append(partial(countfreqhist, types="event_ids", dfcol="event_id", freq=True))

feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=False))
feature_funcs.append(partial(countfreqhist, types="worlds", dfcol="world", freq=True))

feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=False))
feature_funcs.append(partial(countfreqhist, types="event_codes", dfcol="event_code", freq=True))
@array_output
def overall_event_count_stats(df):
    "overall event count stats until assessment begin"
    return df['event_count'].agg(stats)

feature_funcs.append(overall_event_count_stats)
overall_event_count_stats(user_assessment_df)
@array_output
def event_count_stats_each(df, types, dfcol):
    "event count stats per media types until assessment begin, all zeros if media type missing for user"
    types = UNIQUE_COL_VALS.__dict__[types]
    _stats_df = df.groupby(dfcol)['event_count'].agg(stats)
    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))
    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]
feature_funcs.append(partial(event_count_stats_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(event_count_stats_each, types="titles", dfcol="title"))
feature_funcs.append(partial(event_count_stats_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(event_count_stats_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(event_count_stats_each, types="event_codes", dfcol="event_code"))
@array_output
def overall_session_game_time_stats(df):
    "overall session game time stats until assessment begin"
    return df['game_time'].agg(stats)

feature_funcs.append(overall_session_game_time_stats)
overall_session_game_time_stats(user_assessment_df)
@array_output
def session_game_time_stats_each(df, types, dfcol):
    "session game time stats per media types until assessment begin, all zeros if missing for user"
    types = UNIQUE_COL_VALS.__dict__[types]
    _stats_df = df.groupby(dfcol)['game_time'].agg(stats)
    _d = dict(zip(_stats_df.reset_index()[dfcol].values, _stats_df.values))
    return [_d[t] if t in _d else np.zeros(len(stats)) for t in types]
feature_funcs.append(partial(session_game_time_stats_each, types="media_types", dfcol="type"))
feature_funcs.append(partial(session_game_time_stats_each, types="titles", dfcol="title"))
feature_funcs.append(partial(session_game_time_stats_each, types="event_ids", dfcol="event_id"))
feature_funcs.append(partial(session_game_time_stats_each, types="worlds", dfcol="world"))
feature_funcs.append(partial(session_game_time_stats_each, types="event_codes", dfcol="event_code"))
sample_features = np.concatenate([f(user_assessment_df) for f in feature_funcs]); sample_features.shape
def get_test_assessment_start_idxs(df): 
    return list(df.sort_values("timestamp")
                  .query("type == 'Assessment' & event_code == 2000")
                  .groupby("installation_id").tail(1).index)
# trn_str_idxs = get_assessment_start_idxs_with_labels(train_with_labels_df)
# test_str_idxs = get_test_assessment_start_idxs(test_df)
# Get training features
def get_train_feats_row(ins_id, i):
    "get all assessment start features for an installation id"
    rows = [] # collect rows with features for each assessment start
    user_df = get_sorted_user_df(train_with_labels_df, ins_id)
    start_idxs = get_assessment_start_idxs_with_labels(user_df); start_idxs
    for idx in start_idxs:
        assessment_row = user_df.iloc[idx]
        _df = user_df[:idx+1]
        row_feats = np.concatenate([f(_df) for f in feature_funcs])
        feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])
        row = pd.concat([assessment_row, feat_row])
        rows.append(row)
    return rows
# # compute static features for train assessment start
# installation_ids = train_with_labels_df.installation_id.unique()
# res = parallel(get_train_feats_row, (installation_ids))
# train_with_features_df = pd.concat([row for rows in res for row in rows],1).T
# train_with_features_df.head()
# train_with_features_df.to_csv("train_with_features_part1.csv")
def get_test_feats_row(idx, i):
    "get all faeatures by an installation start idx"
    ins_id = test_df.loc[idx, "installation_id"]
    _df = get_sorted_user_df(test_df, ins_id)
    assessment_row = _df.iloc[-1]
    row_feats = np.concatenate([f(_df) for f in feature_funcs])
    feat_row = pd.Series(row_feats, index=[f"static_feat{i}"for i in range(len(row_feats))])
    row = pd.concat([assessment_row, feat_row])
    return row
# # compute static features for test assessment start and save 
# start_idxs = get_test_assessment_start_idxs(test_df)
# res = parallel(get_test_feats_row, start_idxs)
# test_with_features_df = pd.concat(res,1).T
# test_with_features_df.head()
# test_with_features_df.to_csv("test_with_features_part1.csv")