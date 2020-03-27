
import gc, os, sys, time

import pandas as pd, numpy as np

from itertools import combinations

from IPython.display import HTML, display



pd.options.display.max_rows = 200
IN_DIR = '../input'

len(os.listdir(IN_DIR))
users = pd.read_csv(f'{IN_DIR}/Users.csv', index_col='Id')

users.shape
EXCLUDE_USERS = [2080166] # Kaggle Kerneler - very high stats that distort the league tables!



users.loc[EXCLUDE_USERS].T
users = users.drop(EXCLUDE_USERS)

users.shape
users.head()
def columns(fn):

    df = pd.read_csv(fn, nrows=5)

    return df.columns



def user_columns(fn):

    return [c for c in columns(fn) if 'UserId' in c]
for f in os.listdir(IN_DIR):

    if '.csv' in f:

        csv = IN_DIR+'/'+f

        cols = user_columns(csv)

        if len(cols) < 1:

            continue

        table = f.replace('.csv', '')

        df = pd.read_csv(csv, usecols=cols)

        for col in cols:

            tag = f'Count_{table}_{col}'

            print(tag)

            vc = df[col].value_counts()

            ser = users.index.map(vc)

            users[tag] = ser.fillna(0).astype('int32')
N_SHOW = 50
def user_name_link(r):

    return f'<a href="https://www.kaggle.com/{r.UserName}">{r.DisplayName}</a>'



TIERS = np.asarray([

    '<font color=green>novice</font>',

    '<font color=blue>novice</font>',

    '<font color=purple>expert</font>',

    '<font color=orange>master</font>',

    '<font color=gold>grandmaster</font>',

    '<font color=black>staff</font>',

])



def league_table(col, src_df=users):

    display(HTML("<H1>" + col.replace('_', ' ') + "</H1>"))

    df = src_df.sort_values(col, ascending=False).head(N_SHOW)

    uid = df.apply(user_name_link, axis=1)

    df.pop('UserName')

    df.pop('DisplayName')

    df.insert(0, 'Tier', TIERS[df.PerformanceTier])

    df.insert(0, 'DisplayName', uid)

    df['Rank'] = df[col].rank(method='min', ascending=False).astype(int)

    return df[['Rank','DisplayName','Tier',col]].style.bar(subset=[col], vmin=0)



# for c in activity_sums.index: print(f'league_table("{c}")')
league_table("Count_TeamMemberships_UserId")
league_table("Count_Submissions_SubmittedUserId")
league_table("Count_KernelVotes_UserId")
league_table("Count_DatasetVotes_UserId")
league_table("Count_KernelVersions_AuthorUserId")
league_table("Count_Kernels_AuthorUserId")
league_table("Count_ForumMessages_PostUserId")
league_table("Count_UserFollowers_UserId")
league_table("Count_ForumMessageVotes_FromUserId")
league_table("Count_UserFollowers_FollowingUserId")
league_table("Count_ForumMessageVotes_ToUserId")
league_table("Count_DatasetVersions_CreatorUserId")
league_table("Count_Datasets_CreatorUserId")
league_table("Count_Datasources_CreatorUserId")
league_table("Count_Datasets_OwnerUserId")
league_table("Count_UserOrganizations_UserId")
all_col_counts = users.columns[users.columns.str.startswith('Count_')]

len(all_col_counts)
users.Count_UserAchievements_UserId.value_counts()
count_cols = [c for c in all_col_counts if c != 'Count_UserAchievements_UserId']

len(count_cols)
users[count_cols].sum(1).value_counts().head()
users.query('UserName=="jtrotman"').T
users['Sum_Activity_Flags'] = (users[count_cols]>0).sum(1)
users.Sum_Activity_Flags.value_counts()
(users.Sum_Activity_Flags==0).mean()
users.Sum_Activity_Flags.max()
idx = users.Sum_Activity_Flags==users.Sum_Activity_Flags.max()

idx.sum()
show = ['UserName', 'DisplayName', 'RegisterDate', 'PerformanceTier']
users[idx][show]
league_table('Total_Activities', users.assign(Total_Activities=users[count_cols].sum(1)))
def users_with_n_activities(n, min_count=0):

    bi_sum = users.Sum_Activity_Flags==n

    for cols in combinations(count_cols, n):

        idx = bi_sum

        for c in cols:

            idx = (idx & (users[c]>0))

            n = idx.sum()

            if n<min_count:

                break

        if n>=min_count:

            yield (n,) + cols



def users_with_n_activities_df(n, min_count=0):

    df = pd.DataFrame.from_records(

        users_with_n_activities(n, min_count),

        columns=['Count'] + list(range(n))

    )

    return df
users_with_n_activities_df(1).sort_values('Count', ascending=False).reset_index(drop=True)
users_with_n_activities_df(2).sort_values('Count', ascending=False).reset_index(drop=True)
users_with_n_activities_df(3, min_count=2000).sort_values('Count', ascending=False).reset_index(drop=True)
users.shape
entered = users.Count_TeamMemberships_UserId>0

submitted = users.Count_Submissions_SubmittedUserId>0



idx = (

 (users.Sum_Activity_Flags==0)

 | 

 ((users.Sum_Activity_Flags==1) & (entered))

 | 

 ((users.Sum_Activity_Flags==2) & (entered) & (submitted))

)
idx.sum()
idx.mean()
users.shape[0] - idx.sum()
activity_sums = (users[count_cols]>0).sum(0)

activity_sums = activity_sums.sort_values(ascending=False)

activity_sums
performanceTiers = np.asarray(['novice', 'contributor', 'expert', 'master', 'gm', 'staff'])
users.PerformanceTier.value_counts()
vc = users.PerformanceTier.value_counts()
vc.index = performanceTiers[vc.index]
vc