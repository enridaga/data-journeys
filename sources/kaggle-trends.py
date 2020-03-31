
### matplotlib inline
import os
import pandas as pd
import datetime as dt
import numpy as np
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [16, 10]
plt.rcParams['font.size'] = 14
import seaborn as sns
sns.set_palette(sns.color_palette('tab20', 20))
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
from datetime import date, timedelta
class MetaData():
    def __init__(self, path='../input'):
        self.path = path

    def Competitions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Competitions.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'CompetitionId'})

    def CompetitionTags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'CompetitionTags.csv'), nrows=nrows)

    def Datasets(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Datasets.csv'), nrows=nrows)

    def DatasetTags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasetTags.csv'), nrows=nrows)

    def DatasetVersions(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasetVersions.csv'), nrows=nrows)

    def DatasetVotes(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasetVotes.csv'), nrows=nrows)

    def DatasourceObjects(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasourceObjects.csv'), nrows=nrows)

    def Datasources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Datasources.csv'), nrows=nrows)

    def DatasourceVersionObjectTables(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'DatasourceVersionObjectTables.csv'), nrows=nrows)

    def ForumMessages(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessages.csv'), nrows=nrows)
        df['PostDate'] = pd.to_datetime(df['PostDate'])
        df['PostWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.PostDate]
        return df.rename(columns={'Id': 'ForumMessageId'})

    def ForumMessageVotes(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumMessageVotes.csv'), nrows=nrows)
        df['VoteDate'] = pd.to_datetime(df['VoteDate'])
        df['VoteWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.VoteDate]
        return df

    def Forums(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Forums.csv'), nrows=nrows).rename(columns={'Id': 'ForumId'})

    def ForumTopics(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'ForumTopics.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'ForumTopicId'})

    def KernelLanguages(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'KernelLanguages.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'KernelLanguageId', 'DisplayName': 'KernelLanguageName'})

    def Kernels(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Kernels.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'KernelId'})

    def KernelTags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelTags.csv'), nrows=nrows)

    def KernelVersionCompetitionSources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionCompetitionSources.csv'), nrows=nrows)

    def KernelVersionDatasetSources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionDatasetSources.csv'), nrows=nrows)

    def KernelVersionKernelSources(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionKernelSources.csv'), nrows=nrows)

    def KernelVersionOutputFiles(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'KernelVersionOutputFiles.csv'), nrows=nrows)

    def KernelVersions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'KernelVersions.csv'), nrows=nrows)
        df['CreationDate'] = pd.to_datetime(df['CreationDate'])
        df['CreationWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.CreationDate]
        return df.rename(columns={'Id': 'KernelVersionId'})

    def KernelVotes(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'KernelVotes.csv'), nrows=nrows)
        df['VoteDate'] = pd.to_datetime(df['VoteDate'])
        df['VoteWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.VoteDate]
        return df

    def Medals(self):
        df = pd.DataFrame([
            [1, 'Gold', '#FFCE3F', '#A46A15'],
            [2, 'Silver', '#E6E6E6', '#787775'],
            [3, 'Bronze', '#EEB171', '#835036'],
        ], columns=['Medal', 'MedalName', 'MedalBody', 'MedalBorder'])
        return df

    def Organizations(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Organizations.csv'), nrows=nrows)

    def Submissions(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Submissions.csv'), nrows=nrows,
                         usecols=['SubmittedUserId', 'TeamId', 'SubmissionDate'])
        df['SubmissionDate'] = pd.to_datetime(df['SubmissionDate'])
        df['SubmissionWeek'] = [date_to_first_day_of_week(pd.Timestamp(d).date()) for d in df.SubmissionDate]
        return df

    def Tags(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'Tags.csv'), nrows=nrows)

    def TeamMemberships(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'TeamMemberships.csv'), nrows=nrows)

    def Teams(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Teams.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'TeamId'})

    def UserAchievements(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserAchievements.csv'), nrows=nrows)

    def UserFollowers(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserFollowers.csv'), nrows=nrows)

    def UserOrganizations(self, nrows=None):
        return pd.read_csv(os.path.join(self.path, 'UserOrganizations.csv'), nrows=nrows)

    def Users(self, nrows=None):
        df = pd.read_csv(os.path.join(self.path, 'Users.csv'), nrows=nrows)
        return df.rename(columns={'Id': 'UserId'})

    def PerformanceTiers(self):
        df = pd.DataFrame([
            [0, 'Novice', '#5ac995'],
            [1, 'Contributor', '#00BBFF'],
            [2, 'Expert', '#95628f'],
            [3, 'Master', '#f96517'],
            [4, 'GrandMaster', '#dca917'],
            [5, 'KaggleTeam', '#008abb'],
        ], columns=['PerformanceTier', 'PerformanceTierName', 'PerformanceTierColor'])
        return df

    def get_weekly_forum_votes(self):
        fmv = self.ForumMessageVotes()
        fm = self.ForumMessages()
        ft = self.ForumTopics()
        f = self.Forums()
        fmv['cnt'] = 1
        weekly_message_votes = fmv.groupby(['ForumMessageId', 'VoteWeek'])[['cnt']].sum().reset_index()
        weekly_message_votes = weekly_message_votes.merge(fm[['ForumMessageId', 'ForumTopicId', 'Message']],
                                                          on='ForumMessageId')
        weekly_message_votes = weekly_message_votes.merge(ft[['ForumTopicId', 'ForumId', 'Title']], on='ForumTopicId')
        weekly_message_votes = weekly_message_votes.merge(f[['ForumId', 'Title']], on='ForumId',
                                                          suffixes=['Topic', 'Forum'])
        return weekly_message_votes

    def get_weekly_kernel_version_votes(self):
        kernel_votes = self.KernelVotes()
        kernel_versions = self.KernelVersions()
        kernels = self.Kernels()
        kernel_votes['cnt'] = 1
        weekly_kernel_votes = kernel_votes.groupby(['KernelVersionId', 'VoteWeek'])[['cnt']].sum().reset_index()
        weekly_kernel_votes = weekly_kernel_votes.merge(kernel_versions[['KernelVersionId', 'KernelId']],
                                                        on='KernelVersionId')
        weekly_kernel_votes = weekly_kernel_votes.merge(kernels[['KernelId', 'CurrentKernelVersionId']],
                                                        on='KernelId')
        weekly_kernel_votes = weekly_kernel_votes.merge(
            kernel_versions[['KernelVersionId', 'KernelLanguageId', 'AuthorUserId', 'Title']],
            left_on='CurrentKernelVersionId', right_on='KernelVersionId')
        return weekly_kernel_votes

    def get_kernel_vote_info(self):
        kernel_votes = self.KernelVotes()
        kernel_versions = self.KernelVersions()
        kernels = self.Kernels()[[
            'KernelId', 'CurrentKernelVersionId', 'AuthorUserId', 'Medal', 'CurrentUrlSlug', 'TotalVotes']]
        users = self.Users()[['UserId', 'PerformanceTier', 'DisplayName']]

        df = pd.merge(kernel_votes, users, on='UserId')
        df = df.merge(kernel_versions[['KernelVersionId', 'KernelId']], on='KernelVersionId')
        df = df.merge(kernels, on='KernelId')
        df = df.merge(kernel_versions[['KernelVersionId', 'KernelLanguageId', 'Title']],
                      left_on='CurrentKernelVersionId', right_on='KernelVersionId')
        df = df.drop(['KernelVersionId_x', 'CurrentKernelVersionId', 'KernelVersionId_y'], axis=1)
        df = df.merge(users, left_on='AuthorUserId', right_on='UserId', suffixes=['Voter', 'Author'])
        df = df.drop(['AuthorUserId', 'Id'], axis=1)
        df = df[df.UserIdVoter != df.UserIdAuthor]
        return df


def date_to_first_day_of_week(day: date) -> date:
    return day - timedelta(days=day.weekday())

start = dt.datetime.now()
### cp ../input/metacover/MetaCoverV1.png .

START_DATE = dt.date(2016, 1, 1)
md = MetaData('../input/meta-kaggle')
user_achievements = md.UserAchievements()
tiers = md.PerformanceTiers()
tier_sizes = user_achievements.groupby(['AchievementType', 'Tier'])[['Id']].count().reset_index()
tier_sizes = tier_sizes.merge(tiers, left_on='Tier', right_on='PerformanceTier')
tier_sizes['AchievementType'] = tier_sizes['AchievementType'].replace('Scripts', 'Kernels')
tier_sizes = tier_sizes.sort_values(by=['AchievementType', 'PerformanceTier'])
tier_sizes = tier_sizes[tier_sizes.PerformanceTier > 1]

grandmasters = user_achievements[user_achievements.Tier == 4]
grandmasters = grandmasters.merge(md.Users(), on='UserId').sort_values(by='AchievementType', ascending=False)
grandmasters[['AchievementType', 'TierAchievementDate', 'UserName', 'DisplayName', 'PerformanceTier']].head(20)
data = [
    go.Bar(x=tier_sizes[tier_sizes['PerformanceTierName'] == tier_name].AchievementType.values,
           y=tier_sizes[tier_sizes['PerformanceTierName'] == tier_name].Id.values,
           marker=dict(color=tier_color),
           name=tier_name)
    for tier, tier_name, tier_color in tiers.values[:5]
]
layout = go.Layout(
    barmode='stack',
    title='Kaggle Performance Tiers',
    xaxis=dict(title='Achievement Type', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of users', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='TierSize')
weekly_kernel_version_votes = md.get_weekly_kernel_version_votes()
weekly_kernel_version_votes = weekly_kernel_version_votes[weekly_kernel_version_votes['VoteWeek'] >= START_DATE]
weekly_kernel_version_votes.shape
weekly_kernel_version_votes.head()

weekly_forum_votes = md.get_weekly_forum_votes()
weekly_forum_votes = weekly_forum_votes[weekly_forum_votes['VoteWeek'] >= START_DATE]
weekly_forum_votes = weekly_forum_votes.sort_values(by='cnt', ascending=False)
weekly_forum_votes.head()

weekly_total_forum_votes = weekly_forum_votes.groupby('VoteWeek')[['cnt']].sum().reset_index()
weekly_total_kernel_votes = weekly_kernel_version_votes.groupby('VoteWeek')[['cnt']].sum().reset_index()
data = [
    go.Scatter(
        x=weekly_total_forum_votes.VoteWeek.values,
        y=weekly_total_forum_votes.cnt.values,
        mode='lines',
        name='Discussion',
        line=dict(width=4, color='#5ac995')
    ),
    go.Scatter(
        x=weekly_total_kernel_votes.VoteWeek.values,
        y=weekly_total_kernel_votes.cnt.values,
        mode='lines',
        name='Kernel',
        line=dict(width=4, color='#007FB4')
    ),
]
layout = go.Layout(
    title='Kernels are getting more popular',
    xaxis=dict(title='WeekStart', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (weekly)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='users')
kernel_versions = md.KernelVersions()
forum_messages = md.ForumMessages()
submissions = md.Submissions()

weekly_submitting_users = submissions.groupby('SubmissionWeek')[['SubmittedUserId']].nunique().reset_index()
weekly_submitting_users = weekly_submitting_users[weekly_submitting_users['SubmissionWeek'] > START_DATE]
weekly_kerneling_users = kernel_versions.groupby('CreationWeek')[['AuthorUserId']].nunique().reset_index()
weekly_kerneling_users = weekly_kerneling_users[weekly_kerneling_users['CreationWeek'] > START_DATE]
weekly_discussing_users = forum_messages.groupby('PostWeek')[['PostUserId']].nunique().reset_index()
weekly_discussing_users = weekly_discussing_users[weekly_discussing_users['PostWeek'] > START_DATE]
data = [
    go.Scatter(
        x=weekly_submitting_users.SubmissionWeek.values,
        y=weekly_submitting_users.SubmittedUserId.values,
        mode='lines',
        name='Competition',
        line=dict(width=4, color='#3E4044')
    ),
    go.Scatter(
        x=weekly_discussing_users.PostWeek.values,
        y=weekly_discussing_users.PostUserId.values,
        mode='lines',
        name='Discussion',
        line=dict(width=4, color='#5ac995')
    ),
    go.Scatter(
        x=weekly_kerneling_users.CreationWeek.values,
        y=weekly_kerneling_users.AuthorUserId.values,
        mode='lines',
        name='Kernel',
        line=dict(width=4, color='#007FB4')
    ),
]
layout = go.Layout(
    title='Weekly Active Users',
    xaxis=dict(title='WeekStart', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of users (weekly)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='weekly_users')
weekly_topic_votes = weekly_forum_votes.groupby(['VoteWeek', 'ForumTopicId', 'TitleTopic', 'TitleForum'])[['cnt']].sum()
weekly_topic_votes = weekly_topic_votes.reset_index()
weekly_topic_votes['TopicRank'] = weekly_topic_votes.groupby('VoteWeek')['cnt'].rank(ascending=False, method='first')
weekly_topic_votes = weekly_topic_votes.sort_values(by='cnt', ascending=False)

weekly_topic_votes.cnt.sum()
weekly_topic_votes.head()
weekly_topic_votes.shape

weekly_top_topics = weekly_topic_votes[weekly_topic_votes.TopicRank == 1]
weekly_top_topics = weekly_top_topics.sort_values(by='VoteWeek', ascending=False)
weekly_top_topics.head()
weekly_top_topics.cnt.sum()
weekly_top_topics.shape

weekly_top_topics['Title'] = weekly_top_topics['TitleTopic'] + ' - ' +  weekly_top_topics['TitleForum']
data = [
    go.Scatter(
        y=weekly_top_topics['cnt'].values,
        x=weekly_top_topics['VoteWeek'].values,
        mode='markers',
        marker=dict(sizemode='diameter',
                    sizeref=1,
                    size=np.sqrt(weekly_top_topics['cnt'].values),
                    color=weekly_top_topics['cnt'].values,
                    colorscale='Viridis',
                    showscale=True
                    ),
        text=weekly_top_topics.Title.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Weekly hottest forum topics',
    hovermode='closest',
    xaxis=dict(title='WeekStart', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (weekly)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='WeeklyTopTopics')
weekly_kernel_votes = weekly_kernel_version_votes.groupby([
    'VoteWeek', 'KernelId', 'KernelLanguageId', 'AuthorUserId', 'Title'
])[['cnt']].sum().reset_index()
weekly_kernel_votes['KernelRank'] = weekly_kernel_votes.groupby('VoteWeek')['cnt'].rank(ascending=False, method='first')
weekly_kernel_votes = weekly_kernel_votes.sort_values(by='cnt', ascending=False)

weekly_kernel_version_votes.shape
weekly_kernel_version_votes.head()
weekly_kernel_votes.shape
weekly_kernel_votes.head()

weekly_kernel_votes = weekly_kernel_votes.merge(md.Users()[['UserId', 'UserName', 'DisplayName']], left_on='AuthorUserId', right_on='UserId')
weekly_kernel_votes['AuthorTitle'] = weekly_kernel_votes['DisplayName'] + ' - ' + weekly_kernel_votes['Title']

weekly_top_kernels = weekly_kernel_votes[weekly_kernel_votes.KernelRank == 1]
weekly_top_kernels = weekly_top_kernels.merge(md.KernelLanguages(), on='KernelLanguageId')
weekly_top_kernels = weekly_top_kernels.sort_values(by='VoteWeek', ascending=False)
weekly_top_kernels.head()
weekly_top_kernels.cnt.sum()
weekly_top_kernels.shape
data = []
for language, language_color in [('Python', '#5ac995'), ('R', '#007FB4')]:
    df = weekly_top_kernels[weekly_top_kernels['KernelLanguageName'] == language]
    data.append(
        go.Scatter(y=df['cnt'].values, x=df['VoteWeek'].values, mode='markers',
                   marker=dict(sizemode='diameter',
                               sizeref=0.7,
                               size=np.sqrt(df['cnt'].values),
                               color=language_color),
                   text=df.AuthorTitle.values,
                   name=language)
    )
layout = go.Layout(
    autosize=True,
    title='Weekly hottest kernels',
    hovermode='closest',
    xaxis=dict(title='WeekStart', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (weekly)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='WeeklyTopKernels')
kernel_info = md.get_kernel_vote_info()

kernel_info['cnt'] = 1
kernel_info.shape
kernel_info.head(3)

top_kernels = kernel_info.groupby([
    'KernelId', 'CurrentUrlSlug', 'Title', 'DisplayNameAuthor'])[['cnt']].sum()
top_kernels = top_kernels.sort_values(by='cnt', ascending=False).reset_index()
top_kernels['KernelRank'] = np.arange(len(top_kernels)) + 1

kernels_tier_vote = kernel_info.groupby(['KernelId', 'PerformanceTierVoter'])[['cnt']].sum()
kernels_tier_vote = kernels_tier_vote.reset_index().pivot('KernelId', 'PerformanceTierVoter', 'cnt')
kernels_tier_vote = kernels_tier_vote.fillna(0).reset_index()

top_kernels = top_kernels.merge(kernels_tier_vote, on='KernelId')
top_kernels['AuthorTitle'] = top_kernels['DisplayNameAuthor'] + ' - ' + top_kernels['Title']
top_kernels.head()

tier_votes = kernel_info.groupby('PerformanceTierVoter')[['cnt']].count().reset_index()
tier_votes = tier_votes.merge(md.PerformanceTiers(), left_on='PerformanceTierVoter', right_on='PerformanceTier')
tier_votes
top_k_kernels = top_kernels[:50]
data = [
    go.Bar(x=[cnt],
           y=[1],
           marker=dict(color=tier_color),
           name=tier_name,
           orientation='h'
           )
    for tier_name, cnt, tier_color in tier_votes[['PerformanceTierName', 'cnt', 'PerformanceTierColor']].values
]
layout = go.Layout(
    barmode='stack',
    height=300,
    title='Votes by voter tier',
    xaxis=dict(title='Votes', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(showgrid=False, zeroline=False, showline=False, showticklabels=False),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='VoterTierVotes')
top_k_kernels = top_kernels[:50]
data = [
    go.Bar(
        y=top_k_kernels['cnt'].values,
        x=top_k_kernels['KernelRank'].values,
        marker=dict(
            color=top_k_kernels['cnt'].values,
            colorscale='Viridis',
            showscale=True
        ),
        text=top_k_kernels.AuthorTitle.values,
    )
]
layout = go.Layout(
    autosize=True,
    title='Alltime hottest kernels',
    hovermode='closest',
    xaxis=dict(title='Rank', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (all time)', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='AlltimeHottestKernels')
kernel_points = kernel_info[kernel_info.PerformanceTierVoter > 0]
kernel_points.shape
kernel_points.head()

top_authors = kernel_points.groupby([
    'UserIdAuthor', 'PerformanceTierAuthor', 'DisplayNameAuthor'])[['cnt']].sum()
top_authors = top_authors.sort_values(by='cnt', ascending=False).reset_index()
top_authors['AuthorRank'] = np.arange(len(top_authors)) + 1
top_authors = top_authors.merge(tiers, left_on='PerformanceTierAuthor', right_on='PerformanceTier')
top_authors = top_authors.sort_values(by='cnt', ascending=False).reset_index()

top_authors.head(3)
top_k_authors = top_authors[:30]
data = [
    go.Bar(
        y=top_k_authors['DisplayNameAuthor'].values,
        x=top_k_authors['cnt'].values,
        marker=dict(color=top_k_authors['PerformanceTierColor'].values),
        orientation='h',
        text=top_k_authors.DisplayNameAuthor.values,
    )
]
layout = go.Layout(
    height=700,
    autosize=True,
    title='Alltime top authors',
    hovermode='closest',
    xaxis=dict(title='Votes', ticklen=5, zeroline=False, gridwidth=2, domain=[0.1, 1]),
    yaxis=dict(title='', ticklen=5, gridwidth=2),
    showlegend=False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='AlltimeTopAuthors')
top_k_authors = top_authors[top_authors.PerformanceTierAuthor < 5]
top_k_authors = top_k_authors[:10]
top_author_points = kernel_points[kernel_points.UserIdAuthor.isin(top_k_authors.UserIdAuthor)]
top_author_points = top_author_points[['cnt', 'VoteDate', 'UserIdAuthor', 'PerformanceTierAuthor', 'DisplayNameAuthor']]
top_author_points['one'] = 1

top_author_points.shape
top_author_points.head()

vote_dates = top_author_points[['VoteDate']].drop_duplicates()
vote_dates['one'] = 1
vote_dates = vote_dates[vote_dates['VoteDate'] > dt.datetime(2017, 1, 1)]
cross_joined_points = pd.merge(top_author_points, vote_dates, on='one', suffixes=['Past', ''])
cross_joined_points = cross_joined_points[cross_joined_points['VoteDatePast'] <= cross_joined_points['VoteDate']]

cross_joined_points.shape
cross_joined_points.head()

cumulative_points = cross_joined_points.groupby(['DisplayNameAuthor', 'VoteDate'])[['one']].sum()
cumulative_points = cumulative_points.reset_index().pivot('VoteDate','DisplayNameAuthor','one').fillna(0)

cumulative_points.head()
cumulative_points.shape
cumulative_points.columns
data = [
    go.Scatter(
        x=cumulative_points.index,
        y=cumulative_points[name].values,
        mode='lines',
        name=name,
        line=dict(width=4)
    ) for name in cumulative_points.columns
]
layout = go.Layout(
    title='Battle for the top',
    xaxis=dict(title='Date', ticklen=5, zeroline=False, gridwidth=2),
    yaxis=dict(title='Number of votes (cumulative)', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='Battle4TheTop')
user_achievements = md.UserAchievements()

kernel_medalists = user_achievements[user_achievements.AchievementType == 'Scripts'].copy()
kernel_medalists['TotalMedal'] = kernel_medalists['TotalGold'] + kernel_medalists['TotalSilver'] + kernel_medalists['TotalBronze']
kernel_medalists = kernel_medalists[kernel_medalists['TotalMedal'] > 0].copy()
kernel_medalists = kernel_medalists.merge(md.Users(), on='UserId')
kernel_medalists = kernel_medalists.sort_values(by=['TotalGold', 'TotalSilver', 'TotalBronze'], ascending=False)

kernel_medalists.shape
kernel_medalists.head()
top_k_medalists = kernel_medalists[:30]
data = [
    go.Bar(
        y=top_k_medalists['DisplayName'].values,
        x=top_k_medalists['Total{}'.format(medal_name)].values,
        marker=dict(color=medal_body,
                    line=dict(color=medal_border, width=0)),
        orientation='h',
        text=top_k_medalists.DisplayName.values,
        name=medal_name
    )
    for medal, medal_name, medal_body, medal_border in md.Medals().values
]
layout = go.Layout(
    height=700,
    barmode='stack',
    autosize=True,
    title='Top Kernel Medalists',
    hovermode='closest',
    xaxis=dict(title='Medals', ticklen=5, zeroline=False, gridwidth=2, domain=[0.05, 1]),
    yaxis=dict(title='', ticklen=5, gridwidth=2),
    showlegend=True
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig, filename='TopMedalists')
end = dt.datetime.now()
print('Latest run {}.\nTotal time {}s'.format(end, (end - start).seconds))

s ='''
L
a
s
t
R
u
n
2
0
1
9
0
4
0
9
'''