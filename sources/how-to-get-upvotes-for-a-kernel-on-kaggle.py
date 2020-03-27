
#import linear algebra and data manipulation libraries
import numpy as np
import pandas as pd

#import date manipulation
import datetime as dt
from datetime import date, timedelta

#import standard visualization
import matplotlib.pyplot as plt
import seaborn as sns

#import machine learning
from sklearn.linear_model import LinearRegression
import xgboost

from sklearn.model_selection import train_test_split #split
from sklearn.metrics import r2_score, mean_squared_error #metrics

# import helpers
import re
kernels = pd.read_csv('../input/Kernels.csv')
kernel_lang = pd.read_csv('../input/KernelLanguages.csv')
kernel_tags = pd.read_csv('../input/KernelTags.csv')
kernel_votes = pd.read_csv('../input/KernelVotes.csv')
kernel_versions = pd.read_csv('../input/KernelVersions.csv')
kernel_competitions = pd.read_csv('../input/KernelVersionCompetitionSources.csv')
kernel_users = pd.read_csv('../input/Users.csv')
datasets = pd.read_csv('../input/Datasets.csv')
kernel_datasets = pd.read_csv('../input/KernelVersionDatasetSources.csv')
dataset_versions = pd.read_csv('../input/DatasetVersions.csv')
tags = pd.read_csv('../input/Tags.csv')
competitions = pd.read_csv('../input/Competitions.csv')
messages = pd.read_csv('../input/ForumMessages.csv')
total_kernels = len(kernels['Id'].unique())
print('Total number of kernels: {total_kernels}'.format(total_kernels = total_kernels))

total_public = len(kernels[pd.notnull(kernels['MadePublicDate'])])
print('Total number of public kernels: {total_public_kernels}'.format(total_public_kernels = total_public))

total_upvoted = len(kernels[ kernels['TotalVotes'] > 0])
print('Total number of kernels with upvotes: {total_upvoted}'.format(total_upvoted = total_upvoted))

total_awarded = len(kernels[np.isfinite(kernels['Medal'])])
print('Total number of kernels with any medal: {total_awarded}'.format(total_awarded = total_awarded))

total_no_upvotes = total_public - total_upvoted
print('Total number of public kernels with no upvotes: {total_no_upvotes}'.format(total_no_upvotes = total_no_upvotes))

total_private = total_kernels - total_public
print('Total number of private kernels: {total_private}'.format(total_private = total_private))

total_upvoted_not_awarded = total_upvoted - total_awarded
print('Total number of upvoted, but not awarded: {total_upvoted_not_awarded}'.format(total_upvoted_not_awarded = total_upvoted_not_awarded))
#plot statistics with matplotlib

labels = 'Private', 'Public with no upvotes', 'Upvoted but not awarded', 'Awarded'
sizes = [total_private, total_no_upvotes, total_upvoted_not_awarded, total_awarded]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=180)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

fig1.set_figheight(7)
fig1.set_figwidth(7)

plt.title('Statistics for upvoted and awarded kernels')

plt.show()
#columns to find out correlation
cols = ['TotalViews', 'TotalComments', 'TotalVotes']

#Find out correlation between columns and plot
corrs = np.corrcoef(kernels[cols].values.T)
sns.set(font_scale=1)
sns.set(rc={'figure.figsize':(7,7)})
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = cols, xticklabels = cols).set_title('Correlations heatmap')

fig = hm.get_figure()
fig.savefig('correlations.png')
#get average number of votes per number of views
votes_by_views = kernels.groupby('TotalViews').mean().reset_index()[['TotalViews', 'TotalVotes']]
votes_by_views = votes_by_views.sort_values(by = 'TotalViews', ascending = True)

# fit a linear regression model on views and votes data
model = LinearRegression()
model.fit(votes_by_views['TotalViews'].values.reshape(-1, 1), votes_by_views['TotalVotes'].values.reshape(-1, 1))

inputs = np.linspace(1, 1200000, num=50)
predictions = model.predict(inputs.reshape(-1,1))
#plot with matplotlib along with predictions from the linear regression model

plt.scatter(votes_by_views['TotalViews'], votes_by_views['TotalVotes'])
plt.plot(inputs, predictions, color='green')

plt.title("Number of votes per number of views")
plt.xlabel("Views")
plt.ylabel("Votes")

plt.show()
#get average number of votes per number of comments
votes_by_comments = kernels.groupby('TotalComments').mean().reset_index()[['TotalComments', 'TotalVotes']]
votes_by_comments = votes_by_comments.sort_values(by = 'TotalComments', ascending = True)

# fit a linear regression model on comments and votes data
model = LinearRegression()
model.fit(votes_by_comments['TotalComments'].values.reshape(-1, 1), votes_by_comments['TotalVotes'].values.reshape(-1, 1))

inputs = np.linspace(1, 1000, num=50)
predictions = model.predict(inputs.reshape(-1,1))
#plot with matplotlib along with predictions from the linear regression model

plt.scatter(votes_by_comments['TotalComments'], votes_by_comments['TotalVotes'])
plt.plot(inputs, predictions, color='green')

plt.title("Number of votes per number of views")
plt.xlabel("Comments")
plt.ylabel("Votes")

plt.show()
#merge kernels with users
kernel_users_renamed = kernel_users.rename(columns={'Id': 'AuthorUserId'})
kernels_user_tiers = kernels.merge(kernel_users_renamed, on = 'AuthorUserId')

#get average number of votes per number of comments
votes_by_usertier= kernels_user_tiers.groupby('PerformanceTier').mean().reset_index()[['PerformanceTier', 'TotalVotes']]
votes_by_usertier = votes_by_usertier.sort_values(by = 'PerformanceTier', ascending = True)
#plot with matplotlib along with predictions from the linear regression model

plt.plot(votes_by_usertier['PerformanceTier'], votes_by_usertier['TotalVotes'])

plt.title("Average number of votes per kernel depending on user performance tier")
plt.xlabel("Performance Tier")
plt.ylabel("Average number of votes per kernel")

plt.show()
#prepare data for the analysis

kernels_truncated = kernels[['Id' , 'CurrentKernelVersionId', 'TotalVotes']]
kernels_truncated = kernels_truncated.rename(columns={'Id': 'KernelId', 'TotalVotes':'TotalKernelVotes',
                                                      'CurrentKernelVersionId':'KernelVersionId'})

kernel_datasets_truncated = kernel_datasets[['KernelVersionId', 'SourceDatasetVersionId']]

dataset_versions_truncated = dataset_versions[['DatasetId', 'DatasourceVersionId']]
dataset_versions_truncated = dataset_versions_truncated.rename(columns={'DatasourceVersionId': 'SourceDatasetVersionId'})

datasets_truncated = datasets[['Id', 'TotalVotes', 'TotalDownloads']]
datasets_truncated = datasets_truncated.rename(columns={'Id': 'DatasetId', 'TotalVotes':'TotalDatasetVotes',
                                                      'TotalDownloads':'TotalDatasetDownloads'})

df_merge = kernels_truncated.merge(kernel_datasets_truncated, on = 'KernelVersionId')
df_merge = df_merge.merge(dataset_versions_truncated, on = 'SourceDatasetVersionId')
df_merge = df_merge.merge(datasets_truncated, on = 'DatasetId')
#columns to find out correlation
cols = ['TotalKernelVotes', 'TotalDatasetVotes', 'TotalDatasetDownloads']

#Find out correlation between columns and plot
corrs = np.corrcoef(df_merge[cols].values.T)
sns.set(font_scale=1)
sns.set(rc={'figure.figsize':(7,7)})
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = cols, xticklabels = cols).set_title('Correlations heatmap')

plt.yticks(rotation=0)
fig = hm.get_figure()
fig.savefig('correlations.png')
#get average number of votes per number of dataset votes
votes_by_dvotes = df_merge.groupby('TotalDatasetVotes').mean().reset_index()[['TotalDatasetVotes', 'TotalKernelVotes']]
votes_by_dvotes = votes_by_dvotes.sort_values(by = 'TotalDatasetVotes', ascending = True)
#plot with matplotlib

plt.scatter(votes_by_dvotes['TotalDatasetVotes'], votes_by_dvotes['TotalKernelVotes'])

plt.title("Number of votes per number of votes for related dataset")
plt.xlabel("Dataset Votes")
plt.ylabel("Kernel Votes")

plt.show()
#get average number of votes per number of dataset downloads
votes_by_downloads = df_merge.groupby('TotalDatasetDownloads').mean().reset_index()[['TotalDatasetDownloads', 'TotalKernelVotes']]
votes_by_downloads = votes_by_downloads.sort_values(by = 'TotalDatasetDownloads', ascending = True)
#plot with matplotlib

plt.scatter(votes_by_downloads['TotalDatasetDownloads'], votes_by_downloads['TotalKernelVotes'])

plt.title("Number of votes per number of downloads for related dataset")
plt.xlabel("Dataset Downloads")
plt.ylabel("Kernel Votes")

plt.show()
#prepare the data for the analysis

kernels_trc = kernels[['Id' , 'TotalVotes', 'CurrentKernelVersionId']]
kernels_trc = kernels_trc.rename(columns = {'Id': 'KernelId', 'CurrentKernelVersionId' : 'VersionId'})

kernel_versions_trc = kernel_versions[['Id', 'KernelId', 'KernelLanguageId']]
kernel_versions_trc = kernel_versions_trc.rename(columns = {'Id' : 'VersionId'})

kernel_lang_ren = kernel_lang.rename(columns = {'Id' : 'KernelLanguageId'})

kernels_trc_lang = kernels_trc.merge(kernel_versions_trc, on = ['VersionId', 'KernelId'])
kernels_trc_lang = kernels_trc_lang.merge(kernel_lang_ren, on = 'KernelLanguageId')
#get average number of votes per language name
votes_by_lang = kernels_trc_lang.groupby('Name').mean().reset_index()[['Name', 'TotalVotes']]
votes_by_lang = votes_by_lang.sort_values(by = 'Name', ascending = True)
votes_by_lang = votes_by_lang.rename(columns = {'TotalVotes' : 'Average Votes'})
#plot bar chart with matplotlib

plt.figure(figsize=(13,7))
plt.bar(votes_by_lang['Name'], votes_by_lang['Average Votes'])
plt.xticks(votes_by_lang['Name'])
plt.xticks(rotation=60)

plt.xlabel('Language Name')
plt.ylabel('Average Number of Votes')

plt.title('Average number of votes per kernel language')

plt.show()
#prepare data

kernels_trc_tag = kernels[['Id' , 'TotalVotes']]
kernels_trc_tag = kernels_trc_tag.rename(columns = {'Id' : 'KernelId'})

tags_trc = tags[['Id', 'Name', 'DatasetCount', 'CompetitionCount', 'KernelCount']]
tags_trc = tags_trc.rename(columns = {'Id' : 'TagId'})

kernels_trc_tag = kernels_trc_tag.merge(kernel_tags[['KernelId', 'TagId']], on='KernelId')
kernels_trc_tag = kernels_trc_tag.merge(tags_trc, on = 'TagId')
#get average number of votes per number of tage usages in datasets
votes_by_tag_dataset = kernels_trc_tag.groupby('DatasetCount').mean().reset_index()[['DatasetCount', 'TotalVotes']]
votes_by_tag_dataset = votes_by_tag_dataset.sort_values(by = 'DatasetCount', ascending = True)
#plot with matplotlib

plt.scatter(votes_by_tag_dataset['DatasetCount'], votes_by_tag_dataset['TotalVotes'])
plt.title("Number of votes per number of tag usages in datasets")
plt.xlabel("Tag usage for datasets")
plt.ylabel("Votes")

plt.show()
#get average number of votes per number of tage usages in competitions
votes_by_tag_competition = kernels_trc_tag.groupby('CompetitionCount').mean().reset_index()[['CompetitionCount', 'TotalVotes']]
votes_by_tag_competition = votes_by_tag_competition.sort_values(by = 'CompetitionCount', ascending = True)
#plot with matplotlib

plt.scatter(votes_by_tag_competition['CompetitionCount'], votes_by_tag_competition['TotalVotes'])
plt.title("Number of votes per number of tag usages in competitions")
plt.xlabel("Tag usage for competitions")
plt.ylabel("Votes")

plt.show()
#get average number of votes per number of tage usages in kernels
votes_by_tag_kernel = kernels_trc_tag.groupby('KernelCount').mean().reset_index()[['KernelCount', 'TotalVotes']]
votes_by_tag_kernel = votes_by_tag_kernel.sort_values(by = 'KernelCount', ascending = True)
#plot with matplotlib

plt.scatter(votes_by_tag_kernel['KernelCount'], votes_by_tag_kernel['TotalVotes'])
plt.title("Number of votes per number of tag usages in kernels")
plt.xlabel("Tag usage for kernels")
plt.ylabel("Votes")

plt.show()
#find out total usage of the tag and store in TotalCount column
tags_total = tags_trc.copy()

tags_total['TotalCount'] = tags_total['DatasetCount'] + tags_total['CompetitionCount'] + tags_total['KernelCount']

#find out top-20 most popular tags
tags_total = tags_total.sort_values('TotalCount', ascending = False)
tags_total = tags_total[:20]

#find average number of votes for each tag
votes_by_tag = kernels_trc_tag.groupby('Name').mean().reset_index()[['Name', 'TotalVotes']]

#filter votes by tag for only top-20 most popular tags
votes_by_tag_popular = votes_by_tag[votes_by_tag['Name'].isin(tags_total['Name'].values.tolist())]
#plot bar chart with matplotlib

plt.figure(figsize=(20,7))
plt.barh(votes_by_tag_popular['Name'], votes_by_tag_popular['TotalVotes'])
plt.yticks(votes_by_tag_popular['Name'])

plt.ylabel('Tag Name')
plt.xlabel('Average Number of Votes')

plt.title('Average number of votes per tag')

plt.show()
#sort votes_by_tag by average number of votes and take top-20

votes_by_tag_aver = votes_by_tag.sort_values('TotalVotes', ascending = False)[:20]
#plot bar chart with matplotlib

plt.figure(figsize=(20,7))
plt.barh(votes_by_tag_aver['Name'], votes_by_tag_aver['TotalVotes'])
plt.yticks(votes_by_tag_aver['Name'])

plt.ylabel('Tag Name')
plt.xlabel('Average Number of Votes')

plt.title('Average number of votes per tag')

plt.show()
# create msg_ref dataframe by removing rows with empty messages from the original dataset
msg_ref = messages[['Id', 'Message']].dropna(subset = ['Message'])

# filter rows: find rows with messages which contain references to kaggle
msg_ref = msg_ref[msg_ref.Message.str.match('.*https://www.kaggle.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+.*')]
def get_referenced_object(row):
    '''
    Function returns the last part of the url.
    Example:
        input: 'https://www.kaggle.com/aleksandradeis/how-to-get-upvotes-for-a-kernel-on-kaggle'
        returns: 'how-to-get-upvotes-for-a-kernel-on-kaggle'
    '''
    # extract url from message
    url = re.findall('https://www.kaggle.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'\
                         , row['Message'])[0]
    
    # remove tags
    cleanr = re.compile('<.*?>')
    clean_url= re.sub(cleanr, '', url)
    
    # get the last part of url, which corresponds to kernel title
    ref = clean_url.split('/')[-1]
    
    return ref

def get_author(row):
    '''
    Function returns the last part of the url.
    Example:
        input: 'https://www.kaggle.com/aleksandradeis/how-to-get-upvotes-for-a-kernel-on-kaggle'
        returns: 'aleksandradeis'
    '''
    # extract url from message
    url = re.findall('https://www.kaggle.com/(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'\
                         , row['Message'])[0]
    
    # remove tags
    cleanr = re.compile('<.*?>')
    clean_url= re.sub(cleanr, '', url)
    
    # get the last part of url, which corresponds to kernel title
    author = clean_url.split('/')[-2]
    
    return author

# create column 'Ref' with the last part of the url
# which is the title of the kernel (in case of url to a kernel)
msg_ref['Ref'] = msg_ref.apply(lambda row: get_referenced_object(row),axis=1)
msg_ref['Ref_author'] = msg_ref.apply(lambda row: get_author(row),axis=1)
msg_ref.head()
# merge kernel votes and kernel titles (for current kernel version)
kernel_title = kernels[['Id', 'AuthorUserId', 'CurrentKernelVersionId', 'TotalVotes']]\
.merge(kernel_versions[['Id', 'Title']].rename(columns={"Id": "CurrentKernelVersionId"}))
# merge kernel votes and kernel authors
kernel_title = kernel_title.merge(kernel_users[['Id', 'UserName']].rename(columns={"Id":"AuthorUserId"}))
def process_kernel_title(row):
    '''
    Function removes symbols from title, translates to lower case,
    and replaces spaces with '-'
    Example:
        input: 'How to get upvotes for a kernel on Kaggle?'
        returns: 'how-to-get-upvotes-for-a-kernel-on-kaggle'
    '''
    try:
        title = row['Title']

        # remove specific symbols and translate to lower case
        title_cleaned = re.sub('[!@#$%?*,.;:()]', '', title).lower()

        # replace spaces with '-'
        title_cleaned = title_cleaned.replace(' ', '-')
    except:
        return ' '
    return title_cleaned

# create column title_cleaned with preprocessed title as in url
kernel_title['title_cleaned'] = kernel_title.apply(lambda row: process_kernel_title(row),axis=1)
kernel_title.head()
# merge kernels and messages (outer)
promotions = pd.merge(msg_ref, kernel_title, how='outer', left_on='Ref', right_on='title_cleaned',
         suffixes=('_messages', '_kernels'), copy=True)

# remove rows with NaN
promotions = promotions.dropna()

# remove rows with title_cleaned in ['', 'competitions', 'rules']
promotions = promotions[~promotions['title_cleaned'].isin(['', 'competitions', 'rules'])]

# remove rows where authors don't match

promotions = promotions[promotions['Ref_author'] == promotions['UserName']]
promotions.head()
n_promotions = promotions[['Id_kernels', 'CurrentKernelVersionId', 'TotalVotes']]\
.groupby('Id_kernels').count().reset_index()[["Id_kernels", "CurrentKernelVersionId"]]\
.rename(columns = {'CurrentKernelVersionId' : 'Promotions'})
n_kernel_promotions = kernels[['Id', 'TotalVotes', 'Medal']].merge(n_promotions.rename(columns = {'Id_kernels' : 'Id'})\
                                                          , how = 'outer')
n_kernel_promotions = n_kernel_promotions.fillna(0)
# fit a linear regression model on mentions and votes data
model = LinearRegression()
model.fit(n_kernel_promotions['Promotions'].values.reshape(-1, 1), n_kernel_promotions['TotalVotes'].values.reshape(-1, 1))

inputs = np.linspace(1, 20, num=50)
predictions = model.predict(inputs.reshape(-1,1)).reshape(50)
#plot with matplotlib

plt.scatter(n_kernel_promotions['Promotions'], n_kernel_promotions['TotalVotes'])
plt.plot(inputs, predictions, color='green')

plt.title("Number of votes per number of mentions")
plt.xlabel("Number of mentions in Kaggle forum messages")
plt.ylabel("Votes")

plt.show()
#columns to find out correlation
cols = ['TotalVotes', 'Promotions']

#Find out correlation between columns and plot
corrs = np.corrcoef(n_kernel_promotions[cols].values.T)
sns.set(font_scale=1)
sns.set(rc={'figure.figsize':(7,7)})
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = cols, xticklabels = cols).set_title('Correlations heatmap')

fig = hm.get_figure()
#columns to find out correlation
cols = ['Medal', 'Promotions']

#Find out correlation between columns and plot
corrs = np.corrcoef(n_kernel_promotions[cols].values.T)
sns.set(font_scale=1)
sns.set(rc={'figure.figsize':(7,7)})
hm=sns.heatmap(corrs, cbar = True, annot=True, square = True, fmt = '.2f',
              yticklabels = cols, xticklabels = cols).set_title('Correlations heatmap (medals)')

fig = hm.get_figure()
votes_with_mentions = n_kernel_promotions[n_kernel_promotions['Promotions'] > 0]['TotalVotes'].mean()
print('Average number of votes for advertised kernels: {}'.format(votes_with_mentions))

kernels_with_promotions = n_kernel_promotions[n_kernel_promotions['Promotions'] > 0]['Promotions'].count()
print('Total number of kernels mentioned in forum messages: {}'.format(kernels_with_promotions))

votes_no_mentions = n_kernel_promotions[n_kernel_promotions['Promotions'] == 0]['TotalVotes'].mean()
print('Average number of votes for non advertised kernels: {}'.format(votes_no_mentions))
medals_with_mentions = n_kernel_promotions[n_kernel_promotions['Promotions'] > 0]['Medal'].mean()
print('Average medal rank for advertised kernels: {}'.format(medals_with_mentions))

medals_no_mentions = n_kernel_promotions[n_kernel_promotions['Promotions'] == 0]['Medal'].mean()
print('Average medal rank for non advertised kernels: {}'.format(medals_no_mentions))
#prepare data

def has_tag(row, tag):
    ''' Function returns 1 of tag id found in row['Tags'] column '''
    
    for row_tag in row['Tags'].replace(' ','').replace('{','').replace('}','').split(','):
        if row_tag == tag:
            return 1
    return 0

def is_forked(row):
    ''' Functions returns 0 if row['ForkParentKernelVersionId'] is nan, 1 - otherwise '''
    if np.isnan(row['ForkParentKernelVersionId']):
        return 0
    else:
        return 1

def has_linked_topic(row):
    ''' Functions returns 0 if row['ForumTopicId'] is nan, 1 - otherwise '''
    if np.isnan(row['ForumTopicId']):
        return 0
    else:
        return 1
    
def get_days_from_registration(row):
    '''Function returns number of days since users registration date'''
    
    today = dt.datetime.now().date()
    days = (today - dt.datetime.strptime(row['RegisterDate'], "%m/%d/%Y").date()).days
    
    return days

def get_days_from_kernel_creation(row):
    '''Function returns number of days since kernel creation date'''
    
    today = dt.datetime.now().date()
    
    days = (today - dt.datetime.strptime(row['CreationDate'], "%m/%d/%Y %H:%M:%S").date()).days
    
    return days

def prepare_data(kernels, kernel_lang, kernel_tags, kernel_versions, kernel_users, kernel_datasets, datasets, dataset_versions, tags, competitions, kernel_competitions):
    '''
    Function to prepare meta Kaggle data to be used in regression models
    
    INPUTS:
    1. kernels - Kernels.csv dataset, 
    2. kernel_lang - KernelLanguages.csv dataset,
    3. kernel_tags - KernelTags.csv dataset,
    4. kernel_versions - KernelVersions.csv dataset,
    5. kernel_users - KernelUsers.csv dataset,
    6. kernel_datasets - KernelVersionDatasetSources.csv dataset,
    7. datasets - Datasets.csv dataset,
    8. dataset_versions - DatasetVersions.csv dataset,
    9. tags - Tags.csv dataset.
    
    RETURNS:
    df - dataset, wchich contains preprocessed data
    '''
    #1. merge datasets
    
    #remove irrelevant columns from dataset and rename other columns to perform merge
    df = kernels[['Id', 'CurrentKernelVersionId' , 'ForkParentKernelVersionId', 'ForumTopicId', 'TotalVotes', 'AuthorUserId', 'CreationDate']].rename(columns = 
                                                                                                                      {'Id': 'KernelId', 
                                                                                                           'TotalVotes': 'TotalKernelVotes', 
                                                                                                                      'CurrentKernelVersionId':'KernelVersionId'})
    
    # create new column for age of the kernel (number of days since the kernel creation)
    # and drop old column CreationDate
    df[['CreationDate']] = df[['CreationDate']].fillna(value=dt.datetime.now().strftime('%m/%d/%Y %H:%M:%S'))
    df['KernelAge'] = df.apply(lambda row: get_days_from_kernel_creation(row),axis=1)
    df = df.drop('CreationDate', axis = 1)
    
    #merge with users to add PerformanceTier of the author and number of days since author's registration on Kaggle
    kernel_users['DaysSinceRegistration'] = kernel_users.apply(lambda row: get_days_from_registration(row),axis=1)
    kernel_users_renamed = kernel_users[['Id', 'PerformanceTier', 'DaysSinceRegistration']].rename(columns={'Id': 'AuthorUserId',
                                                                                  'PerformanceTier' : 'AuthorPerformanceTier'})
    df = df.merge(kernel_users_renamed, on = 'AuthorUserId').drop('AuthorUserId', axis = 1)
    
    #merge with versions to add VersionNumber and KernelLanguageId
    kernel_versions_renamed = kernel_versions[['Id', 'KernelId', 'KernelLanguageId', 'VersionNumber']].rename(columns = {'Id':'KernelVersionId'})
    df = df.merge(kernel_versions_renamed, on = ['KernelVersionId', 'KernelId'])
    
    #merge with datasets to add TotalDatasetVotes and TotalDatasetDownloads
    kernel_datasets_truncated = kernel_datasets[['KernelVersionId', 'SourceDatasetVersionId']]

    dataset_versions_truncated = dataset_versions[['DatasetId', 'DatasourceVersionId']]
    dataset_versions_truncated = dataset_versions_truncated.rename(columns={'DatasourceVersionId': 'SourceDatasetVersionId'})

    datasets_truncated = datasets[['Id', 'TotalVotes', 'TotalDownloads']]
    datasets_truncated = datasets_truncated.rename(columns={'Id': 'DatasetId', 'TotalVotes':'TotalDatasetVotes',
                                                      'TotalDownloads':'TotalDatasetDownloads'})

    df = df.merge(kernel_datasets_truncated, on = 'KernelVersionId')
    df = df.merge(dataset_versions_truncated, on = 'SourceDatasetVersionId')
    df = df.merge(datasets_truncated, on = 'DatasetId').drop(['SourceDatasetVersionId', 'DatasetId'], axis = 1)
    
    # merge with type of competition
    # if kernel is not linked to a competition - NaN
    competitions_type = competitions[['Id', 'HostSegmentTitle']].rename(columns={'Id': 'SourceCompetitionId'})
    kernel_competition_type = kernel_competitions.merge(competitions_type, on='SourceCompetitionId')[['KernelVersionId', 'HostSegmentTitle']] # merge kernel versions with competitions
    df = df.merge(kernel_competition_type, on = 'KernelVersionId', how='outer')
    df = df.drop('KernelVersionId', axis = 1)
    
    #create dummy variables for competition types and drop HostSegmentTitle
    df = pd.get_dummies(df, columns = ['HostSegmentTitle'], prefix = 'comp', dummy_na = True)
       
    #merge with tags
    tags_trc = tags[['Id', 'Name']]
    tags_trc = tags_trc.rename(columns = {'Id' : 'TagId', 'Name' : 'TagName'})
    df = df.merge(kernel_tags, on = 'KernelId').drop(['Id'], axis = 1)
    df = df.merge(tags_trc, on = 'TagId')
    
    #find most popular tags
    top20_tags = tags[['Id', 'Name', 'DatasetCount', 'CompetitionCount', 'KernelCount']].rename(columns={'Id':'TagId'})
    top20_tags['TotalCount'] = top20_tags['DatasetCount'] + top20_tags['CompetitionCount'] + top20_tags['KernelCount']
    top20_tags = top20_tags.sort_values('TotalCount', ascending = False)
    top20_tags = top20_tags[:20].drop(['DatasetCount', 'CompetitionCount' , 'KernelCount'], axis = 1)
    
    #filter-out rows only for most popular tags
    df = df[df['TagId'].isin(top20_tags['TagId'].values.tolist())]
    df = df.drop('TagId', axis = 1)
    
    #introduce column 'Tags' with set of most popular tags used for kernel
    df_1 = df.groupby('KernelId')['TagName'].apply(lambda x: "{%s}" % ', '.join(x)).reset_index().rename(columns = {'TagName':'Tags'})
    df = df.drop('TagName', axis = 1)
    df = df_1.merge(df, on='KernelId')
    
    #create dummy columns for top-20 tags
    for top20_tag in top20_tags['Name'].values.tolist():
        df['tag_' + str(top20_tag)] = df.apply(lambda row: has_tag(row, top20_tag),axis=1)
        
    df = df.drop('Tags', axis = 1)
    
    #2. process column values
    
    #introduce column IsForked for kernels forked from other and drop original column
    df['IsForked'] = df.apply(lambda row: is_forked(row),axis=1)
    df = df.drop('ForkParentKernelVersionId', axis = 1)
    
    #introduce colum HasForumTopic for kernels linked to forum topic id and drop original column
    df['HasForumTopic'] = df.apply(lambda row: has_linked_topic(row),axis=1)
    df = df.drop(['ForumTopicId', 'KernelId'], axis = 1)
    
     
    return df

df = prepare_data(kernels, kernel_lang, kernel_tags, kernel_versions, kernel_users, kernel_datasets, datasets, dataset_versions, tags, competitions, kernel_competitions)

df.head(5)
#split into train and test
TEST_SIZE = 0.3
RAND_STATE = 42

X = df.drop(columns = 'TotalKernelVotes')
y = df[['TotalKernelVotes']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = TEST_SIZE, random_state=RAND_STATE)
#train XGBoost model
xgb = xgboost.XGBRegressor(n_estimators=100, learning_rate=0.08, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=7)
xgb.fit(X_train,y_train)

#calculate and print scores for the model for top 15 features
y_train_preds = xgb.predict(X_train)
y_test_preds = xgb.predict(X_test)

print('MSE train: %.3f, test: %.3f' % (
        mean_squared_error(y_train, y_train_preds),
        mean_squared_error(y_test, y_test_preds)))
print('R^2 train: %.3f, test: %.3f' % (
        r2_score(y_train, y_train_preds),
        r2_score(y_test, y_test_preds)))
#plot bar chart with matplotlib
plt.figure(figsize=(7,10))

y_pos = np.arange(len(X.columns))

plt.barh(y_pos, xgb.feature_importances_, align='center', alpha=0.5)
plt.yticks(y_pos, X.columns)

plt.ylabel('Features')
plt.xlabel('Feature Importance')

plt.title('Feature importances')

plt.show()
data = np.array([30, 2, 365, 8, 1, 30, 2070, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
columns = X_train.columns.values

bm_kernel = pd.DataFrame(data = [data], columns = columns)
predict_votes = xgb.predict(bm_kernel)
predict_votes