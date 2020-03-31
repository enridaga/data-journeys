
import os
import re
import math
import random
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
import networkx as nx

from wordcloud import WordCloud
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import spacy
nlp = spacy.load('en')
nlp.remove_pipe('parser')
nlp.remove_pipe('ner')
#nlp.remove_pipe('tagger')

import gensim
import pyLDAvis
import pyLDAvis.gensim
pyLDAvis.enable_notebook()
input_dir = '../input'
#print(os.listdir(input_dir))

professionals = pd.read_csv(os.path.join(input_dir, 'professionals.csv'))
groups = pd.read_csv(os.path.join(input_dir, 'groups.csv'))
comments = pd.read_csv(os.path.join(input_dir, 'comments.csv'))
school_memberships = pd.read_csv(os.path.join(input_dir, 'school_memberships.csv'))
tags = pd.read_csv(os.path.join(input_dir, 'tags.csv'))
emails = pd.read_csv(os.path.join(input_dir, 'emails.csv'))
group_memberships = pd.read_csv(os.path.join(input_dir, 'group_memberships.csv'))
answers = pd.read_csv(os.path.join(input_dir, 'answers.csv'))
students = pd.read_csv(os.path.join(input_dir, 'students.csv'))
matches = pd.read_csv(os.path.join(input_dir, 'matches.csv'))
questions = pd.read_csv(os.path.join(input_dir, 'questions.csv'))
tag_users = pd.read_csv(os.path.join(input_dir, 'tag_users.csv'))
tag_questions = pd.read_csv(os.path.join(input_dir, 'tag_questions.csv'))
answer_scores = pd.read_csv(os.path.join(input_dir, 'answer_scores.csv'))
question_scores = pd.read_csv(os.path.join(input_dir, 'question_scores.csv'))
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', -1)

seed = 13
random.seed(seed)
np.random.seed(seed)
# Spacy Tokenfilter for part-of-speech tagging
token_pos = ['NOUN', 'VERB', 'PROPN', 'ADJ', 'INTJ', 'X']

# The data export was from 1. February 2019. For Production use datetime.now()
actual_date = datetime(2019, 2 ,1)
def nlp_preprocessing(data):
    """ Use NLP to transform the text corpus to cleaned sentences and word tokens

        param data: List with sentences, which should be processed.
        return processed_tokens: List with the cleaned and tokenized sentences
    """    
    def token_filter(token):
        """ Keep tokens who are alphapetic, in the pos (part-of-speech) list and not in stop list
            
        """    
        return not token.is_stop and token.is_alpha and token.pos_ in token_pos
    
    data = [re.compile(r'<[^>]+>').sub('', x) for x in data] #Remove HTML-tags
    processed_tokens = []
    data_pipe = nlp.pipe(data)
    for doc in data_pipe:
        filtered_tokens = [token.lemma_.lower() for token in doc if token_filter(token)]
        processed_tokens.append(filtered_tokens)
    return processed_tokens
### %time
# Transform datetime datatypes
questions['questions_date_added'] = pd.to_datetime(questions['questions_date_added'], infer_datetime_format=True)
answers['answers_date_added'] = pd.to_datetime(answers['answers_date_added'], infer_datetime_format=True)
professionals['professionals_date_joined'] = pd.to_datetime(professionals['professionals_date_joined'], infer_datetime_format=True)
students['students_date_joined'] = pd.to_datetime(students['students_date_joined'], infer_datetime_format=True)
emails['emails_date_sent'] = pd.to_datetime(emails['emails_date_sent'], infer_datetime_format=True)
comments['comments_date_added'] = pd.to_datetime(comments['comments_date_added'], infer_datetime_format=True)

### Questions
# Merge Question Title and Body
questions['questions_full_text'] = questions['questions_title'] +'\r\n\r\n'+ questions['questions_body']
# Count of answers
temp = answers.groupby('answers_question_id').size()
questions['questions_answers_count'] = pd.merge(questions, pd.DataFrame(temp.rename('count')), left_on='questions_id', right_index=True, how='left')['count'].fillna(0).astype(int)
# First answer for questions
temp = answers[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').min()
questions['questions_first_answers'] = pd.merge(questions, pd.DataFrame(temp), left_on='questions_id', right_index=True, how='left')['answers_date_added']
# Last answer for questions
temp = answers[['answers_question_id', 'answers_date_added']].groupby('answers_question_id').max()
questions['questions_last_answers'] = pd.merge(questions, pd.DataFrame(temp), left_on='questions_id', right_index=True, how='left')['answers_date_added']
# Hearts Score
temp = pd.merge(questions, question_scores, left_on='questions_id', right_on='id', how='left')
questions['questions_hearts'] = temp['score'].fillna(0).astype(int)
# Questions Tags list
temp = pd.merge(questions, tag_questions, left_on='questions_id', right_on='tag_questions_question_id', how='inner')
temp = pd.merge(temp, tags, left_on='tag_questions_tag_id', right_on='tags_tag_id', how='inner')
temp = temp.groupby('questions_id')['tags_tag_name'].apply(list).rename('questions_tags')
questions['questions_tags'] = pd.merge(questions, temp.to_frame(), left_on='questions_id', right_index=True, how='left')['questions_tags']
# Get NLP Tokens
questions['nlp_tokens'] = nlp_preprocessing(questions['questions_full_text'])

### Answers
# Days required to answer the question
temp = pd.merge(questions, answers, left_on='questions_id', right_on='answers_question_id')
answers['time_delta_answer'] = (temp['answers_date_added'] - temp['questions_date_added'])
# Ranking for answers time
answers['answers_time_rank'] = answers.groupby('answers_question_id')['time_delta_answer'].rank(method='min').astype(int)
# Hearts Score
temp = pd.merge(answers, answer_scores, left_on='answers_id', right_on='id', how='left')
answers['answers_hearts'] = temp['score'].fillna(0).astype(int)

### Professionals
# Time since joining
professionals['professionals_time_delta_joined'] = actual_date - professionals['professionals_date_joined']
# Number of answers
temp = answers.groupby('answers_author_id').size()
professionals['professionals_answers_count'] = pd.merge(professionals, pd.DataFrame(temp.rename('count')), left_on='professionals_id', right_index=True, how='left')['count'].fillna(0).astype(int)
# Number of comments
temp = comments.groupby('comments_author_id').size()
professionals['professionals_comments_count'] = pd.merge(professionals, pd.DataFrame(temp.rename('count')), left_on='professionals_id', right_index=True, how='left')['count'].fillna(0).astype(int)
# Last activity (Answer)
temp = answers.groupby('answers_author_id')['answers_date_added'].max()
professionals['date_last_answer'] = pd.merge(professionals, pd.DataFrame(temp.rename('last_answer')), left_on='professionals_id', right_index=True, how='left')['last_answer']
# First activity (Answer)
temp = answers.groupby('answers_author_id')['answers_date_added'].min()
professionals['date_first_answer'] = pd.merge(professionals, pd.DataFrame(temp.rename('first_answer')), left_on='professionals_id', right_index=True, how='left')['first_answer']
# Last activity (Comment)
temp = comments.groupby('comments_author_id')['comments_date_added'].max()
professionals['date_last_comment'] = pd.merge(professionals, pd.DataFrame(temp.rename('last_comment')), left_on='professionals_id', right_index=True, how='left')['last_comment']
# First activity (Comment)
temp = comments.groupby('comments_author_id')['comments_date_added'].min()
professionals['date_first_comment'] = pd.merge(professionals, pd.DataFrame(temp.rename('first_comment')), left_on='professionals_id', right_index=True, how='left')['first_comment']
# Last activity (Total)
professionals['date_last_activity'] = professionals[['date_last_answer', 'date_last_comment']].max(axis=1)
# First activity (Total)
professionals['date_first_activity'] = professionals[['date_first_answer', 'date_first_comment']].min(axis=1)
# Total Hearts score
temp = answers.groupby('answers_author_id')['answers_hearts'].sum()
professionals['professional_answers_hearts'] = pd.merge(professionals, pd.DataFrame(temp.rename('answers_hearts')), left_on='professionals_id', right_index=True, how='left')['answers_hearts'].fillna(0).astype(int)
# Professionals Tags to List
temp = pd.merge(professionals, tag_users, left_on='professionals_id', right_on='tag_users_user_id', how='inner')
temp = pd.merge(temp, tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')
temp = temp.groupby('professionals_id')['tags_tag_name'].apply(list).rename('professionals_tags')
professionals['professionals_tags'] = pd.merge(professionals, temp.to_frame(), left_on='professionals_id', right_index=True, how='left')['professionals_tags']

### Students
# Time since joining
students['students_time_delta_joined'] = actual_date - students['students_date_joined']
# Number of answers
temp = questions.groupby('questions_author_id').size()
students['students_questions_count'] = pd.merge(students, pd.DataFrame(temp.rename('count')), left_on='students_id', right_index=True, how='left')['count'].fillna(0).astype(int)
# Number of comments
temp = comments.groupby('comments_author_id').size()
students['students_comments_count'] = pd.merge(students, pd.DataFrame(temp.rename('count')), left_on='students_id', right_index=True, how='left')['count'].fillna(0).astype(int)
# Last activity (Question)
temp = questions.groupby('questions_author_id')['questions_date_added'].max()
students['date_last_question'] = pd.merge(students, pd.DataFrame(temp.rename('last_question')), left_on='students_id', right_index=True, how='left')['last_question']
# First activity (Question)
temp = questions.groupby('questions_author_id')['questions_date_added'].min()
students['date_first_question'] = pd.merge(students, pd.DataFrame(temp.rename('first_question')), left_on='students_id', right_index=True, how='left')['first_question']
# Last activity (Comment)
temp = comments.groupby('comments_author_id')['comments_date_added'].max()
students['date_last_comment'] = pd.merge(students, pd.DataFrame(temp.rename('last_comment')), left_on='students_id', right_index=True, how='left')['last_comment']
# First activity (Comment)
temp = comments.groupby('comments_author_id')['comments_date_added'].min()
students['date_first_comment'] = pd.merge(students, pd.DataFrame(temp.rename('first_comment')), left_on='students_id', right_index=True, how='left')['first_comment']
# Last activity (Total)
students['date_last_activity'] = students[['date_last_question', 'date_last_comment']].max(axis=1)
# First activity (Total)
students['date_first_activity'] = students[['date_first_question', 'date_first_comment']].min(axis=1)
# Total Hearts score
temp = questions.groupby('questions_author_id')['questions_hearts'].sum()
students['students_questions_hearts'] = pd.merge(students, pd.DataFrame(temp.rename('questions_hearts')), left_on='students_id', right_index=True, how='left')['questions_hearts'].fillna(0).astype(int)
# Students Tags to List
temp = pd.merge(students, tag_users, left_on='students_id', right_on='tag_users_user_id', how='inner')
temp = pd.merge(temp, tags, left_on='tag_users_tag_id', right_on='tags_tag_id', how='inner')
temp = temp.groupby('students_id')['tags_tag_name'].apply(list).rename('students_tags')
students['students_tags'] = pd.merge(students, temp.to_frame(), left_on='students_id', right_index=True, how='left')['students_tags']

### Emails Response
emails_response = pd.merge(emails, matches, left_on='emails_id', right_on='matches_email_id', how='inner')
emails_response = pd.merge(emails_response, questions, left_on='matches_question_id', right_on='questions_id', how='inner')
emails_response = pd.merge(emails_response, answers, left_on=['emails_recipient_id', 'matches_question_id'], right_on=['answers_author_id', 'answers_question_id'], how='left')
emails_response = emails_response.drop(['matches_email_id', 'matches_question_id', 'answers_id', 'answers_author_id', 'answers_body', 'answers_question_id'], axis=1)
emails_response = emails_response.drop(['questions_author_id', 'questions_title', 'questions_body', 'questions_full_text'], axis=1)
emails_response['time_delta_email_answer'] = (emails_response['answers_date_added'] - emails_response['emails_date_sent'])
emails_response['time_delta_question_email'] = (emails_response['emails_date_sent'] - emails_response['questions_date_added'])
plt_professionals = (professionals.groupby([professionals['professionals_date_joined'].dt.year]).size()/len(professionals.index))
plt_students = (students.groupby([students['students_date_joined'].dt.year]).size()/len(students.index))
plt_questions = (questions.groupby([questions['questions_date_added'].dt.year]).size()/len(questions.index))
plt_answers = (answers.groupby([answers['answers_date_added'].dt.year]).size()/len(answers.index))
plt_emails = (emails.groupby([emails['emails_date_sent'].dt.year]).size()/len(emails.index))
plt_comments = (comments.groupby([comments['comments_date_added'].dt.year]).size()/len(comments.index))
plt_data = pd.DataFrame({'students':plt_students,
                         'questions': plt_questions,
                         'professionals':plt_professionals,
                        'answers':plt_answers,
                        'comments':plt_comments,
                        'emails':plt_emails})
plt_data.plot(kind='bar', figsize=(15, 5))
plt.xlabel('Year')
plt.ylabel('Proportion')
plt.title('Distribution over time')
plt.show()
temp = professionals[['professionals_location', 'professionals_industry', 'professionals_headline']].fillna('Missing')
temp = temp.applymap(lambda x: x if x == 'Missing' else 'Available')
plt_professionals_location = temp.groupby('professionals_location').size()/len(temp.index)
plt_professionals_industry = temp.groupby('professionals_industry').size()/len(temp.index)
plt_professionals_headline = temp.groupby('professionals_headline').size()/len(temp.index)

plt_professionals_tags = tag_users['tag_users_user_id'].unique()
plt_professionals_tags = professionals['professionals_id'].apply(lambda x: 'Available' if x in plt_professionals_tags else 'Missing').rename('professionals_tags')
plt_professionals_tags = plt_professionals_tags.groupby(plt_professionals_tags).size()/len(plt_professionals_tags.index)
plt_professionals_group = group_memberships['group_memberships_user_id'].unique()
plt_professionals_group = professionals['professionals_id'].apply(lambda x: 'Available' if x in plt_professionals_group else 'Missing').rename('professionals_groups')
plt_professionals_group = plt_professionals_group.groupby(plt_professionals_group).size()/len(plt_professionals_group.index)
plt_professionals_school = school_memberships['school_memberships_user_id'].unique()
plt_professionals_school = professionals['professionals_id'].apply(lambda x: 'Available' if x in plt_professionals_school else 'Missing').rename('professionals_schools')
plt_professionals_school = plt_professionals_school.groupby(plt_professionals_school).size()/len(plt_professionals_school.index)

temp = professionals[['professionals_answers_count', 'professionals_comments_count']]
temp = temp.applymap(lambda x: 'Available' if x > 0 else 'Missing')
plt_professionals_answers = temp.groupby('professionals_answers_count').size()/len(temp.index)
plt_professionals_comments = temp.groupby('professionals_comments_count').size()/len(temp.index)

plt_data = pd.DataFrame({'Location': plt_professionals_location,
                        'Industry': plt_professionals_industry,
                        'Headline': plt_professionals_headline,
                        'Tags': plt_professionals_tags,
                        'Groups': plt_professionals_group,
                        'Schools': plt_professionals_school,
                        'Answers': plt_professionals_answers,
                        'Comments': plt_professionals_comments,})

plt_data.T.plot(kind='bar', stacked=True, figsize=(15, 5))
plt.ylabel('Proportion')
plt.title('Missing values for professionals')
plt.yticks(np.arange(0, 1.05, 0.1))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt_data.T
temp = students[['students_location']].fillna('Missing')
temp = temp.applymap(lambda x: x if x == 'Missing' else 'Available')
plt_students_location = temp.groupby('students_location').size()/len(temp.index)

plt_students_tags = tag_users['tag_users_user_id'].unique()
plt_students_tags = students['students_id'].apply(lambda x: 'Available' if x in plt_students_tags else 'Missing').rename('students_tags')
plt_students_tags = plt_students_tags.groupby(plt_students_tags).size()/len(plt_students_tags.index)
plt_students_group = group_memberships['group_memberships_user_id'].unique()
plt_students_group = students['students_id'].apply(lambda x: 'Available' if x in plt_students_group else 'Missing').rename('students_groups')
plt_students_group = plt_students_group.groupby(plt_students_group).size()/len(plt_students_group.index)
plt_students_school = school_memberships['school_memberships_user_id'].unique()
plt_students_school = students['students_id'].apply(lambda x: 'Available' if x in plt_students_school else 'Missing').rename('students_schools')
plt_students_school = plt_students_school.groupby(plt_students_school).size()/len(plt_students_school.index)

temp = students[['students_questions_count', 'students_comments_count']]
temp = temp.applymap(lambda x: 'Available' if x > 0 else 'Missing')
plt_students_questions = temp.groupby('students_questions_count').size()/len(temp.index)
plt_students_comments = temp.groupby('students_comments_count').size()/len(temp.index)

plt_data = pd.DataFrame({'Location': plt_students_location,
                        'Tags': plt_students_tags,
                        'Groups': plt_students_group,
                        'Schools': plt_students_school,
                        'Answers': plt_students_questions,
                        'Comments': plt_students_comments,})

plt_data.T.plot(kind='bar', stacked=True, figsize=(15, 5))
plt.ylabel('Proportion')
plt.title('Missing values for students')
plt.yticks(np.arange(0, 1.05, 0.1))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
plt_data.T
students_tags = tag_users[tag_users['tag_users_user_id'].isin(students['students_id'])]
students_tags = pd.merge(students_tags, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')
students_tags['user_type'] = 'student'

professionals_tags = tag_users[tag_users['tag_users_user_id'].isin(professionals['professionals_id'])]
professionals_tags = pd.merge(professionals_tags, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')
professionals_tags['user_type'] = 'professional'

questions_tags = tag_questions
questions_tags = pd.merge(questions_tags, tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
questions_tags['user_type'] = 'question'

plt_data = pd.concat([students_tags, professionals_tags, questions_tags])
plt_data = plt_data[['tags_tag_name', 'user_type']].pivot_table(index='tags_tag_name', columns='user_type', aggfunc=len, fill_value=0)
plt_data['professional'] = plt_data['professional'] / professionals.shape[0]
plt_data['student'] = plt_data['student'] / students.shape[0]
plt_data['question'] = plt_data['question'] / questions.shape[0]
plt_data['sum'] = (plt_data['professional'] + plt_data['student'] + plt_data['question'])
plt_data = plt_data.sort_values(by='sum', ascending=False).drop(['sum'], axis=1).head(100)

# Bubble chart
fig, ax = plt.subplots(facecolor='w',figsize=(15, 15))
ax.set_xlabel('Professionals')
ax.set_ylabel('Questions')
ax.set_title('Tags Matching')
ax.set_xlim([0, max(plt_data['professional'])+0.001])
ax.set_ylim([0, max(plt_data['question'])+0.005])
import matplotlib.ticker as mtick
ax.xaxis.set_major_formatter(mtick.FuncFormatter("{:.2%}".format))
ax.yaxis.set_major_formatter(mtick.FuncFormatter("{:.2%}".format))
ax.grid(True)
i = 0
for key, row in plt_data.iterrows():
    ax.scatter(row['professional'], row['question'], s=10+row['student']*10**5, alpha=.5)
    if i < 25:
        ax.annotate('{}: {:.2%}'.format(key, row['student']), xy=(row['professional'], row['question']))
    i += 1
plt.show()

# Wordcloud
plt.figure(figsize=(20, 20))
wordloud_values = ['student', 'professional', 'question']
axisNum = 1
for wordcloud_value in wordloud_values:
    wordcloud = WordCloud(margin=0, max_words=20, random_state=seed).generate_from_frequencies(plt_data[wordcloud_value])
    ax = plt.subplot(1, 3, axisNum)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.title(wordcloud_value)
    plt.axis("off")
    axisNum += 1
plt.show()    
plt_professionals = professionals
plt_professionals = plt_professionals[(plt_professionals['professionals_date_joined'] >= '01-01-2016') & (plt_professionals['professionals_date_joined'] <= '30-06-2018')]
plt_professionals = (plt_professionals['date_first_activity'] - plt_professionals['professionals_date_joined']).dt.days.fillna(9999).astype(int)
plt_professionals = plt_professionals.groupby(plt_professionals).size()/len(plt_professionals.index)
plt_professionals = plt_professionals.rename(lambda x: 0 if x < 0.0 else x)
plt_professionals = plt_professionals.rename(lambda x: x if x <= 7.0 or x == 9999 else '> 7')
plt_professionals = plt_professionals.rename({9999:'NaN'})
plt_professionals = plt_professionals.groupby(level=0).sum()

plt_students = students
plt_students = plt_students[(plt_students['students_date_joined'] >= '01-01-2016') & (plt_students['students_date_joined'] <= '30-06-2018')]
plt_students = (plt_students['date_first_activity'] - plt_students['students_date_joined']).dt.days.fillna(9999).astype(int)
plt_students = plt_students.groupby(plt_students).size()/len(plt_students.index)
plt_students = plt_students.rename(lambda x: 0 if x < 0.0 else x)
plt_students = plt_students.rename(lambda x: x if x <= 7.0 or x == 9999 else '> 7')
plt_students = plt_students.rename({9999:'NaN'})
plt_students = plt_students.groupby(level=0).sum()

plt_data = pd.DataFrame({'Professionals': plt_professionals,
                        'Students': plt_students})

plt_data.plot(kind='bar', figsize=(15, 5))
plt.xlabel('Days')
plt.ylabel('Proportion')
plt.title('Days for first activity after registration')
plt.show()
plt_professionals = ((actual_date - professionals['date_last_activity']).dt.days/30).dropna().apply(lambda x: math.ceil(x)).astype(int)
plt_professionals = plt_professionals.groupby(plt_professionals).size()/len(plt_professionals.index)
plt_professionals = plt_professionals.rename(lambda x: 0 if x < 0.0 else x)
plt_professionals = plt_professionals.rename(lambda x: x if x <= 36.0 or x == 9999 else '> 36')
plt_professionals = plt_professionals.rename({9999:'NaN'})
plt_professionals = plt_professionals.groupby(level=0).sum()

plt_students = ((actual_date - students['date_last_activity']).dt.days/30).dropna().apply(lambda x: math.ceil(x)).astype(int)
plt_students = plt_students.groupby(plt_students).size()/len(plt_students.index)
plt_students = plt_students.rename(lambda x: 0 if x < 0.0 else x)
plt_students = plt_students.rename(lambda x: x if x <= 36.0 or x == 9999 else '> 36')
plt_students = plt_students.rename({9999:'NaN'})
plt_students = plt_students.groupby(level=0).sum()

plt_data = pd.DataFrame({'Professionals': plt_professionals,
                        'Students': plt_students})

plt_data.plot(kind='bar', figsize=(15, 5))
plt.xlabel('Months')
plt.ylabel('Proportion')
plt.title('Months for last activity')
plt.show()
plt_professionals = ((actual_date - professionals['date_last_activity']).dt.days/30).dropna().apply(lambda x: math.ceil(x)).astype(int)
plt_students = ((actual_date - students['date_last_activity']).dt.days/30).dropna().apply(lambda x: math.ceil(x)).astype(int)
plt_data = pd.DataFrame({'Professionals': plt_professionals, 'Students': plt_students})
plt_total = pd.concat([plt_data['Professionals'], plt_data['Students']]).rename('All Users')
plt_data.plot(kind='hist', bins=1000, density=True, histtype='step', cumulative=True, figsize=(15, 7), lw=2, grid=True)
plt_total.plot(kind='hist', bins=1000, density=True, histtype='step', cumulative=True, figsize=(15, 7), lw=2, grid=True)
plt.xlabel('Months')
plt.ylabel('Cumulative')
plt.title('Cumulative histogram for last activity')
plt.legend(loc='upper left')
plt.xlim([0, 50])
plt.xticks(range(0, 51, 1))
plt.yticks(np.arange(0, 1.05, 0.05))
plt.show()
plt_questions = questions
plt_questions = plt_questions[(plt_questions['questions_date_added'] >= '01-01-2016') & (plt_questions['questions_date_added'] <= '30-06-2018')]
plt_questions = ((plt_questions['questions_first_answers'] - plt_questions['questions_date_added']).dt.days/7).fillna(9999).apply(lambda x: math.ceil(x)).astype(int)
plt_questions = plt_questions.groupby(plt_questions).size()/len(plt_questions.index)
plt_questions = plt_questions.rename(lambda x: 0 if x < 0.0 else x)
plt_questions = plt_questions.rename(lambda x: x if x <= 30.0 or x == 9999 else '> 30')
plt_questions = plt_questions.rename({9999:'NaN'})
plt_questions = plt_questions.groupby(level=0).sum()

plt_data = pd.DataFrame({'Questions': plt_questions})
plt_data.plot(kind='bar', figsize=(15, 5))
plt.xlabel('Weeks')
plt.ylabel('Frequency')
plt.title('Weeks for first answer')
plt.show()
plt_data = questions[['questions_date_added', 'questions_first_answers']]
plt_data = plt_data[plt_data['questions_date_added'] > '2015-01-01']
plt_data['answer_time'] = (plt_data['questions_first_answers'] - plt_data['questions_date_added'])
plt_data['days'] = (plt_data['questions_first_answers'] - plt_data['questions_date_added']).dt.days
plt_data['date'] = plt_data['questions_date_added'].dt.strftime('%B %Y')
plt_data['date'] = plt_data['date'].apply(lambda x: str(datetime.strptime(x, "%B %Y")))  
plt_data['date2'] = plt_data['questions_date_added'].dt.strftime('%B %Y')

plt_data_1 = (plt_data[plt_data['days'] <= 0].groupby(['date', 'date2']).size() / plt_data.groupby(['date', 'date2']).size()).rename('< 1 day')
plt_data_2 = (plt_data[plt_data['days'] <= 3].groupby(['date', 'date2']).size() / plt_data.groupby(['date', 'date2']).size()).rename('<= 3 days')
plt_data_3 = (plt_data[plt_data['days'] <= 7].groupby(['date', 'date2']).size() / plt_data.groupby(['date', 'date2']).size()).rename('<= 7 days')
plt_data = pd.concat([plt_data_1, plt_data_2, plt_data_3], axis=1).fillna(0)
plt_data.index = plt_data.index.droplevel('date')

plt_data.plot(kind='bar', figsize=(15, 5), width=0.9)
plt.xlabel('Date')
plt.ylabel('Frequency')
plt.title('Response time in days')
plt.show()
plt_email_response = emails_response[(emails_response['emails_date_sent'] >= '01-01-2016')].dropna()

plt_data = pd.DataFrame()
title_mapping = {'time_delta_email_answer':'Answers', 'time_delta_question_email':'Questions'}
for qa in ['time_delta_email_answer', 'time_delta_question_email']:
    plt_data = pd.DataFrame()
    for fl in ['email_notification_daily', 'email_notification_weekly', 'email_notification_immediate']:
        temp = plt_email_response[plt_email_response['emails_frequency_level'] == fl]
        temp = temp[qa].dt.days.astype(int)
        temp = temp.groupby(temp).size()/len(temp.index)
        temp = temp.rename(lambda x: 0 if x < 0.0 else x)
        temp = temp.rename(lambda x: x if x <= 14.0 else '> 14')
        temp = temp.groupby(level=0).sum() 
        plt_data = pd.concat([plt_data, temp], axis=1, sort=False)
    plt_data.columns = ['Daily', 'Weekly', 'Immediate']

    plt_data.plot(kind='bar', figsize=(15, 5))
    plt.xlabel('Days')
    plt.ylabel('Frequency')
    plt.title('Email response time ({})'.format(title_mapping[qa]))
    plt.legend(loc='upper center')
    plt.show()
plt_data_questions = questions['questions_full_text'].apply(lambda x: len(x.split())).rename("Questions")
plt_data_answers = answers['answers_body'].astype(str).apply(lambda x: len(x.split())).rename("Answers")
plt_data = pd.DataFrame([plt_data_questions, plt_data_answers]).T

plt_data.plot(kind='box', showfliers=False, vert=False, figsize=(15, 5), grid=True)
plt.xticks(range(0, 400, 25))
plt.xlabel('Words')
plt.title('Word count')
plt.show()
def plot_user_activity(user_id, xticks_type='Month', xticks_interval=3, plot_emails=False):
    """ Merges all relevant data for a user together and builds a timeline chart.
        
        :param user_id: Index of the 'students' dataframe (default: seed)
        :param xticks_type: Grouping x axis by 'Month', 'Day' or 'Year' (default: 'Month')
        :param xticks_interval: Integer to plot every n ticks (default: 3)
        :param plot_emails: Also consider emails (default: False)
    """ 
    # student registration
    graph_student = students[students['students_id'] == user_id]
    graph_student = pd.DataFrame({'type':'Registration', 
                               'id':graph_student['students_id'], 
                               'date':graph_student['students_date_joined'],
                              'color':'green'})
    # professional registration
    graph_professional = professionals[professionals['professionals_id'] == user_id]
    graph_professional = pd.DataFrame({'type':'Registration', 
                               'id':graph_professional['professionals_id'], 
                               'date':graph_professional['professionals_date_joined'],
                              'color':'green'})
    # questions
    graph_questions = questions[questions['questions_author_id'] == user_id]
    graph_questions = pd.DataFrame({'type':'Questions',
                                   'id':graph_questions['questions_id'], 
                                   'date':graph_questions['questions_date_added'],
                                  'color':'blue'})
    # answers
    graph_answers = answers[answers['answers_author_id'] == user_id]
    graph_answers = pd.DataFrame({'type':'Answers',
                                   'id':graph_answers['answers_id'], 
                                   'date':graph_answers['answers_date_added'],
                                  'color':'red'})
    # commments
    graph_comments = comments[comments['comments_author_id'] == user_id]
    graph_comments = pd.DataFrame({'type':'Comments',
                                   'id':graph_comments['comments_id'], 
                                   'date':graph_comments['comments_date_added'],
                                  'color':'orange'})
    if plot_emails:
        graph_emails = emails[emails['emails_recipient_id'] == user_id]
        graph_emails = pd.DataFrame({'type':'Emails', 
                                   'id':graph_emails['emails_id'], 
                                   'date':graph_emails['emails_date_sent'],
                                  'color':'grey'})
    else:
        graph_emails = None

    graph_data = pd.concat([graph_student, graph_professional, graph_questions, graph_comments, graph_answers, graph_emails])
    # Group data by date
    if xticks_type=='Day':
        graph_data['date'] = graph_data['date'].dt.strftime('%Y-%m-%d')
        graph_data['date'] = graph_data['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    else:
        graph_data['date'] = graph_data['date'].dt.strftime('%B %Y')
        graph_data['date'] = graph_data['date'].apply(lambda x: datetime.strptime(x, "%B %Y"))     
    
    graph_data = graph_data.groupby(['type', 'date', 'color']).size().rename('count').reset_index().sort_values('date')
    graph_data['name'] = graph_data['count'].map(str)+ ' '+graph_data['type']
    names = graph_data['name'].tolist()
    colors = graph_data['color'].tolist()
    dates = graph_data['date'].tolist()

    # Plot
    levels = np.array([-9, 9, -7, 7, -5, 5, -3, 3, -1, 1])
    fig, ax = plt.subplots(figsize=(15, 8))

    # Create the base line
    start = min(dates)
    stop = max(dates)
    ax.plot((start, stop), (0, 0), 'k', alpha=.5)

    # Create annotations
    for ii, (iname, idate, icol) in enumerate(zip(names, dates, colors)):
        level = levels[ii % len(levels)]
        vert = 'top' if level < 0 else 'bottom'
        ax.scatter(idate, 0, s=100, facecolor=icol, edgecolor='k', zorder=9999)
        ax.plot((idate, idate), (0, level), c=icol, alpha=1.0, lw=2)
        ax.text(idate, level, iname, horizontalalignment='center', verticalalignment=vert, fontsize=12, backgroundcolor=icol)
    ax.set(title="Timeline for user: {}".format(user_id))
    # Set the xticks formatting
    if xticks_type=='Month':
        ax.get_xaxis().set_major_locator(mdates.MonthLocator(interval=xticks_interval))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%B %Y"))
    elif xticks_type=='Day':
        ax.get_xaxis().set_major_locator(mdates.DayLocator(interval=xticks_interval))
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    elif xticks_type=='Year':
        ax.get_xaxis().set_major_locator(mdates.YearLocator())
        ax.get_xaxis().set_major_formatter(mdates.DateFormatter("%Y"))        
    fig.autofmt_xdate()
    #Legend
    legend = []
    for index, row in graph_data[['type', 'color']].drop_duplicates().iterrows():
        legend += [mpatches.Patch(color=row['color'], label=row['type'])]
    plt.legend(handles=legend, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Remove components for a cleaner look
    plt.setp((ax.get_yticklabels() + ax.get_yticklines() + list(ax.spines.values())), visible=False)
    plt.show()
user_id = '977428d851b24183b223be0eb8619a8c'
plot_user_activity(user_id=user_id, xticks_type='Month', xticks_interval=1)
user_id = 'e1d39b665987455fbcfbec3fc6df6056'
plot_user_activity(user_id=user_id, xticks_type='Month', xticks_interval=3, plot_emails=True)
user_id = '16908136951a48ed942738822cedd5c2'
plot_user_activity(user_id=user_id, xticks_type='Month', xticks_interval=1)
user_id = 'e5c389a88c884e13ac828dd22628acc8'
plot_user_activity(user_id=user_id, xticks_type='Day', xticks_interval=7)
def plot_dependecy_graph(emails_id=[], details=['tag', 'group', 'school'], details_min_edges=1):
    """ Merges all relevant data for a given email together and builds a dependency graph and report.
        
        :param emails_id: 'email_id' of the 'emails' dataframe
        :param details: List which details should be ploted (default: ['tags', 'groups', 'schools'])
        :param details_min_edges: How many edges a node from the details list must have (default: 1 (show all))
    """  
    graph_edges = pd.DataFrame()
    graph_nodes = pd.DataFrame()
    #email
    graph_emails = emails[emails['emails_id'].isin(emails_id)]
    graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_emails['emails_id'], 'type':'email', 'color':'grey', 'size':1})])
    #questions
    graph_matches = matches[matches['matches_email_id'].isin(graph_emails['emails_id'])]
    graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_matches['matches_question_id'], 'type':'question', 'color':'blue', 'size':1})])
    graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_matches['matches_email_id'], 'target':graph_matches['matches_question_id']})])
    #answers
    graph_answers = answers[answers['answers_question_id'].isin(graph_matches['matches_question_id'])]
    graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_answers['answers_id'], 'type':'answer', 'color':'red', 'size':1})])
    graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_answers['answers_question_id'], 'target':graph_answers['answers_id']})])
    #professionals
    graph_professionals = answers[answers['answers_question_id'].isin(graph_matches['matches_question_id'])]
    graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_professionals['answers_author_id'], 'type':'professional', 'color':'cyan', 'size':1})])
    graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_professionals['answers_id'], 'target':graph_professionals['answers_author_id']})])
    #students
    graph_students = questions[questions['questions_id'].isin(graph_matches['matches_question_id'])]
    graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_students['questions_author_id'], 'type':'student', 'color':'green', 'size':1})])
    graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_students['questions_id'], 'target':graph_students['questions_author_id']})])
    if 'tag' in details:
        #question tags
        graph_questions_tags = tag_questions[tag_questions['tag_questions_question_id'].isin(graph_matches['matches_question_id'])]
        graph_questions_tags = pd.merge(graph_questions_tags, tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_questions_tags['tags_tag_name'], 'type':'tag', 'color':'yellow', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_questions_tags['tag_questions_question_id'], 'target':graph_questions_tags['tags_tag_name']})])  
        #professional tags
        graph_professionals_tags = tag_users[tag_users['tag_users_user_id'].isin(graph_professionals['answers_author_id'])]
        graph_professionals_tags = pd.merge(graph_professionals_tags, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_professionals_tags['tags_tag_name'], 'type':'tag', 'color':'yellow', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_professionals_tags['tag_users_user_id'], 'target':graph_professionals_tags['tags_tag_name']})])     
        #students tags
        graph_students_tags = tag_users[tag_users['tag_users_user_id'].isin(graph_students['questions_author_id'])]
        graph_students_tags = pd.merge(graph_students_tags, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_students_tags['tags_tag_name'], 'type':'tag', 'color':'yellow', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_students_tags['tag_users_user_id'], 'target':graph_students_tags['tags_tag_name']})]) 
    if 'group' in details:
        #professional group
        graph_professionals_group = group_memberships[group_memberships['group_memberships_user_id'].isin(graph_professionals['answers_author_id'])]
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_professionals_group['group_memberships_group_id'], 'type':'group', 'color':'orange', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_professionals_group['group_memberships_user_id'], 'target':graph_professionals_group['group_memberships_group_id']})]) 
        #students group
        graph_students_group = group_memberships[group_memberships['group_memberships_user_id'].isin(graph_students['questions_author_id'])]
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_students_group['group_memberships_group_id'], 'type':'group', 'color':'orange', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_students_group['group_memberships_user_id'], 'target':graph_students_group['group_memberships_group_id']})]) 
    if 'school' in details:
        #professional school
        graph_professionals_school = school_memberships[school_memberships['school_memberships_user_id'].isin(graph_professionals['answers_author_id'])]
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_professionals_school['school_memberships_school_id'], 'type':'school', 'color':'purple', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_professionals_school['school_memberships_user_id'], 'target':graph_professionals_school['school_memberships_school_id']})])
        #students school
        graph_students_school = school_memberships[school_memberships['school_memberships_user_id'].isin(graph_students['questions_author_id'])]
        graph_nodes = pd.concat([graph_nodes, pd.DataFrame({'node':graph_students_school['school_memberships_school_id'], 'type':'school', 'color':'purple', 'size':1})])
        graph_edges = pd.concat([graph_edges, pd.DataFrame({'source':graph_students_school['school_memberships_user_id'], 'target':graph_students_school['school_memberships_school_id']})])
    
    # check min count of edges for details
    graph_nodes = graph_nodes.drop_duplicates()
    temp = pd.concat([graph_edges[['source', 'target']], graph_edges[['target', 'source']].rename(columns={'target':'source', 'source':'target'})])
    temp = temp[temp['source'].isin(graph_nodes[graph_nodes['type'].isin(details)]['node'])]
    temp = temp.drop_duplicates().groupby('source').size()
    graph_nodes = graph_nodes[~graph_nodes['node'].isin(temp[temp<details_min_edges].index.values)]
    graph_edges = graph_edges[(~graph_edges['source'].isin(temp[temp<details_min_edges].index.values)) & (~graph_edges['target'].isin(temp[temp<details_min_edges].index.values))]
    graph_nodes_color = graph_nodes['color']
    
    plt.figure(figsize=(15, 15)) 
    G = nx.Graph()
    G.add_nodes_from(graph_nodes['node'])
    G.add_edges_from({tuple(row) for i,row in graph_edges[['source', 'target']].iterrows()})
    nx.draw_networkx(G, with_labels=True, node_color=graph_nodes_color, font_size=8, node_size=900/len(emails_id))
    plt.title('Dependency graph for email {}'.format(emails_id))
    plt.axis('off')

    legend = []
    for index, row in graph_nodes[['type', 'color']].drop_duplicates().iterrows():
        legend += [mpatches.Patch(color=row['color'], label=row['type'])]
    plt.legend(handles=legend)
    plt.show()
emails_id = emails.loc[seed, 'emails_id']
plot_dependecy_graph(emails_id=[emails_id])
emails_id = emails.loc[seed, 'emails_id']
plot_dependecy_graph(emails_id=[emails_id], details_min_edges=2)
emails_id = emails.loc[seed*2, 'emails_id']
plot_dependecy_graph(emails_id=[emails_id])
# Gensim Dictionary Filter
extremes_no_below = 20
extremes_no_above = 0.6
extremes_keep_n = 8000

# LDA
num_topics = 21
passes = 15
chunksize = 1000
alpha = 1/50
def get_model_results(ldamodel, corpus, dictionary):
    """ Create doc-topic probabilities table and visualization for the LDA model

    """  
    vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, sort_topics=False)
    transformed = ldamodel.get_document_topics(corpus)
    df = pd.DataFrame.from_records([{v:k for v, k in row} for row in transformed])
    return vis, df  
def get_model_wordcloud(ldamodel):
    """ Create a Word Cloud for each topic of the LDA model

    """  
    plot_cols = 3
    plot_rows = math.ceil(num_topics / 3)
    axisNum = 0
    plt.figure(figsize=(5*plot_cols, 3*plot_rows))
    for topicID in range(ldamodel.state.get_lambda().shape[0]):
        #gather most relevant terms for the given topic
        topics_terms = ldamodel.state.get_lambda()
        tmpDict = {}
        for i in range(1, len(topics_terms[0])):
            tmpDict[ldamodel.id2word[i]]=topics_terms[topicID,i]

        # draw the wordcloud
        wordcloud = WordCloud( margin=0,max_words=20 ).generate_from_frequencies(tmpDict)
        axisNum += 1
        ax = plt.subplot(plot_rows, plot_cols, axisNum)

        plt.imshow(wordcloud, interpolation='bilinear')
        title = topicID
        plt.title(title)
        plt.axis("off")
        plt.margins(x=0, y=0)
    plt.show()
def topic_query(data, query):
    """ Get Documents matching the query with the doc-topic probabilities

    """  
    result = data
    result['sort'] = 0
    for topic in query:
        result = result[result[topic] >= query[topic]]
        result['sort'] += result[topic]
    result = result.sort_values(['sort'], ascending=False)
    result = result.drop('sort', axis=1)
    result = result.head(5)
    return result
def get_text_topics(text, top=20):
    """ Get the topics probabilities for a text and highlight relevant words

    """    
    def token_topic(token):
        return topic_words.get(token, -1)
    
    colors = ['\033[46m', '\033[45m', '\033[44m', '\033[43m', '\033[42m', '\033[41m', '\033[47m']    
    nlp_tokens = nlp_preprocessing([text])

    bow_text = [lda_dic.doc2bow(doc) for doc in nlp_tokens]
    bow_text = lda_tfidf[bow_text]
    topic_text = lda_model.get_document_topics(bow_text)
    topic_text = pd.DataFrame.from_records([{v:k for v, k in row} for row in topic_text])
    
    print('Question:')
    topic_words = []
    topic_labeled = 0
    for topic in topic_text.columns.values:
        topic_terms = lda_model.get_topic_terms(topic, top)
        topic_words = topic_words+[[topic_labeled, lda_dic[pair[0]], pair[1]] for pair in topic_terms]
        topic_labeled += 1
    topic_words = pd.DataFrame(topic_words, columns=['topic', 'word', 'value']).pivot(index='word', columns='topic', values='value').idxmax(axis=1)
    nlp_doc = nlp(text)
    text_highlight = ''.join([x.string if token_topic(x.lemma_.lower()) <0  else colors[token_topic(x.lemma_.lower()) % len(colors)] + x.string + '\033[0m' for x in nlp_doc])
    print(text_highlight) 
    
    print('\nTopics:')
    topic_labeled = 0
    for topic in topic_text:
        print(colors[topic_labeled % len(colors)]+'Topic '+str(topic)+':', '{0:.2%}'.format(topic_text[topic].values[0])+'\033[0m')
        topic_labeled += 1

    # Plot Pie chart
    plt_data = topic_text
    plt_data.columns = ['Topic '+str(c) for c in plt_data.columns]
    plt_data['Others'] = 1-plt_data.sum(axis=1)
    plt_data = plt_data.T
    plt_data.plot(kind='pie', y=0, autopct='%.2f')
    plt.xlabel('')
    plt.ylabel('')
    plt.title('Topics Probabilities')
    plt.show()
### %time
lda_tokens = questions['nlp_tokens']
# Gensim Dictionary
lda_dic = gensim.corpora.Dictionary(lda_tokens)
lda_dic.filter_extremes(no_below=extremes_no_below, no_above=extremes_no_above, keep_n=extremes_keep_n)
lda_corpus = [lda_dic.doc2bow(doc) for doc in lda_tokens]

lda_tfidf = gensim.models.TfidfModel(lda_corpus)
lda_corpus = lda_tfidf[lda_corpus]

# Create LDA Model
lda_model = gensim.models.ldamodel.LdaModel(lda_corpus, num_topics=num_topics, 
                                            id2word = lda_dic, passes=passes,
                                            chunksize=chunksize,update_every=0,
                                            alpha=alpha, random_state=seed)

# Create Visualization and Doc-Topic Probabilities
lda_vis, lda_result = get_model_results(lda_model, lda_corpus, lda_dic)
lda_questions = questions[['questions_id', 'questions_title', 'questions_body']]
lda_questions = pd.concat([lda_questions, lda_result.add_prefix('Topic_')], axis=1)
get_model_wordcloud(lda_model)
#lda_vis
lda_questions.head(5).dropna(axis=1, how='all').T
query_text = 'I will finish my college next year and would like to start a career as a Data Scientist. \n\n'\
            +'What is the best way to become a good Data Scientist? #data-science'
get_text_topics(query_text, 100)
query = {'Topic_3':0.4, 'Topic_18':0.4}
topic_query(lda_questions, query).dropna(axis=1, how='all').head(2).T
get_text_topics(questions['questions_full_text'][7582], 50)
print()
get_text_topics(questions['questions_full_text'][1597], 50)
def get_similar_text_questions(questions_text, questions=questions, threshold=0.01, top=5, answered=True):
    """ Calculates the similarity to the existing questions and returns the most similar ones.
        
        :param questions_text: List with text of new questions to get the similiarities
        :param threshold: Threshold to filter out all questions with similarity below the value (default: 0.01)
        :param top: Top N similar questions (default: 5)
        :param answered: Consider only answered questions (default: True)
        :return result: DataFrame with the top similar questions
    """ 
    nlp_corpus = [' '.join(x) for x in questions['nlp_tokens']]
    nlp_text = [' '.join(x) for x in nlp_preprocessing(questions_text)]
    vectorizer = TfidfVectorizer()
    vectorizer.fit(nlp_corpus)
    corpus_tfidf = vectorizer.transform(nlp_corpus)
    
    text_tfidf = vectorizer.transform(nlp_text)
    sim = cosine_similarity(corpus_tfidf, text_tfidf)
    result = pd.DataFrame({'questions_id':np.tile(questions['questions_id'], sim.shape[1]),
                           'similarity':sim.reshape(-1,),
                          'title':np.tile(questions['questions_title'], sim.shape[1]),
                          'body':np.tile(questions['questions_body'], sim.shape[1]),
                           'questions_tags':np.tile(questions['questions_tags'], sim.shape[1]),
                          'questions_answers_count':np.tile(questions['questions_answers_count'], sim.shape[1]),
                          'questions_hearts':np.tile(questions['questions_hearts'], sim.shape[1]),},
                         index=np.tile(questions.index, sim.shape[1]))
    if answered:
        result = result[result['questions_answers_count'] > 0]
    result = result[result['similarity'] >= threshold].sort_values('similarity', ascending=False).head(top)
    return result

def get_similar_tags_questions(tags, questions=questions, threshold=0.01, top=5, answered=True):
    """ Calculates the similarity to the existing questions and returns the most similar ones.
        
        :param tags: List with tags to get the similiarities
        :param threshold: Threshold to filter out all questions with similarity below the value (default: 0.01)
        :param top: Top N similar questions (default: 5)
        :param answered: Consider only answered questions (default: True)
        :return result: DataFrame with the top similar questions
    """ 
    sim = questions['questions_tags'].fillna('').apply(lambda x: len(set(x) & set(tags))/len(set(x) | set(tags)) if len(set(x) | set(tags)) > 0 else 0)
    result = pd.DataFrame({'questions_id':questions['questions_id'],
                           'similarity':sim,
                          'title':questions['questions_title'],
                          'body':questions['questions_body'],
                           'questions_tags':questions['questions_tags'],
                          'questions_answers_count':questions['questions_answers_count'],
                          'questions_hearts':questions['questions_hearts'],},
                         index=questions.index)
    if answered:
        result = result[result['questions_answers_count'] > 0]
    result = result[result['similarity'] >= threshold].sort_values('similarity', ascending=False).head(top)
    return result

def get_similar_topic_questions(questions_text, questions=questions, threshold=0.01, top=5, answered=True):
    """ Calculates the topic similarity to the existing questions and returns the most similar ones.
        
        :param questions_text: List with text of new questions to get the similiarities
        :param threshold: Threshold to filter out all questions with similarity below the value (default: 0.01)
        :param top: Top N similar questions (default: 5)
        :param answered: Consider only answered questions (default: True)
        :return result: DataFrame with the top similar questions
    """ 
    nlp_corpus = questions['nlp_tokens']
    bow_corpus = [lda_dic.doc2bow(doc) for doc in nlp_corpus]
    bow_corpus = lda_tfidf[bow_corpus]
    topic_corpus = lda_model.get_document_topics(bow_corpus, minimum_probability=0)
    topic_corpus = pd.DataFrame.from_records([{v:k for v, k in row} for row in topic_corpus])
    
    nlp_text = nlp_preprocessing(questions_text)
    bow_text = [lda_dic.doc2bow(doc) for doc in nlp_text]
    bow_text = lda_tfidf[bow_text]
    topic_text = lda_model.get_document_topics(bow_text, minimum_probability=0)
    topic_text = pd.DataFrame.from_records([{v:k for v, k in row} for row in topic_text])
    
    sim = cosine_similarity(topic_corpus.fillna(0), topic_text.fillna(0))
    result = pd.DataFrame({'questions_id':np.tile(questions['questions_id'], sim.shape[1]),
                           'similarity':sim.reshape(-1,),
                          'title':np.tile(questions['questions_title'], sim.shape[1]),
                          'body':np.tile(questions['questions_body'], sim.shape[1]),
                           'questions_tags':np.tile(questions['questions_tags'], sim.shape[1]),
                          'questions_answers_count':np.tile(questions['questions_answers_count'], sim.shape[1]),
                          'questions_hearts':np.tile(questions['questions_hearts'], sim.shape[1]),},
                         index=np.tile(questions.index, sim.shape[1]))
    if answered:
        result = result[result['questions_answers_count'] > 0]
    result = result[result['similarity'] >= threshold].sort_values('similarity', ascending=False).head(top)
    return result
query_text = ['I will finish my college next year and would like to start a career as a Data Scientist. \n\n'\
            +'What is the best way to become a good Data Scientist? #data-science']
questions_corpus = questions[questions['questions_date_added'] > '2017-01-01']
get_similar_text_questions(query_text, questions=questions_corpus).T
def recommend_new_question(question_text, questions=questions, top_questions=10, top_professionals=5, weights=[1, 1, 1, 1, 1, 1]):
    """ Determines which professionals are best qualified to answer the question.
        
        :param questions_text: List with text of new questions to get the similiarities
        :param top_questions: Top N similar questions (default: 10)
        :param top_professionals: Top N recommended professionals (default: 5)
        :param weights: Weights for the various scores (default: uniform distribution)
        :return recommendation: DataFrame with the top scored professionals
        :return result: DataFrame with all rows and columns to be able to understand the calculation.
    """ 
    relevant_columns = ['questions_id', 'similarity', 'questions_answers_count', 
                        'answers_author_id', 'time_delta_answer', 'answers_time_rank', 'answers_hearts', 
                        'time_delta_last_activity', 'professional_answers_hearts', 'questions_tags', 'professionals_tags']
    sim_questions = get_similar_text_questions(question_text, questions=questions, threshold=0.0, top=top_questions, answered=True)
    result = pd.merge(sim_questions, answers, left_on='questions_id', right_on='answers_question_id', how='left')
    result = pd.merge(result, professionals, left_on='answers_author_id', right_on='professionals_id', how='left')
    
    result['time_delta_answer'] = result['time_delta_answer'].dt.days
    result['time_delta_last_activity'] = (actual_date - result['date_last_activity']).dt.days
    result = result[relevant_columns]
    result['score_activity'] = 1/np.log10(10+result['time_delta_last_activity'])
    result['score_answer_time'] = 1/np.log10(10+result['time_delta_answer'])
    result['score_answer_rank'] = 1/np.log10(9+result['answers_time_rank'])
    result['score_answers_hearts'] = np.log10(10+result['answers_hearts'])
    result['score_professional_hearts'] = np.log10(10+result['professional_answers_hearts'])
    result['final_score'] = (weights[0]*result['similarity']+
                             weights[1]*result['score_answer_time']+
                             weights[2]*result['score_answer_rank']+
                             weights[3]*result['score_answers_hearts']
                            )*(
                             weights[4]*result['score_activity']+
                             weights[5]*result['score_professional_hearts'])
    
    recommendation = result.groupby('answers_author_id')['final_score'].sum().sort_values(ascending=False).head(top_professionals)
    return recommendation, result
weights=[3, 1, 1, 0.3, 1, 0]
query_text = ['I will finish my college next year and would like to start a career as a Data Scientist. \n\n'\
            +'What is the best way to become a good Data Scientist? #data-science']
questions_corpus = questions[questions['questions_date_added'] > '2017-01-01']
recommendation, scores = recommend_new_question(query_text, questions=questions_corpus, top_questions=10, top_professionals=10, weights=weights)
pd.DataFrame(recommendation)
def recommend_questions_for_professional(professional_id, questions=questions, top=5, weights=[1, 1, 1, 1]):
    """ Recommends questions for an professional to answer.
        
        :param professional_id: ID of the professional
        :param top: Top N recommended questions (default: 5)
        :param top_professionals: Top N recommended professionals (default: 5)
        :param weights: Weights for the various scores (default: uniform distribution)
        :return recommendation: DataFrame with the top scored professionals
        :return sim_questions: DataFrame with all rows and columns to be able to understand the calculation.
    """ 
    professional = professionals[professionals['professionals_id'] == professional_id]
    professional_tags = professional['professionals_tags'].values[0]
    answered_questions = pd.merge(professional, answers, left_on='professionals_id', right_on='answers_author_id', how='left')
    answered_questions = pd.merge(answered_questions, questions, left_on='answers_question_id', right_on='questions_id', how='left')
    sim_tags_questions = get_similar_tags_questions(professional_tags, questions=questions, top=top**2, answered=False)[['questions_id', 'similarity']]
    sim_questions = get_similar_text_questions(answered_questions['questions_full_text'].fillna('').tolist(), questions=questions, top=top**2, answered=False)[['questions_id', 'similarity']]
    sim_questions = pd.concat([sim_questions, sim_tags_questions])
    sim_questions = sim_questions[~sim_questions['questions_id'].isin(answered_questions['questions_id'])]
    sim_questions = pd.merge(sim_questions, questions, left_on='questions_id', right_on='questions_id')
    sim_questions['time_delta_added'] = (actual_date - sim_questions['questions_date_added']).dt.days
    sim_questions['score_answers'] = 1/np.log10(10+sim_questions['questions_answers_count'])
    sim_questions['score_date_added'] = np.log10(sim_questions['time_delta_added'])
    sim_questions['score_tags'] = sim_questions['questions_tags'].fillna('').apply(lambda x: len(set(x) & set(professional_tags))/len(set(x) | set(professional_tags)) if len(set(x) | set(professional_tags)) > 0 else 0)
    sim_questions['final_score'] = (weights[0]*sim_questions['similarity']+
                                    weights[1]*sim_questions['score_answers']+
                                    weights[2]*sim_questions['score_date_added']+
                                    weights[3]*sim_questions['score_tags'])
    
    recommendation = sim_questions.groupby('questions_id')['final_score'].sum().sort_values(ascending=False).head(top)
    return recommendation, sim_questions
weights=[3, 1, 1, 3]
professional_id = 'd7f9afe721af42b1a03a993909e0568c'
questions_corpus = questions[questions['questions_date_added'] > '2018-01-01']
recommendation, score = recommend_questions_for_professional(professional_id, questions=questions_corpus, weights=weights)
pd.DataFrame(recommendation)
weights=[0, 5, 1/5, 3]
professional_id = 'ea75c5fce38348e0a151c3c346929e6a'
questions_corpus = questions[questions['questions_date_added'] > '2018-01-01']
recommendation, score = recommend_questions_for_professional(professional_id, questions=questions_corpus, weights=weights)
pd.DataFrame(recommendation)
def get_user_tags(user_id):
    """ Get a list with user tags
        
        :param user_id: ID of the user
        :return user_tags: Series with tags.
    """ 
    user_tags = tag_users[tag_users['tag_users_user_id'] == user_id]
    user_tags = pd.merge(user_tags, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')['tags_tag_name']
    return user_tags

def recommend_tags_industry(professional_id, threshold=0.01, top=5):
    """ Recommends tags for an professional depending on his industry.
        
        :param professional_id: ID of the professional
        :param threshold: Minimum percentage other professionals within the industry follow the tags.
        :param top: Top N recommended tags (default: 5)
        :return top_tags: DataFrame with the top tags and how many other professionals within industry follow these.
    """ 
    professional_tags = get_user_tags(professional_id)
    industry = professionals[professionals['professionals_id'] == professional_id]['professionals_industry']
    professionals_industry = professionals[professionals['professionals_industry'].isin(industry)]['professionals_id']
    top_tags = tag_users[tag_users['tag_users_user_id'].isin(professionals_industry)]
    top_tags = pd.merge(top_tags, tags, left_on='tag_users_tag_id', right_on='tags_tag_id')
    top_tags = top_tags[~top_tags['tags_tag_name'].isin(professional_tags)]
    top_tags = top_tags.groupby('tags_tag_name').size()/len(professionals_industry.index)
    top_tags = top_tags[top_tags > threshold]
    top_tags = top_tags.sort_values(ascending=False).head(top)
    return top_tags

def recommend_tags_questions(professional_id, threshold=0.01, top=5):
    """ Recommends tags for an professional depending on answered questions.
        
        :param professional_id: ID of the professional
        :param threshold: Minimum percentage of questions with the tags.
        :param top: Top N recommended tags (default: 5)
        :return top_tags: DataFrame with the top tags and how many answered questions have these.
    """ 
    professional_tags = get_user_tags(professional_id)
    professional = professionals[professionals['professionals_id'] == professional_id]
    professional_questions = answers[answers['answers_author_id'] == professional_id]['answers_question_id']
    top_tags = tag_questions[tag_questions['tag_questions_question_id'].isin(professional_questions)]
    top_tags = pd.merge(top_tags, tags, left_on='tag_questions_tag_id', right_on='tags_tag_id')

    top_tags = top_tags[~top_tags['tags_tag_name'].isin(professional_tags)]
    top_tags = top_tags.groupby('tags_tag_name').size()/len(professional_questions.index)
    top_tags = top_tags[top_tags > threshold]
    top_tags = top_tags.sort_values(ascending=False).head(top)
    return top_tags

def recommend_tags_other_professionals(professional_id, threshold=0.01, top=5):
    """ Recommends tags for an professional depending on answered questions.
        
        :param professional_id: ID of the professional
        :param threshold: Minimum percentage of questions with the tags.
        :param top: Top N recommended tags (default: 5)
        :return top_tags: DataFrame with the top tags and how many answered questions have these.
    """ 
    professional_tags = get_user_tags(professional_id)
    top_tags = professionals[professionals['professionals_id'] != professional_id][['professionals_tags']].dropna()
    top_tags['sim'] = top_tags['professionals_tags'].apply(lambda x: 1-len(set(professional_tags) - set(x))/len(set(professional_tags)) if len(set(professional_tags)) > 0 else 0)
    top_tags['professionals_tags'] = top_tags['professionals_tags'].apply(lambda x: set(x)-set(professional_tags))
    top_tags = top_tags[top_tags['sim'] > threshold]
    top_tags = top_tags.sort_values('sim', ascending=False).head(top ** 2)

    result = []
    for index, row in top_tags.iterrows():
        for tag in row['professionals_tags']:
            result.append([row['sim'], tag])
    result = pd.DataFrame(result, columns=['sim', 'tags_tag_name']).groupby('tags_tag_name').sum()['sim'].sort_values(ascending=False).head(top)
    return result
### %time
professional_id = '8027a8d620a7429f90598adf80db5a88'
print('Professional:', professional_id)
print('\nActual Tags:')
print(', '.join(get_user_tags(professional_id)))
print('\nRecommendation based on the user industry:')
print(', '.join(recommend_tags_industry(professional_id).index))
print('\nRecommendation based on the user answered questions:')
print(', '.join(recommend_tags_questions(professional_id).index))
print('\nRecommendation by other users with similar tags:')
print(', '.join(recommend_tags_other_professionals(professional_id).index))
print()
### %time
professional_id = '2a7db4f2150c493f9d81cb4c2e032a12'
print('Professional:', professional_id)
print('\nActual Tags:')
print(', '.join(get_user_tags(professional_id)))
print('\nRecommendation based on the user industry:')
print(', '.join(recommend_tags_industry(professional_id).index))
print('\nRecommendation based on the user answered questions:')
print(', '.join(recommend_tags_questions(professional_id).index))
print('\nRecommendation by other users with similar tags:')
print(', '.join(recommend_tags_other_professionals(professional_id).index))
print()
### %time
new_questions = questions[(questions['questions_date_added'] > '2019-01-01') & (questions['questions_answers_count'] == 0)]
questions_corpus = questions[questions['questions_date_added'] > '2018-01-01']
print('count:',new_questions.shape[0])
sim = [get_similar_text_questions([x], questions=questions_corpus)['questions_id'] for x in new_questions['questions_full_text']]
plt_data = pd.Series(np.array(sim).reshape(-1))
plt_data = plt_data.groupby(plt_data).size().sort_values(ascending=False)
print('Unique values:', len(plt_data))
print(plt_data.head())
plt_data = pd.DataFrame(plt_data.groupby(plt_data).size()/len(plt_data.index))
plt_data.plot(kind='bar', legend=None, figsize=(10, 5))
plt.xlabel('Amount of recommendations')
plt.ylabel('Frequency')
plt.title('Recommended similar questions')
plt.show()
### %time
weights=[1, 1, 1, 1, 1, 0]
new_questions = questions[(questions['questions_date_added'] > '2019-01-01') & (questions['questions_answers_count'] == 0)]
questions_corpus = questions[questions['questions_date_added'] > '2018-01-01']
print('count:',new_questions.shape[0])
sim = [recommend_new_question([x], questions=questions_corpus, top_questions=10, top_professionals=10, weights=weights)[0] 
       for x in new_questions['questions_full_text']]
plt_data = pd.Series()
for x in sim:
    plt_data = plt_data.append(pd.Series(x.index))
plt_data = plt_data.groupby(plt_data).size().sort_values(ascending=False)
print('Unique values:', len(plt_data))
print(plt_data.head())
plt_data = pd.DataFrame(plt_data.groupby(plt_data).size()/len(plt_data.index))
plt_data.plot(kind='bar', legend=None, figsize=(10, 5))
plt.xlabel('Amount of recommendations')
plt.ylabel('Frequency')
plt.title('Recommended new questions for professionals')
plt.show()
professionals[professionals['professionals_id'] == '36ff3b3666df400f956f8335cf53e09e'].T
### %time
weights=[3, 1, 1, 3]
professionals_email = emails[(emails['emails_date_sent'] > '2019-01-01') 
                              & (emails['emails_frequency_level'] == 'email_notification_weekly')]['emails_recipient_id'].unique()
questions_corpus = questions[questions['questions_date_added'] > '2018-09-01']
print('count:',professionals_email.shape[0])
recommendations = [recommend_questions_for_professional(x, questions=questions_corpus, weights=weights)[0] for x in professionals_email]
plt_data = pd.Series()
for x in recommendations:
    plt_data = plt_data.append(pd.Series(x.index))
plt_data = plt_data.groupby(plt_data).size().sort_values(ascending=False)
print('Unique values:', len(plt_data))
print(plt_data.head())
plt_data = pd.DataFrame(plt_data.groupby(plt_data).size()/len(plt_data.index))
plt_data.plot(kind='bar', legend=None, figsize=(10, 5))
plt.xlabel('Amount of recommendations')
plt.ylabel('Frequency')
plt.title('Recommended questions for professionals')
plt.show()
questions[questions['questions_id']=='8a0b92f702e444509f5bc9a95695663f'].T
### %time
professionals_email = emails[(emails['emails_date_sent'] > '2019-01-01') 
                              & (emails['emails_frequency_level'] == 'email_notification_weekly')]['emails_recipient_id'].unique()
print('count:',professionals_email.shape[0])
recommendations = [recommend_tags_industry(x) for x in professionals_email]
plt_data = pd.Series()
for x in recommendations:
    plt_data = plt_data.append(pd.Series(x.index))
plt_data = plt_data.groupby(plt_data).size().sort_values(ascending=False)
print('Unique values:', len(plt_data))
print(plt_data.head())
plt_data = pd.DataFrame(plt_data.groupby(plt_data).size()/len(plt_data.index))
plt_data.plot(kind='bar', legend=None, figsize=(10, 5))
plt.xlabel('Amount of recommendations')
plt.ylabel('Frequency')
plt.title('Recommended tags by professionals industry')
plt.show()
### %time
professionals_email = emails[(emails['emails_date_sent'] > '2019-01-01') 
                              & (emails['emails_frequency_level'] == 'email_notification_weekly')]['emails_recipient_id'].unique()
print('count:',professionals_email.shape[0])
recommendations = [recommend_tags_questions(x) for x in professionals_email]
plt_data = pd.Series()
for x in recommendations:
    plt_data = plt_data.append(pd.Series(x.index))
plt_data = plt_data.groupby(plt_data).size().sort_values(ascending=False)
print('Unique values:', len(plt_data))
print(plt_data.head())
plt_data = pd.DataFrame(plt_data.groupby(plt_data).size()/len(plt_data.index))
plt_data.plot(kind='bar', legend=None, figsize=(10, 5))
plt.xlabel('Amount of recommendations')
plt.ylabel('Frequency')
plt.title('Recommended tags by professionals answered questions')
plt.show()