
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# load the data
train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')
# First, lets count how much data we have!
train_len, test_len = len(train_df.index), len(test_df.index)
print(f'train size: {train_len}, test size: {test_len}')
# also, lets take a quick look at what we have 
train_df.head()
# its always a good idea to count the amount of missing values before diving into any analysis
# Lets also see how many missing values (in percentage) we are dealing with
miss_val_train_df = train_df.isnull().sum(axis=0) / train_len
miss_val_train_df = miss_val_train_df[miss_val_train_df > 0] * 100
miss_val_train_df
# lets create a list of all the identities tagged in this dataset. This list given in the data section of this competition. 
identities = ['male','female','transgender','other_gender','heterosexual','homosexual_gay_or_lesbian',
              'bisexual','other_sexual_orientation','christian','jewish','muslim','hindu','buddhist',
              'atheist','other_religion','black','white','asian','latino','other_race_or_ethnicity',
              'physical_disability','intellectual_or_learning_disability','psychiatric_or_mental_illness',
              'other_disability']
# getting the dataframe with identities tagged
train_labeled_df = train_df.loc[:, ['target'] + identities ].dropna()
# lets define toxicity as a comment with a score being equal or .5
# in that case we divide it into two dataframe so we can count toxic vs non toxic comment per identity
toxic_df = train_labeled_df[train_labeled_df['target'] >= .5][identities]
non_toxic_df = train_labeled_df[train_labeled_df['target'] < .5][identities]
# at first, we just want to consider the identity tags in binary format. So if the tag is any value other than 0 we consider it as 1.
toxic_count = toxic_df.where(train_labeled_df == 0, other = 1).sum()
non_toxic_count = non_toxic_df.where(train_labeled_df == 0, other = 1).sum()
# now we can concat the two series together to get a toxic count vs non toxic count for each identity
toxic_vs_non_toxic = pd.concat([toxic_count, non_toxic_count], axis=1)
toxic_vs_non_toxic = toxic_vs_non_toxic.rename(index=str, columns={1: "non-toxic", 0: "toxic"})
# here we plot the stacked graph but we sort it by toxic comments to (perhaps) see something interesting
toxic_vs_non_toxic.sort_values(by='toxic').plot(kind='bar', stacked=True, figsize=(30,10), fontsize=20).legend(prop={'size': 20})
# First we multiply each identity with the target
weighted_toxic = train_labeled_df.iloc[:, 1:].multiply(train_labeled_df.iloc[:, 0], axis="index").sum() 
# changing the value of identity to 1 or 0 only and get comment count per identity group
identity_label_count = train_labeled_df[identities].where(train_labeled_df == 0, other = 1).sum()
# then we divide the target weighted value by the number of time each identity appears
weighted_toxic = weighted_toxic / identity_label_count
weighted_toxic = weighted_toxic.sort_values(ascending=False)
# plot the data using seaborn like before
plt.figure(figsize=(30,20))
sns.set(font_scale=3)
ax = sns.barplot(x = weighted_toxic.values , y = weighted_toxic.index, alpha=0.8)
plt.ylabel('Demographics')
plt.xlabel('Weighted Toxicity')
plt.title('Weighted Analysis of Most Frequent Identities')
plt.show()
# lets take the dataset with identitiy tags, created date, and target column
with_date_df = train_df.loc[:, ['created_date', 'target'] + identities].dropna()
# next we will create a weighted dataframe for each identity tag (like we did before)
# first we divide each identity tag with the total value it has in the dataset
weighted_df = with_date_df.iloc[:, 2:] / with_date_df.iloc[:, 2:].sum()
# then we multiplty this value with the target 
target_weighted_df = weighted_df.multiply(with_date_df.iloc[:, 1], axis="index")
# lets add a column to count the number of comments
target_weighted_df['comment_count'] = 1
# now we add the date to our newly created dataframe (also parse the text date as datetime)
target_weighted_df['created_date'] = pd.to_datetime(with_date_df['created_date'].apply(lambda dt: dt[:10]))
# now we can do a group by of the created date to count the number of times a identity appears for that date
identity_weight_per_date_df = target_weighted_df.groupby(['created_date']).sum().sort_index()
# lets group most of the identities into three major categories as follows for simplified analysis
races = ['black','white','asian','latino','other_race_or_ethnicity']
religions = ['atheist', 'buddhist', 'christian', 'hindu', 'muslim', 'jewish','other_religion']
sexual_orientation = ['heterosexual', 'homosexual_gay_or_lesbian', 'bisexual', 'other_sexual_orientation']
# lets create a column to aggregate our weighted toxicity score per identity group
identity_weight_per_date_df['races_total'] = identity_weight_per_date_df[races].sum(axis=1)
identity_weight_per_date_df['religions_total'] = identity_weight_per_date_df[religions].sum(axis=1)
identity_weight_per_date_df['sexual_orientation_total'] = identity_weight_per_date_df[sexual_orientation].sum(axis=1)
# and then plot a time-series line plot per identity group
identity_weight_per_date_df[['races_total', 'religions_total', 'sexual_orientation_total']].plot(figsize=(15,7), linewidth=1, fontsize=15) 
plt.legend(loc=2, prop={'size': 15})
plt.xlabel('Comment Date', fontsize=15)
plt.ylabel('Weighted Toxic Score', fontsize=15)
identity_weight_per_date_df['comment_count'].plot(figsize=(15,7), linewidth=1, fontsize=15)
plt.xlabel('Comment Date', fontsize = 15)
plt.ylabel('Total Comments', fontsize = 15)
# lets divide by the comment count for the date to get a relative weighted toxic score
identity_weight_per_date_df['races_rel'] = identity_weight_per_date_df['races_total'] / identity_weight_per_date_df['comment_count']
identity_weight_per_date_df['religions_rel'] = identity_weight_per_date_df['religions_total'] / identity_weight_per_date_df['comment_count']
identity_weight_per_date_df['sexual_orientation_rel'] = identity_weight_per_date_df['sexual_orientation_total']  / identity_weight_per_date_df['comment_count']
# now lets plot the data
identity_weight_per_date_df[['races_rel', 'religions_rel', 'sexual_orientation_rel']].plot(figsize=(15,7), linewidth=1, fontsize=20) 
plt.legend(loc=2, prop={'size': 15})
plt.xlabel('Comment Date', fontsize=15)
plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets plot relative weighted toxic score for each identity of races
identity_weight_per_date_df[races].div(identity_weight_per_date_df['comment_count'], axis=0).plot(figsize=(15,7), linewidth=1, fontsize=15)
plt.legend(loc=2, prop={'size': 15})
plt.xlabel('Comment Date', fontsize=15)
plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets plot relative weighted toxic score for each identity of religions
identity_weight_per_date_df[religions].div(identity_weight_per_date_df['comment_count'], axis=0).plot(figsize=(15,7), linewidth=1, fontsize=15)
plt.legend(loc=2, prop={'size': 15})
plt.xlabel('Comment Date', fontsize=15)
plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets plot relative weighted toxic score for each identity of sexual orientation
identity_weight_per_date_df[sexual_orientation].div(identity_weight_per_date_df['comment_count'], axis=0).plot(figsize=(15,7), linewidth=1, fontsize=20)
plt.legend(loc=2, prop={'size': 15})
plt.xlabel('Comment Date', fontsize=15)
plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
# lets import the algorithm
from scipy.signal import argrelextrema
# we will store all the different datapoints in the following dataframe
peaks_df = pd.DataFrame()

# first we loop through all the different identities we are interested in
for col in races + religions + sexual_orientation:
    # we pass the values through the algorithm to get an index of the rel. maximums 
    _max_index = argrelextrema(identity_weight_per_date_df[col].values, np.greater, order=15)
    # we use the index returned to create a dataframe of the values for those index. in this case 
    # we are interested in the created date and the score, notice how the dataframe needs to be 
    # transformed because of the orientation of the arrays we started off with
    col_peaks_df = pd.DataFrame(data = [identity_weight_per_date_df.index[_max_index], identity_weight_per_date_df[col].values[_max_index]]).T
    col_peaks_df.columns = ['created_date','score']
    # we create a new column labeling the identity so we can track which peak came from which identity
    col_peaks_df['identity'] = col
    # and we keep appending to our main dataframe 
    peaks_df = peaks_df.append(col_peaks_df)
# lets set identity as our index and we are done
peaks_df = peaks_df.set_index('identity')
# to count the number of days from the first comment, we take our labeled data and convert the created date column
comments_with_date_df = train_df.loc[:, ['created_date', 'target','comment_text'] + identities].dropna()
comments_with_date_df['created_date'] = pd.to_datetime(with_date_df['created_date'].apply(lambda dt: dt[:10]))
comments_with_date_df['comment_count'] = 1
# calculate days from first comment
first_dt = min(comments_with_date_df['created_date'].values)
last_dt = max(comments_with_date_df['created_date'].values)
peaks_df['days_from_first'] = (peaks_df['created_date'] - first_dt).dt.days
# here is a peak at what our peaks_df looks like
peaks_df.head()
# lets create a function that returns the peaks dataframe for a given identity
# we also want to get the number of toxic comments made against that identity in the dataframe
def get_identity_peaks_df(identity, peaks_df, comments_with_date_df):
    # select subset and sort
    identity_peaks_df = peaks_df[peaks_df.index==identity].sort_values(by='score', ascending=False)
    # change the score type to float
    identity_peaks_df['score'] = identity_peaks_df.score.astype(float)
    # use created date as the index so we can join over in later step
    identity_peaks_df = identity_peaks_df.set_index('created_date')
    # calculate how many toxic comments were made targetting the given identity group
    identity_comment_count_df = comments_with_date_df[comments_with_date_df[identity] > 0][['created_date','comment_count']].groupby('created_date').sum()
    # do an inner join to also get the total number of comments made that day for the given identity
    identity_peaks_df = identity_peaks_df.join(identity_comment_count_df)
    return identity_peaks_df
# to make our scatter plot more presentable we will set the max and min of our y axis
y_lim_min = peaks_df['score'].max() + peaks_df['score'].max() / 3 # adding a little bit head room on y axis
y_lim_max = peaks_df['score'].min() 
# now lets write a function that draws the scatter plot for a given identity
def identity_scatter_plot(identity, identity_peaks_df, y_lim_min, y_lim_max):
    x = identity_peaks_df['days_from_first'].values
    y = identity_peaks_df['score'].values
    size = identity_peaks_df['comment_count'].values
    label = identity_peaks_df['comment_count'].index
    plt.figure(figsize=(15,7))
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.scatter(x, y, s=size, label=label)
    plt.ylim(y_lim_max, y_lim_min)
    axis_font = {'fontname':'Arial', 'size':'14'}
    plt.title('Relative Maximums - Targeted Against '+ identity.capitalize() +' Identity', fontsize=15)
    plt.xlabel('Comment Date', fontsize=15)
    plt.ylabel('Relative Weighted Toxic Score', fontsize=15)
identity = 'white'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_scatter_plot(identity, identity_peaks_df, y_lim_min, y_lim_max)
identity = 'black'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_scatter_plot(identity, identity_peaks_df, y_lim_min, y_lim_max)
identity = 'homosexual_gay_or_lesbian'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_scatter_plot(identity, identity_peaks_df, y_lim_min, y_lim_max)
identity = 'muslim'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_scatter_plot(identity, identity_peaks_df, y_lim_min, y_lim_max)
identity = 'jewish'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_scatter_plot(identity, identity_peaks_df, y_lim_min, y_lim_max)
# Lets import seaborn
import seaborn as sns
# Compute the correlation matrix
corr = comments_with_date_df[identities].corr()
# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
sns.set(font_scale = 1)
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
# import wordcloud
from wordcloud import WordCloud
# import NLTK mainly for stopwords
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
# we will write a simple function to generate the wordcloud per identity group
def generate_word_cloud(identity, toxic_comments, non_toxic_comments):
    # convert stop words to sets as required by the wordcloud library
    stop_words = set(stopwords.words("english"))
    # create toxic wordcloud
    wordcloud_toxic = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(toxic_comments)
    # create non-toxic wordcloud
    wordcloud_non_toxic = WordCloud(max_font_size=100, max_words=100, background_color="white", stopwords=stop_words).generate(non_toxic_comments)
    # draw the two wordclouds side by side using subplot
    fig = plt.figure(figsize=[15,5])
    fig.add_subplot(1, 2, 1).set_title("Toxic Wordcloud", fontsize=10)
    plt.imshow(wordcloud_toxic, interpolation="bilinear")
    plt.axis("off")
    fig.add_subplot(1, 2, 2).set_title("Non Toxic Wordcloud", fontsize=10)
    plt.imshow(wordcloud_non_toxic, interpolation="bilinear")
    plt.axis("off")
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Word Cloud - {} Identity'.format(identity), size = 16)
    plt.show()
# this function returns a tuple (toxic comments, non toxic comments) for the peaks
def get_comments(identity, identity_peaks_df):
    # create an empty array to return comments
    toxic_comments, non_toxic_comments = np.array([]), np.array([])
    # go over all the dates and grab the relevant comments for the given identity
    for dt in identity_peaks_df.index:
        # get the toxic comments
        comments_dt_df = comments_with_date_df[(comments_with_date_df['created_date'] == dt) \
                                               & (comments_with_date_df[identity] > 0) \
                                               & (comments_with_date_df['target'] >= .5)]
        toxic_comments = np.append(toxic_comments, comments_dt_df['comment_text'].values)
        
        # get the non toxic comments
        comments_dt_df = comments_with_date_df[(comments_with_date_df['created_date'] == dt) \
                                               & (comments_with_date_df[identity] > 0) \
                                               & (comments_with_date_df['target'] < .5)]
        non_toxic_comments = np.append(non_toxic_comments, comments_dt_df['comment_text'].values)
    
    return (toxic_comments, non_toxic_comments)
# lets draw the wordcloud for white identity
identity = 'white'
# get the peaks for the given identity
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
# get the comments for the given identity for the peaks
identity_comments = get_comments(identity, identity_peaks_df)
# lets convert the comments as one long string (as needed by wordcloud)
toxic_comments = ' '.join(identity_comments[0])
non_toxic_comments = ' '.join(identity_comments[1])
# draw the wordcloud using the function we created earlier
generate_word_cloud(identity, toxic_comments, non_toxic_comments)
# like white identity, lets draw the wordcloud for the other top identities
identity = 'black'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_comments = get_comments(identity, identity_peaks_df)
toxic_comments = ' '.join(identity_comments[0])
non_toxic_comments = ' '.join(identity_comments[1])
generate_word_cloud(identity, toxic_comments, non_toxic_comments)
identity = 'homosexual_gay_or_lesbian'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_comments = get_comments(identity, identity_peaks_df)
toxic_comments = ' '.join(identity_comments[0])
non_toxic_comments = ' '.join(identity_comments[1])
generate_word_cloud(identity, toxic_comments, non_toxic_comments)
identity = 'muslim'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_comments = get_comments(identity, identity_peaks_df)
toxic_comments = ' '.join(identity_comments[0])
non_toxic_comments = ' '.join(identity_comments[1])
generate_word_cloud(identity, toxic_comments, non_toxic_comments)
identity = 'jewish'
identity_peaks_df = get_identity_peaks_df(identity, peaks_df, comments_with_date_df)
identity_comments = get_comments(identity, identity_peaks_df)
toxic_comments = ' '.join(identity_comments[0])
non_toxic_comments = ' '.join(identity_comments[1])
generate_word_cloud(identity, toxic_comments, non_toxic_comments)
toxic_comments = comments_with_date_df[comments_with_date_df['target'] >= .5]['comment_text'].values
non_toxic_comments = comments_with_date_df[comments_with_date_df['target'] < .5]['comment_text'].values
toxic_comments = ' '.join(toxic_comments)
non_toxic_comments = ' '.join(non_toxic_comments)
generate_word_cloud('All', toxic_comments, non_toxic_comments)
import emoji
from collections import Counter
from itertools import chain
# we will use this simple function to process a string and return all the emojis as a list
def extract_emojis(str):
    return [c for c in str if c in emoji.UNICODE_EMOJI]
# createa new column to count the number of emojis per row / comment
comments_with_date_df['emoji_count'] = comments_with_date_df['comment_text'].apply(lambda x: len(extract_emojis(x)))
# here we will count the mean or average number of emojis for toxic and non toxic comments per identity

emoji_mean_per_identity = []

for identity in identities:
    toxic_emoji_mean = comments_with_date_df[(comments_with_date_df[identity]> 0) & (comments_with_date_df['target'] >= .5)]['emoji_count'].mean()
    non_toxic_emoji_mean = comments_with_date_df[(comments_with_date_df[identity]> 0) & (comments_with_date_df['target'] < .5)]['emoji_count'].mean()
    emoji_mean_per_identity.append([identity, toxic_emoji_mean, non_toxic_emoji_mean])

emoji_mean_per_identity_df = pd.DataFrame(emoji_mean_per_identity, columns = ['identity','toxic','non toxic']).set_index('identity')
# now we can plot our dataframe and see what we have
emoji_mean_per_identity_df.plot.bar(figsize=(15,5))
plt.ylabel('mean emojis per comment')
plt.title('Emojis usage in comments for different identities')
# createa new column to count the number of emojis per row / comment
comments_with_date_df['uses_emoji'] = comments_with_date_df['comment_text'].apply(lambda x: 1 if len(extract_emojis(x)) > 0 else 0)
comment_using_emoji_per_identity = []

for identity in identities:
    toxic_emoji_mean = comments_with_date_df[(comments_with_date_df[identity]> 0) & (comments_with_date_df['target'] >= .5)]['uses_emoji'].mean()
    non_toxic_emoji_mean = comments_with_date_df[(comments_with_date_df[identity]> 0) & (comments_with_date_df['target'] < .5)]['uses_emoji'].mean()
    comment_using_emoji_per_identity.append([identity, toxic_emoji_mean, non_toxic_emoji_mean])

comment_using_emoji_per_identity_df = pd.DataFrame(comment_using_emoji_per_identity, columns = ['identity','toxic','non toxic']).set_index('identity')
# now we can plot our dataframe and see what we have
comment_using_emoji_per_identity_df.plot.bar(figsize=(15,5))
plt.ylabel('mean for at least one emoji used in a comment')
plt.title('Emojis used in comment for different identities - Normalized')
# take the relevant subset of our data
severe_toxicity_df = train_df.loc[ : ,['target','severe_toxicity']]
# make a categorical column to classify a comment toxic and non toxic
severe_toxicity_df['is_toxic'] = severe_toxicity_df['target'].apply(lambda x: 'Toxic' if x >= .5 else 'Not Toxic')
# draw boxplot using subplot
fig, axes = plt.subplots(1,2, figsize=(15,5))
# boxplot for target for toxic and non-toxic comments
a1 = severe_toxicity_df.boxplot('target', by="is_toxic", ax=axes.flatten()[0])
# boxplot for severe toxicity for toxic and non toxic comments
a2 = severe_toxicity_df.boxplot('severe_toxicity', by="is_toxic", ax=axes.flatten()[1])
# remove the ugly title that is generated for boxplot by default
plt.suptitle("")
# show me the plot
plt.tight_layout() 
plt.show()
# lets collect the comments with severe toxicity so we can do some further study 
comments_severe_toxicity_df = train_df.loc[:, ['severe_toxicity','comment_text'] + identities].dropna()
# get the 75th percentile score for severe toxicity for only toxic comments
comments_severe_toxicity_q75 = comments_severe_toxicity_df[comments_severe_toxicity_df['severe_toxicity'] > 0]['severe_toxicity'].quantile(q=.75)
# lets do the word cloud!
toxic_comments = comments_severe_toxicity_df[comments_severe_toxicity_df['severe_toxicity'] >= comments_severe_toxicity_q75]['comment_text'].values
non_toxic_comments = comments_severe_toxicity_df[comments_severe_toxicity_df['severe_toxicity']==0]['comment_text'].values
toxic_comments = ' '.join(toxic_comments)
non_toxic_comments = ' '.join(non_toxic_comments)
generate_word_cloud('Severe Toxicity - All', toxic_comments, non_toxic_comments)
import multiprocessing
from nltk import word_tokenize, sent_tokenize
from gensim.models.word2vec import Word2Vec
from itertools import chain
from sklearn.manifold import TSNE
# first lets capture all the comments we have, training and test
comments = np.append(train_df['comment_text'].values, test_df['comment_text'].values)
# we will use the sentence tokenizer from the NLTK package to turn each comment into a list of sentences
comments = [sent_tokenize(comment) for comment in comments]
# we will flatten our list so we just have a sequence of sentences
comments = list(chain.from_iterable(comments))
# then we use the word tokenizer to create list of words per sentence 
# this is the format we need to transform our data before feeding word2vec model
comments = [word_tokenize(comment.lower()) for comment in comments]
# train our model
model = Word2Vec(sentences=comments, size=64, sg=1, window=10, min_count=5, seed=42, workers=multiprocessing.cpu_count())
def display_closestwords_tsnescatterplot(model, dim, word, topn):
    
    arr = np.empty((0,dim), dtype='f')
    word_labels = [word]

    # get close words
    close_words = model.wv.similar_by_word(word, topn)
    
    # add the vector for each of the closest words to the array
    arr = np.append(arr, np.array([model.wv[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model.wv[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
        
    # find tsne coords for 2 dimensions
    tsne = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)

    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    # display scatter plot
    plt.figure(figsize=(15,5))
    plt.scatter(x_coords, y_coords, marker='', )
    
    
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=14)
    
    plt.xlim(x_coords.min() - 1, x_coords.max() + 1)
    plt.ylim(y_coords.min() - 1, y_coords.max() + 1)
    plt.axis('off')
    plt.suptitle('Similar Words for Identity {}'.format(word.capitalize()), fontsize=20)
    plt.show()
display_closestwords_tsnescatterplot(model, 64, 'white', 25) 
display_closestwords_tsnescatterplot(model, 64, 'black', 25)
display_closestwords_tsnescatterplot(model, 64, 'homosexual', 25)
display_closestwords_tsnescatterplot(model, 64, 'gay', 25)
display_closestwords_tsnescatterplot(model, 64, 'lesbian', 25)
display_closestwords_tsnescatterplot(model, 64, 'muslim', 25)
display_closestwords_tsnescatterplot(model, 64, 'jewish', 25)
