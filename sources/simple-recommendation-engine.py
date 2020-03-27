
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
os.listdir("../input/movielens-20m-dataset")
os.chdir("../input/movielens-20m-dataset/")
genome_tags = pd.read_csv("genome_tags.csv")

# We drop this dataset, since this doesn't have any useful features for predictions
link = pd.read_csv("link.csv")
genome_scores = pd.read_csv("genome_scores.csv")

# For efficiency and compatability We pick top 5000 rows
movies = pd.read_csv("movie.csv",nrows=5000)
rating = pd.read_csv("rating.csv")
tag = pd.read_csv("tag.csv")

# Dataset shape
print("genome_tags shape is {}".format(genome_tags.shape))
print("genome_scores shape is {}".format(genome_scores.shape))
print("movies shape is {}".format(movies.shape))
print("rating shape is {}".format(rating.shape))
print("tag shape is {}".format(tag.shape))
print(genome_scores.columns)
print(movies.columns)
print(rating.columns)
print(tag.columns)
# genome_scores dataset has relevance feature which says that how much a tag is relevant to the movie and
# it's value range from 0 to 1, we'll consider only the value which has more than 0.5 relevance. So this gives better 
# predicrion. And We'll merge the tag with genome_scores dataset.
genome_scores = genome_scores[genome_scores['relevance']> 0.5].merge(genome_tags,on='tagId',how='left') 

# concatenating all the tag that belongs to a movie and forming a tag collection for each movie
genome_scores = genome_scores.groupby('movieId')['tag'].apply(' '.join).reset_index()
final_dataset = pd.merge(movies,genome_scores,on='movieId',how='left')
# renaming tag as keywords
tag = tag.rename(columns = {"tag":"keywords"})
tag['keywords'].fillna('',inplace=True)
tag = tag.groupby('movieId')['keywords'].apply(' '.join).reset_index()
final_dataset = pd.merge(final_dataset,tag,on='movieId',how='left')
final_dataset.head()
final_dataset['genres'].head()
final_dataset['keywords'] = final_dataset['keywords'] + " " +final_dataset['tag'] +  " " + \
    final_dataset['genres'].str.replace("|"," ")
final_dataset['keywords'].fillna("",inplace=True)
# rating will be used for collabarative filtering, so we'll skip this now
# final_dataset = pd.merge(final_dataset,rating,on='movieId',how='left')
# Both tag and genres values has added to keywords so we drop this 
final_dataset.drop(['tag','genres'],inplace=True,axis=1)
final_dataset.columns
c_vect = TfidfVectorizer(stop_words='english')
X = c_vect.fit_transform(final_dataset['keywords'])
# There are other similiary distance metric available which are euclidean distance,manhattan distance, Pearson coefficient etc
# But for sparse matrix cosine similarity works better
cosine_sim = cosine_similarity(X)
def get_movie_recommendation(movie_name):
    idx = final_dataset[final_dataset['title'].str.contains(movie_name)].index
    if len(idx):
        movie_indices = sorted(list(enumerate(cosine_sim[idx[0]])), key=lambda x: x[1], reverse=True)[1:11]
        movie_indices = [i[0] for i in movie_indices]
        return movie_indices
    else : 
        return []
title = "Toy Story 2"
recommended_movie_list = get_movie_recommendation(title)
movies.iloc[recommended_movie_list].set_index('movieId')
