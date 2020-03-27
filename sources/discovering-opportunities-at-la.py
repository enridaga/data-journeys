
from wand.image import Image as Img
Img(filename='../input/cityofla/CityofLA/Additional data/PDFs/2017/july 2017/July 21/ARTS ASSOCIATE 2454 072117 REV 072817.pdf', resolution=300)


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
import os
import numpy as np
from datetime import datetime
from collections  import Counter
from nltk import word_tokenize
import seaborn as sns
import matplotlib.pyplot as plt
import calendar
from wordcloud import WordCloud ,STOPWORDS
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
print(os.listdir("../input"))
from gensim.models import word2vec
from sklearn.manifold import TSNE
from nltk import pos_tag
from nltk.help import upenn_tagset
import gensim
import matplotlib.colors as mcolors
from nltk import jaccard_distance
from nltk import ngrams
#import textstat
plt.style.use('ggplot')
files=[dir for dir in os.walk('../input/cityofla')]
for file in files:
    print(os.listdir(file[0]))
    print("\n")
bulletins=os.listdir("../input/cityofla/CityofLA/Job Bulletins/")
additional=os.listdir("../input/cityofla/CityofLA/Additional data/")
csvfiles=[]
for file in additional:
    if file.endswith('.csv'):
        print(file)
        csvfiles.append("../input/cityofla/CityofLA/Additional data/"+file)
        
job_title=pd.read_csv(csvfiles[0])
sample_job=pd.read_csv(csvfiles[1])
kaggle_data=pd.read_csv(csvfiles[2])
job_title.head()
print("The are %d rows and %d cols in job_title file" %(job_title.shape))
    sample_job[sample_job['Field Name']=='SCHOOL_TYPE']['Description']
print("The are %d rows and %d cols in sample_job file" %(sample_job.shape))

kaggle_data.head()
print("The are %d rows and %d cols in kaggle_data file" %(kaggle_data.shape))
print("There are %d text files in bulletin directory" %len(bulletins))
def get_headings(bulletin):       
    
    """"function to get the headings from text file
        takes a single argument
        1.takes single argument list of bulletin files"""
    
    with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:    ##reading text files 
        data=f.read().replace('\t','').split('\n')
        data=[head for head in data if head.isupper()]
        return data
        
def clean_text(bulletin):      
    
    
    """function to do basic data cleaning
        takes a single argument
        1.takes single argument list of bulletin files"""
                                            
    
    with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[bulletin]) as f:
        data=f.read().replace('\t','').replace('\n','')
        return data
get_headings(1)

get_headings(2)



def to_dataframe(num,df):
    """"function to extract features from job bulletin text files and convert to
    pandas dataframe.
    function take two arguments 
                        1.the number of files to be read
                        2.dataframe object                                      """
    

    
    opendate=re.compile(r'(Open [D,d]ate:)(\s+)(\d\d-\d\d-\d\d)')       #match open date
    
    salary=re.compile(r'\$(\d+,\d+)((\s(to|and)\s)(\$\d+,\d+))?')       #match salary
    
    requirements=re.compile(r'(REQUIREMENTS?/\s?MINIMUM QUALIFICATIONS?)(.*)(PROCESS NOTE)')      #match requirements
    
    for no in range(0,num):
        with open("../input/cityofla/CityofLA/Job Bulletins/"+bulletins[no],encoding="ISO-8859-1") as f:         #reading files 
                try:
                    file=f.read().replace('\t','')
                    data=file.replace('\n','')
                    headings=[heading for heading in file.split('\n') if heading.isupper()]             ##getting heading from job bulletin

                    sal=re.search(salary,data)
                    date=datetime.strptime(re.search(opendate,data).group(3),'%m-%d-%y')
                    try:
                        req=re.search(requirements,data).group(2)
                    except Exception as e:
                        req=re.search('(.*)NOTES?',re.findall(r'(REQUIREMENTS?)(.*)(NOTES?)',
                                                              data)[0][1][:1200]).group(1)
                    
                    duties=re.search(r'(DUTIES)(.*)(REQ[A-Z])',data).group(2)
                    try:
                        enddate=re.search(
                                r'(JANUARY|FEBRUARY|MARCH|APRIL|MAY|JUNE|JULY|AUGUST|SEPTEMBER|OCTOBER|NOVEMBER|DECEMBER)\s(\d{1,2},\s\d{4})'
                                ,data).group()
                    except Exception as e:
                        enddate=np.nan
                    
                    selection= [z[0] for z in re.findall('([A-Z][a-z]+)((\s\.\s)+)',data)]     ##match selection criteria
                    
                    df=df.append({'File Name':bulletins[no],'Position':headings[0].lower(),'salary_start':sal.group(1),
                               'salary_end':sal.group(5),"opendate":date,"requirements":req,'duties':duties,
                                'deadline':enddate,'selection':selection},ignore_index=True)
                    
                    
                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)\s(years?)\s(of\sfull(-|\s)time)')
                    df['EXPERIENCE_LENGTH']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)
                    df['FULL_TIME_PART_TIME']=df['EXPERIENCE_LENGTH'].apply(lambda x:  'FULL_TIME' if x is not np.nan else np.nan )
                    
                    reg=re.compile(r'(One|Two|Three|Four|Five|Six|Seven|Eight|Nine|Ten|one|two|three|four)(\s|-)(years?)\s(college)')
                    df['EDUCATION_YEARS']=df['requirements'].apply(lambda x :  re.search(reg,x).group(1) if re.search(reg,x) is not None  else np.nan)
                    df['SCHOOL_TYPE']=df['EDUCATION_YEARS'].apply(lambda x : 'College or University' if x is not np.nan else np.nan)
                    
                except Exception as e:
                    print('umatched sequence')
                    
                    
                
                
        
           
    return df

            
            
            
            
df=pd.DataFrame(columns=['File Name','Position','salary_start','salary_end','opendate','requirements','duties','deadline'])
df=to_dataframe(len(bulletins),df)
df.to_csv('job class output.csv')
df.shape
df.head()
data_dictionary=pd.DataFrame({'Field Name':['File Name','Position','salary_start','salary_end','opendate',
                                            'requirements','duties','deadline','selection','EXPERIENCE_LENGTH','FULL_TIME_PART_TIME','EDUCATION_YEARS','SCHOOL_TYPE'],
                             })

data_dictionary['Description']=['The file name of the job bulletin from which each record came','The title of the particular class (e.g., Systems Analyst, Carpenter)',
                              'The overall salary start','The overall maximum salary','The date the job bulletin opened','Overall requirement that has to be filled',
                              'A summary of what someone does in the particular job\n','The date the job bulletin closed','list of selection criterias','Years required in a particular job class or external role.',
                              'Whether the required experience is full-time, part','Years required in a particular education program',
                               'School Type: School type required (e.g. college or university, high school)']

data_dictionary['Data Type']=['string']*13

data_dictionary['Accepts Null Values?']=['Yes']*13
data_dictionary
data_dictionary.to_csv('data dictionary.csv')
print('There are %d different jobs available' %df['Position'].nunique())
plt.figure(figsize=(8,5))
text=''.join(job for job in df['Position'])                                ##joining  data to form text
text=word_tokenize(text)
jobs=Counter(text)                                                         ##counting number of occurences
jobs_class=[job for job in jobs.most_common(12) if len(job[0])>3]          ##selecting most common words
#offers=[job[1] for job in jobs.most_common(12) if len(job[0]>3)]
a,b=map(list, zip(*jobs_class))
sns.barplot(b,a,palette='rocket')                                           ##creating barplot
plt.title('Job sectors')
plt.xlabel("count")
plt.ylabel('sector')

""""
    convert salary to proper  form 
    by removing '$' and ',' symbols.
                                    """

df['salary_start']=[int(sal.split(',')[0]+sal.split(',')[1] ) for sal in df['salary_start']]   
df['salary_end']=[sal.replace('$','')  if sal!= None else 0 for sal in df['salary_end']  ]
df['salary_end']=[int(sal.split(',')[0]+sal.split(',')[1] ) if type(sal)!=int else 0 for sal in df['salary_end']]
plt.figure(figsize=(7,5))
sns.distplot(df['salary_start'])
plt.title('salary distribution')
plt.show()
'''finding the most paid 10 jobs at LA'''

most_paid=df[['Position','salary_start']].sort_values(by='salary_start',ascending=False)[:10]
plt.figure(figsize=(7,5))
sns.barplot(y=most_paid['Position'],x=most_paid['salary_start'],palette='rocket')
plt.title('Best paid jobs in LA')
''''calculating salary start - salary end '''

df['salary_diff']=abs(df['salary_start']-df['salary_end'])

ranges=df[['Position','salary_diff']].sort_values(by='salary_diff',ascending=False)[:10]
plt.figure(figsize=(7,5))
sns.barplot(y=ranges['Position'],x=ranges['salary_diff'],palette='RdBu')   ##plotting


ranges
'''Extracting year out of opendate timestamp object and counting
    the number of each occurence of each year using count_values() '''

df['year_of_open']=[date.year for date in df['opendate']]

count=df['year_of_open'].value_counts(ascending=True)
years=['2020','2019','2018', '2017', '2016', '2015', '2014', '2013', '2012', '2008', '2006',
           '2005', '2002', '1999']
plt.figure(figsize=(7,5))
plt.plot([z for z in reversed(years)],count.values,color='blue')

plt.title('Oppurtunities over years')
plt.xlabel('years')
plt.ylabel('count')
plt.gca().set_xticklabels([z for z in reversed(years)],rotation='45')
plt.show()
experience=df['EXPERIENCE_LENGTH'].value_counts().reset_index()
experience['index']=experience['index'].apply(lambda x : x.lower())
experience=experience.groupby('index',as_index=False).agg('sum')
labels=experience['index']
sizes=experience['EXPERIENCE_LENGTH']
plt.figure(figsize=(5,7))
plt.pie(sizes,explode=(0, 0.1, 0, 0,0,0,0),labels=labels)
plt.gca().axis('equal')
plt.title('Experience value count')
plt.show()
x1=df['SCHOOL_TYPE'].value_counts()[0]
x2=df['FULL_TIME_PART_TIME'].value_counts()[0]
plt.figure(figsize=(5,5))
plt.bar(height=[x1,x2],x=['College Degree','Experience'])

'''Extracting month out of opendate timestamp object and counting
    the number of each occurence of each months using count_values() '''


plt.figure(figsize=(7,5))
df['open_month']=[z.month for z in df['opendate']]
count=df['open_month'].value_counts(sort=False)
sns.barplot(y=count.values,x=count.index,palette='rocket')
plt.gca().set_xticklabels([calendar.month_name[x] for x in count.index],rotation='45')
plt.show()
'''Extracting weekday out of opendate timestamp object and counting
    the number of each occurence of each weekday using count_values() '''


plt.figure(figsize=(7,5))

df['open_day']=[z.weekday() for z in df['opendate']]
count=df['open_day'].value_counts(sort=False)
sns.barplot(y=count.values,x=count.index,palette='rocket')
plt.gca().set_xticklabels([calendar.day_name[x] for x in count.index],rotation='45')
plt.show()
print('%d job applications may close without prior notice' %df['deadline'].isna().sum())

#df['dealine']=df['deadline'].fillna(method='backfill',inplace=True
#deadline=[datetime.strptime(x,'%B %d, %Y')  for x in df['deadline'] ]

req=' '.join(text for text in df['requirements'])


def show_wordcloud(data, title = None):
    
    
    '''funtion to produce and display wordcloud
        taken 2 arguments
        1.data to produce wordcloud
        2.title of wordcloud'''
    
    
    wordcloud = WordCloud(
        background_color='white',
        stopwords=set(STOPWORDS),
        max_words=250,
        max_font_size=40, 
        scale=3,
        random_state=1 # chosen at random by flipping a coin; it was heads
    ).generate(str(data))

    fig = plt.figure(1, figsize=(12, 12))
    plt.axis('off')
    if title: 
        fig.suptitle(title, fontsize=20)
        fig.subplots_adjust(top=2.3)

    plt.imshow(wordcloud)
    plt.show()
show_wordcloud(text,'REQUIREMENTS')
lem=WordNetLemmatizer()
text=[lem.lemmatize(w) for w in word_tokenize(req)]
vect=TfidfVectorizer(ngram_range=(1,3),max_features=100)
vectorized_data=vect.fit_transform(text)
#id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vect.vocabulary_.keys()
def build_corpus(df,col):
    
    '''function to build corpus from dataframe'''
    lem=WordNetLemmatizer()
    corpus= []
    for x in df[col]:
        
        
        words=word_tokenize(x)
        corpus.append([lem.lemmatize(w) for w in words])
    return corpus

corpus=build_corpus(df,'requirements')
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=30, workers=4)

def tsne_plot(model,title='None'):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=80, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(12, 12)) 
    plt.title(title)
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()
tsne_plot(model,'Requirements')
token=word_tokenize(req)
counter=Counter(token)
count=[x[0] for x in counter.most_common(40) if len(x[0])>3]
print("Most common words in Requirement")
print(count)
duties= ' '.join(d for d in df['duties'])
show_wordcloud(duties,'Duties')
lem=WordNetLemmatizer()
text=[lem.lemmatize(w) for w in word_tokenize(duties)]
vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)
vectorized_data=vect.fit_transform(text)
#id_map=dict((v,k) for k,v in vect.vocabulary_.items())
vect.vocabulary_.keys()
token=word_tokenize(duties)
counter=Counter(token)
count=[x[0] for x in counter.most_common(40) if len(x[0])>3]
print("Most common words in Duties")
print(count)
corpus=build_corpus(df,'duties')
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=40, workers=4)

tsne_plot(model,'Duties')
lem=WordNetLemmatizer()
text=[lem.lemmatize(w) for w in word_tokenize(duties)]
vect=TfidfVectorizer(ngram_range=(1,3),max_features=200)
vectorized_data=vect.fit_transform(text)
id2word=dict((v,k) for k,v in vect.vocabulary_.items())


corpus=gensim.matutils.Sparse2Corpus(vectorized_data,documents_columns=False)
ldamodel = gensim.models.ldamodel.LdaModel(corpus,id2word=id2word,num_topics=8,random_state=34,passes=25,per_word_topics=True)

ldamodel.show_topic(1)
def format_topics_sentences(ldamodel, corpus, texts):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row_list in enumerate(ldamodel[corpus]):
        row = row_list[0] if ldamodel.per_word_topics else row_list            
        # print(row)
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=build_corpus(df,'duties'))

# Format
df_dominant_topic = df_topic_sents_keywords.reset_index()
df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']
df_dominant_topic.dropna(inplace=True)
df_dominant_topic.head(5)
data=build_corpus(df,'duties')
topics = ldamodel.show_topics(formatted=False)
data_flat = [w for w_list in build_corpus(df,'duties') for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df_plot= pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(4, 2, figsize=(10,12), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df_plot.loc[df_plot.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df_plot.loc[df_plot.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    #ax_twin.set_ylim(0, 0.040); ax.set_ylim(0, 4000)
    ax.set_title('Topic: ' + str(i), color=cols[i])
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df_plot.loc[df_plot.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords',y=1)    
plt.show()

plt.figure(figsize=(7,7))
count=df['selection'].astype(str).value_counts()[:10]
sns.barplot(y=count.index,x=count,palette='rocket')
plt.gca().set_yticklabels(count.index,rotation='45')
plt.show()

def pronoun(data):
    
    '''function to tokenize data and perform pos_tagging.Returns tokens having "PRP" tag'''
    
    prn=[]
    vrb=[]
    token=word_tokenize(data)
    pos=pos_tag(token)
   
    vrb=Counter([x[0] for x in pos if x[1]=='PRP'])
    
    return vrb
    


req_prn=pronoun(req)
duties_prn=pronoun(duties)
print('pronouns used in requirement section are')
print(req_prn.keys())
print('\npronouns used in duties section are')
print(duties_prn.keys())

names=['senior waterman','policeman']
for name in names:
    z=re.match(r'\w+?\s?\w+(man|women)$',name)
    print(z)
for name in df['Position']:
    z=re.match(r'\w+?\s?\w+(man|women)$',name)
    if z is not None:
        print(z)
    

def similar_jobs(job):
    
    ''' function to find and return jobs with similar job title.take a single argument
            - job title
            returns
                -list of similar jobs '''
    
    word1=word_tokenize(job)
    jobs=[]
    for i,name in enumerate(df['Position']):
        word2=word_tokenize(name)
        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))
        if(distance<.55):
            jobs.append((name,i))
    return jobs
similar_jobs(df['Position'][10])
def similar_req(job):
    
    ''' function to find and return jobs with similar job title.take a single argument
            - job title
            returns
                -list of similar jobs '''
    
    word1=word_tokenize(job)
    jobs=[]
    for i,name in enumerate(df['requirements']):
        word2=word_tokenize(name)
        distance=jaccard_distance(set(ngrams(word1,n=1)),set(ngrams(word2,n=1)))
        if(distance<.5):
            jobs.append((name,df.iloc[i]['Position']))
    return jobs
similar_req(df['requirements'][10])
df['requirements'][312]
reading=[]
for file in df['File Name']:
    text=open("../input/cityofla/CityofLA/Job Bulletins/"+file,'r',encoding="ISO-8859-1").read()
    sentence = text.count('.') + text.count('!') + text.count(';') + text.count(':') + text.count('?')
    words = len(text.split())
    syllable = 0
    for word in text.split():
        for vowel in ['a','e','i','o','u']:
            syllable += word.count(vowel)
        for ending in ['es','ed','e']:
            if word.endswith(ending):
                   syllable -= 1
        if word.endswith('le'):
            syllable += 1
            
    G = round((0.39*words)/sentence+ (11.8*syllable)/words-15.59)
    reading.append(G)
plt.hist(reading)
plt.xlabel('Flesch Index')
plt.title('Flesch index distribution')
plt.show()
