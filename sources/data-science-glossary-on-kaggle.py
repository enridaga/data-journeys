
import pandas as pd
from IPython.core.display import HTML

path = "../input/"

versions = pd.read_csv(path+"KernelVersions.csv")
kernels = pd.read_csv(path+"Kernels.csv")
users = pd.read_csv(path+"Users.csv")

language_map = {'1' : 'R','5' : 'R', '12' : 'R', '13' : 'R', '15' : 'R', '16' : 'R',
                '2' : 'Python','8' : 'Python', '9' : 'Python', '14' : 'Python'}

def pressence_check(title, tokens, ignore = []):
    present = False
    for token in tokens:
        words = token.split()
        if all(wrd.lower().strip() in title.lower() for wrd in words):
            present = True
    for token in ignore:
        if token in title.lower():
            present = False
    return present 

## check if the latest version of the kernel is about the same topic 
def get_latest(idd):
    latest = versions[versions['KernelId'] == idd].sort_values('VersionNumber', ascending = False).iloc(0)[0]
    return latest['VersionNumber']

def get_kernels(tokens, n, ignore = []):
    versions['isRel'] = versions['Title'].apply(lambda x : pressence_check(x, tokens, ignore))
    relevant = versions[versions['isRel'] == 1]
    results = relevant.groupby('KernelId').agg({'TotalVotes' : 'sum', 
                                                'KernelLanguageId' : 'max', 
                                                'Title' : lambda x : "#".join(x).split("#")[-1],
                                                'VersionNumber' : 'max'})
    results = results.reset_index().sort_values('TotalVotes', ascending = False).head(n)
    results = results.rename(columns={'KernelId' : 'Id', 'TotalVotes': 'Votes'})


    results['latest_version']  = results['Id'].apply(lambda x : get_latest(x))
    results['isLatest'] = results.apply(lambda r : 1 if r['VersionNumber'] == r['latest_version'] else 0, axis=1)
    results = results[results['isLatest'] == 1]

    results = results.merge(kernels, on="Id").sort_values('TotalVotes', ascending = False)
    results = results.merge(users.rename(columns={'Id':"AuthorUserId"}), on='AuthorUserId')
    results['Language'] = results['KernelLanguageId'].apply(lambda x : language_map[str(x)] if str(x) in language_map else "")
    results = results.sort_values("TotalVotes", ascending = False)
    return results[['Title', 'CurrentUrlSlug','Language' ,'TotalViews', 'TotalComments', 'TotalVotes', "DisplayName","UserName"]]


def best_kernels(tokens, n = 10, ignore = [], idd = "one"):
    response = get_kernels(tokens, n, ignore)     
    hs = """<style>
                .rendered_html tr {font-size: 12px; text-align: left}
            </style>
            <h3 id='"""+ idd +"""'><font color="#1768ea">"""+tokens[0].title()+"""</font></h3>
            <table>
            <th>
                <td><b>Kernel</b></td>
                <td><b>Author</b></td>
                <td><b>Language</b></td>
                <td><b>Views</b></td>
                <td><b>Comments</b></td>
                <td><b>Votes</b></td>
            </th>"""
    for i, row in response.iterrows():
        url = "https://www.kaggle.com/"+row['UserName']+"/"+row['CurrentUrlSlug']
        aurl= "https://www.kaggle.com/"+row['UserName']
        hs += """<tr>
                    <td>"""+str(i+1)+"""</td>
                    <td><a href="""+url+""" target="_blank"><b>"""  + row['Title'] + """</b></a></td>
                    <td><a href="""+aurl+""" target="_blank">"""  + row['DisplayName'] + """</a></td>
                    <td>"""+str(row['Language'])+"""</td>
                    <td>"""+str(row['TotalViews'])+"""</td>
                    <td>"""+str(row['TotalComments'])+"""</td>
                    <td>"""+str(row['TotalVotes'])+"""</td>
                    </tr>"""
    hs += "</table>"
    display(HTML(hs))
tokens = ["linear regression"]
best_kernels(tokens, 10, idd="1.1")
tokens = ['logistic regression', "logistic"]
best_kernels(tokens, 10, idd="1.2")
tokens = ['Ridge']
best_kernels(tokens, 10, idd="2.1")
tokens = ['Lasso']
best_kernels(tokens, 10, idd="2.2")
tokens = ['ElasticNet']
best_kernels(tokens, 4, idd="2.3")
tokens = ['Decision Tree']
best_kernels(tokens, 10, idd="3.1")
tokens = ['random forest']
best_kernels(tokens, 10, idd="3.2")
tokens = ['lightgbm', 'light gbm', 'lgb']
best_kernels(tokens, 10, idd="3.3")
tokens = ['xgboost', 'xgb']
best_kernels(tokens, 10, idd="3.4")
tokens = ['catboost']
best_kernels(tokens, 10, idd="3.5")
tokens = ['gradient boosting']
best_kernels(tokens, 10, idd="3.6")
tokens = ['neural network']
best_kernels(tokens, 10, idd="4.1")
tokens = ['autoencoder']
best_kernels(tokens, 10, idd="4.2")
tokens = ['deep learning']
best_kernels(tokens, 10, idd="4.3")
tokens = ['convolutional neural networks', 'cnn']
best_kernels(tokens, 10, idd="4.4")
tokens = ['recurrent','rnn']
best_kernels(tokens, 10, idd="4.5")
tokens = ['lstm']
best_kernels(tokens, 10, idd="4.6")
tokens = ['gru']
ignore = ['grupo']
best_kernels(tokens, 10, ignore, idd="4.7")
tokens = ['mxnet']
best_kernels(tokens, 10, idd="4.8")
tokens = ['resnet']
best_kernels(tokens, 10, idd="4.9")
tokens = ['Capsule network', 'capsulenet']
best_kernels(tokens, 5, idd="4.10")
tokens = ['vgg']
best_kernels(tokens, 5, idd="4.11")
tokens = ['unet']
best_kernels(tokens, 10, idd="4.12")
tokens = ['alexnet']
best_kernels(tokens, 5, idd="4.13")
tokens = ['xception']
best_kernels(tokens, 5, idd="4.14")
tokens = ['inception']
best_kernels(tokens, 5, idd="4.15")
tokens = ['computer vision']
best_kernels(tokens, 5, idd="4.16")
tokens = ['transfer']
best_kernels(tokens, 10, idd="4.17")
tokens = ['yolo']
best_kernels(tokens, 5, idd="4.18")
tokens = ['object detection']
best_kernels(tokens, 5, idd="4.19")
tokens = ['rcnn']
best_kernels(tokens, 5, idd="4.20")
tokens = ['mobilenet']
best_kernels(tokens, 5, idd="4.21")
tokens = ['kmeans', 'k means']
best_kernels(tokens, 10, idd="5.1")
tokens = ['hierarchical clustering']
best_kernels(tokens, 3, idd="5.2")
tokens = ['dbscan']
best_kernels(tokens, 10, idd="5.3")
tokens = ['unsupervised']
best_kernels(tokens, 10, idd="5.4")
tokens = ['naive bayes']
best_kernels(tokens, 10, idd="6.1")
tokens = ['svm']
best_kernels(tokens, 10, idd="6.2")
tokens = ['knn']
best_kernels(tokens, 10, idd="6.3")
tokens = ['recommendation engine']
best_kernels(tokens, 5, idd="6.4")
tokens = ['EDA', 'exploration', 'exploratory']
best_kernels(tokens, 10, idd="7.1.a")
tokens = ['feature engineering']
best_kernels(tokens, 10, idd="7.1.b")
tokens = ['feature selection']
best_kernels(tokens, 10, idd="7.1.c")
tokens = ['outlier treatment', 'outlier']
best_kernels(tokens, 10, idd="7.1.d")
tokens = ['anomaly detection', 'anomaly']
best_kernels(tokens, 8, idd="7.1.e")
tokens = ['smote']
best_kernels(tokens, 5, idd="7.1.f")
tokens = ['pipeline']
best_kernels(tokens, 10, idd="7.1.g")
tokens = ['missing value']
best_kernels(tokens, 10, idd="7.1.h")
tokens = ['dataset decomposition', 'dimentionality reduction']
best_kernels(tokens, 2, idd="7.2.a")
tokens = ['PCA']
best_kernels(tokens, 10, idd="7.2.b")
tokens = ['Tsne', 't-sne']
best_kernels(tokens, 10, idd="7.2.c")
tokens = ['svd']
best_kernels(tokens, 10, idd="7.2.d")
tokens = ['cross validation']
best_kernels(tokens, 10, idd="7.3.a")
tokens = ['model selection']
best_kernels(tokens, 10, idd="7.3.b")
tokens = ['model tuning', 'tuning']
best_kernels(tokens, 10, idd="7.3.c")
tokens = ['gridsearch', 'grid search']
best_kernels(tokens, 10, idd="7.3.d")
tokens = ['ensemble']
best_kernels(tokens, 10, idd="7.4.a")
tokens = ['stacking', 'stack']
best_kernels(tokens, 10, idd="7.4.b")
tokens = ['bagging']
best_kernels(tokens, 10, idd="7.4.c")
tokens = ['blend']
best_kernels(tokens, 10, idd="7.4.d")
tokens = ['NLP', 'Natural Language Processing', 'text mining']
best_kernels(tokens, 10, idd="8.1")
tokens = ['topic modelling', 'lda']
best_kernels(tokens, 8, idd="8.2")
tokens = ['word embedding','fasttext', 'glove', 'word2vec', 'word vector']
best_kernels(tokens, 8, idd="8.3")
tokens = ['spacy']
best_kernels(tokens, 10, idd="8.4")
tokens = ['nltk']
best_kernels(tokens, 5, idd="8.5")
tokens = ['textblob']
best_kernels(tokens, 5, idd="8.6")
tokens = ['scikit']
best_kernels(tokens, 10, idd="9.1")
tokens = ['tensorflow', 'tensor flow']
best_kernels(tokens, 10, idd="9.2")
tokens = ['theano']
best_kernels(tokens, 10, idd="9.3")
tokens = ['keras']
best_kernels(tokens, 10, idd="9.4")
tokens = ['pytorch']
best_kernels(tokens, 10, idd="9.5")
tokens = ['vowpal wabbit','vowpalwabbit']
best_kernels(tokens, 10, idd="9.6")
tokens = ['eli5']
best_kernels(tokens, 10, idd="9.7")
tokens = ['hyperopt']
best_kernels(tokens, 5, idd="9.8")
tokens = ['pandas']
best_kernels(tokens, 10, idd="9.9")
tokens = ['SQL']
best_kernels(tokens, 10, idd="9.10")
tokens = ['bigquery', 'big query']
best_kernels(tokens, 10, idd="9.11")
tokens = ['gpu']
best_kernels(tokens, 10, idd="9.12")
tokens = ['h20']
best_kernels(tokens, 5, idd="9.13")
tokens = ['fastai', 'fast.ai']
best_kernels(tokens, 10, idd="9.14")
tokens = ['visualization', 'visualisation']
best_kernels(tokens, 10, idd="10.1")
tokens = ['plotly', 'plot.ly']
best_kernels(tokens, 10, idd="10.2")
tokens = ['seaborn']
best_kernels(tokens, 10, idd="10.3")
tokens = ['d3.js']
best_kernels(tokens, 4, idd="10.4")
tokens = ['bokeh']
best_kernels(tokens, 10, idd="10.5")
tokens = ['highchart']
best_kernels(tokens, 10, idd="10.6")
tokens = ['folium']
best_kernels(tokens, 5, idd="10.7")
tokens = ['ggplot']
best_kernels(tokens, 10, idd="10.8")
tokens = ['time series']
best_kernels(tokens, 10, idd="11.1")
tokens = ['arima']
best_kernels(tokens, 10, idd="11.2")
tokens = ['forecasting']
best_kernels(tokens, 10, idd="11.3")
tokens = ['tutorial']
best_kernels(tokens, 10, idd="12.1")
tokens = ['data leak', 'leak']
best_kernels(tokens, 10, idd="12.2")
tokens = ["adversarial validation"]
best_kernels(tokens, 10, idd="12.3")
tokens = ["generative adversarial networks", "simgan", "-gan"]
best_kernels(tokens, 10, idd="12.4")