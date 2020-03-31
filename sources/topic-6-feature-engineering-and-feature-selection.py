
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Let's load the dataset from Renthop right away
with open('../input/two-sigma-connect-rental-listing-inquiries/train.json', 'r') as raw_data:
    data = json.load(raw_data)
    df = pd.DataFrame(data)
from functools import reduce 
import numpy as np

texts = [['i', 'have', 'a', 'cat'], 
        ['he', 'have', 'a', 'dog'], 
        ['he', 'and', 'i', 'have', 'a', 'cat', 'and', 'a', 'dog']]

dictionary = list(enumerate(set(list(reduce(lambda x, y: x + y, texts)))))

def vectorize(text): 
    vector = np.zeros(len(dictionary)) 
    for i, word in dictionary: 
        num = 0 
        for w in text: 
            if w == word: 
                num += 1 
        if num: 
            vector[i] = num 
    return vector

for t in texts: 
    print(vectorize(t))
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(1,1))
vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
vect.vocabulary_ 
vect = CountVectorizer(ngram_range=(1,2))
vect.fit_transform(['no i have cows', 'i have no cows']).toarray()
vect.vocabulary_
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer(ngram_range=(3,3), analyzer='char_wb')

n1, n2, n3, n4 = vect.fit_transform(['andersen', 'petersen', 'petrov', 'smith']).toarray()

euclidean(n1, n2), euclidean(n2, n3), euclidean(n3, n4)
# Install Keras (https://keras.io/)
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image 
from scipy.misc import face 
import numpy as np

resnet_settings = {'include_top': False, 'weights': 'imagenet'}
resnet = ResNet50(**resnet_settings)

# What a cute raccoon!
img = image.array_to_img(face())
img
# In real life, you may need to pay more attention to resizing
img = img.resize((224, 224))

x = image.img_to_array(img) 
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

# Need an extra dimension because model is designed to work with an array
# of images - i.e. tensor shaped (batch_size, width, height, n_channels)

features = resnet.predict(x)
import pytesseract
from PIL import Image
import requests
from io import BytesIO

##### Just a random picture from search
img_url = 'http://ohscurrent.org/wp-content/uploads/2015/09/domus-01-google.jpg'
img = requests.get(img_url)
img = Image.open(BytesIO(img.content))
text = pytesseract.image_to_string(img)

text
##### This time we take a picture from Renthop
img = requests.get('https://photos.renthop.com/2/8393298_6acaf11f030217d05f3a5604b9a2f70f.jpg')
img = Image.open(BytesIO(img.content))
pytesseract.image_to_string(img)
### pip install -q reverse_geocoder
import reverse_geocoder as revgc

revgc.search([df.latitude[0], df.longitude[0]])
df['dow'] = df['created'].apply(lambda x: pd.to_datetime(x).weekday())
df['is_weekend'] = df['created'].apply(lambda x: 1 if pd.to_datetime(x).weekday() in (5, 6) else 0)
def make_harmonic_features(value, period=24):
    value *= 2 * np.pi / period 
    return np.cos(value), np.sin(value)
from scipy.spatial import distance
euclidean(make_harmonic_features(23), make_harmonic_features(1)) 
euclidean(make_harmonic_features(9), make_harmonic_features(11)) 
euclidean(make_harmonic_features(9), make_harmonic_features(21))
### pip install -q pyyaml ua-parser user-agents
import user_agents

ua = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/56.0.2924.76 Chrome/56.0.2924.76 Safari/537.36'
ua = user_agents.parse(ua)

print('Is a bot? ', ua.is_bot)
print('Is mobile? ', ua.is_mobile)
print('Is PC? ',ua.is_pc)
print('OS Family: ',ua.os.family)
print('OS Version: ',ua.os.version)
print('Browser Family: ',ua.browser.family)
print('Browser Version: ',ua.browser.version)
from sklearn.preprocessing import StandardScaler
from scipy.stats import beta
from scipy.stats import shapiro
import numpy as np

data = beta(1, 10).rvs(1000).reshape(-1, 1)
shapiro(data)
# Value of the statistic, p-value
shapiro(StandardScaler().fit_transform(data))

# With such p-value we'd have to reject the null hypothesis of normality of the data
data = np.array([1, 1, 0, -1, 2, 1, 2, 3, -2, 4, 100]).reshape(-1, 1).astype(np.float64)
StandardScaler().fit_transform(data)
(data - data.mean()) / data.std()
from sklearn.preprocessing import MinMaxScaler

MinMaxScaler().fit_transform(data)
(data - data.min()) / (data.max() - data.min()) 
from scipy.stats import lognorm

data = lognorm(s=1).rvs(1000)
shapiro(data)
shapiro(np.log(data))
# Let's draw plots!
import statsmodels.api as sm

# Let's take the price feature from Renthop dataset and filter by hands the most extreme values for clarity

price = df.price[(df.price <= 20000) & (df.price > 500)]
price_log = np.log(price)

# A lot of gestures so that sklearn didn't shower us with warnings
price_mm = MinMaxScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
price_z = StandardScaler().fit_transform(price.values.reshape(-1, 1).astype(np.float64)).flatten()
sm.qqplot(price, loc=price.mean(), scale=price.std())
sm.qqplot(price_z, loc=price_z.mean(), scale=price_z.std())
sm.qqplot(price_mm, loc=price_mm.mean(), scale=price_mm.std())
sm.qqplot(price_log, loc=price_log.mean(), scale=price_log.std())
rooms = df["bedrooms"].apply(lambda x: max(x, .5))
# Avoid division by zero; .5 is chosen more or less arbitrarily
df["price_per_bedroom"] = df["price"] / rooms
from sklearn.feature_selection import VarianceThreshold
from sklearn.datasets import make_classification

x_data_generated, y_data_generated = make_classification()
x_data_generated.shape
VarianceThreshold(.7).fit_transform(x_data_generated).shape
VarianceThreshold(.8).fit_transform(x_data_generated).shape
VarianceThreshold(.9).fit_transform(x_data_generated).shape
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

x_data_kbest = SelectKBest(f_classif, k=5).fit_transform(x_data_generated, y_data_generated)
x_data_varth = VarianceThreshold(.9).fit_transform(x_data_generated)
cross_val_score(LogisticRegression(), x_data_generated, y_data_generated, scoring='neg_log_loss').mean()
cross_val_score(LogisticRegression(), x_data_kbest, y_data_generated, scoring='neg_log_loss').mean()
cross_val_score(LogisticRegression(), x_data_varth, y_data_generated, scoring='neg_log_loss').mean()
# Synthetic example

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import make_pipeline

x_data_generated, y_data_generated = make_classification()

pipe = make_pipeline(SelectFromModel(estimator=RandomForestClassifier()), LogisticRegression())

lr = LogisticRegression()
rf = RandomForestClassifier()

print(cross_val_score(lr, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(rf, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
print(cross_val_score(pipe, x_data_generated, y_data_generated, scoring='neg_log_loss').mean())
#x_data, y_data = get_data() 
x_data = x_data_generated
y_data = y_data_generated

pipe1 = make_pipeline(StandardScaler(), SelectFromModel(estimator=RandomForestClassifier()), LogisticRegression())

pipe2 = make_pipeline(StandardScaler(), LogisticRegression())

rf = RandomForestClassifier()

print('LR + selection: ', cross_val_score(pipe1, x_data, y_data, scoring='neg_log_loss').mean())
print('LR: ', cross_val_score(pipe2, x_data, y_data, scoring='neg_log_loss').mean())
print('RF: ', cross_val_score(rf, x_data, y_data, scoring='neg_log_loss').mean())
# Install mlxtend
from mlxtend.feature_selection import SequentialFeatureSelector

selector = SequentialFeatureSelector(LogisticRegression(), scoring='neg_log_loss', 
                                     verbose=2, k_features=3, forward=False, n_jobs=-1)

selector.fit(x_data, y_data)