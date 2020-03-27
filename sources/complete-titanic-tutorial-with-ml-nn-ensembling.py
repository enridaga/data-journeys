
import pydot
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

# Ignore warnings
import warnings
warnings.filterwarnings('ignore')

import sklearn
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import export_graphviz, DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.exceptions import NotFittedError

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.wrappers.scikit_learn import KerasClassifier

from IPython.display import display
# Some useful functions we'll use in this notebook
def display_confusion_matrix(target, prediction, score=None):
    cm = metrics.confusion_matrix(target, prediction)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square=True, cmap='Blues_r')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    if score:
        score_title = 'Accuracy Score: {0}'.format(round(score, 5))
        plt.title(score_title, size = 14)
    classification_report = pd.DataFrame.from_dict(metrics.classification_report(target, prediction, output_dict=True), orient='index')
    display(classification_report.round(2))
    
def visualize_tree(tree, feature_names):
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f, feature_names=feature_names)
    try:
        subprocess.check_call(["dot", "-Tpng", "dt.dot", "-o", "dt.png"])
    except:
        exit("Could not run dot, ie graphviz, to produce visualization")
        
def draw_missing_data_table(df):
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    return missing_data
# Path of datasets
path_train = '../input/train.csv'
path_test = '../input/test.csv'
# Create dataframe for training dataset and print five first rows as preview
train_df_raw = pd.read_csv(path_train)
train_df_raw.head()
# Compute some basical statistics on the dataset
train_df_raw.describe()
train_df_raw.info()
draw_missing_data_table(train_df_raw)
figure, axes = plt.subplots(1,1,figsize=(20, 6))
plot = sns.catplot(x="Embarked", y="Fare", hue="Sex", data=train_df_raw, palette=('Set1'), kind="bar", ax=axes)
plt.close(plot.fig)
plt.show()
display(train_df_raw[train_df_raw['Embarked'].isnull()])
# Let's plot some histograms to have a previzualisation of some of the data ...
train_df_raw.drop(['PassengerId'], 1).hist(bins=50, figsize=(20,15))
plt.show()
def preprocess_data(df):
    
    processed_df = df
        
    ########## Deal with missing values ##########
    
    # As we saw before, the two missing values for embarked columns can be replaced by 'C' (Cherbourg)
    processed_df['Embarked'].fillna('C', inplace=True)
    
    # We replace missing ages by the mean age of passengers who belong to the same group of class/sex/family
    processed_df['Age'] = processed_df.groupby(['Pclass','Sex','Parch','SibSp'])['Age'].transform(lambda x: x.fillna(x.mean()))
    processed_df['Age'] = processed_df.groupby(['Pclass','Sex','Parch'])['Age'].transform(lambda x: x.fillna(x.mean()))
    processed_df['Age'] = processed_df.groupby(['Pclass','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
    
    # We replace the only missing fare value for test dataset and the missing values of the cabin column
    processed_df['Fare'] = processed_df['Fare'].interpolate()
    processed_df['Cabin'].fillna('U', inplace=True)
    
    ########## Feature engineering on columns ##########
    
    # Create a Title column from name column
    processed_df['Title'] = pd.Series((name.split('.')[0].split(',')[1].strip() for name in train_df_raw['Name']), index=train_df_raw.index)
    processed_df['Title'] = processed_df['Title'].replace(['Lady', 'the Countess','Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    processed_df['Title'] = processed_df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    processed_df['Title'] = processed_df['Title'].replace('Mme', 'Mrs')
    processed_df['Title'] = processed_df['Title'].map({"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5})
    
    # Filling Age missing values with mean age of passengers who have the same title
    processed_df['Age'] = processed_df.groupby(['Title'])['Age'].transform(lambda x: x.fillna(x.mean()))

    # Transform categorical variables to numeric variables
    processed_df['Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})
    processed_df['Embarked'] = processed_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
    
    # Create a Family Size, Is Alone, Child and Mother columns
    processed_df['FamillySize'] = processed_df['SibSp'] + processed_df['Parch'] + 1
    processed_df['FamillySize'][processed_df['FamillySize'].between(1, 5, inclusive=False)] = 2
    processed_df['FamillySize'][processed_df['FamillySize']>5] = 3
    processed_df['IsAlone'] = np.where(processed_df['FamillySize']!=1, 0, 1)
    processed_df['IsChild'] = processed_df['Age'] < 18
    processed_df['IsChild'] = processed_df['IsChild'].astype(int)
    
    # Modification of cabin column to keep only the letter contained corresponding to the deck of the boat
    processed_df['Cabin'] = processed_df['Cabin'].str[:1]
    processed_df['Cabin'] = processed_df['Cabin'].map({cabin: p for p, cabin in enumerate(set(cab for cab in processed_df['Cabin']))})
    
    # Create a ticket survivor column which is set to 1 if an other passenger with the same ticket survived and 0 else
    # Note : this implementation is ugly and unefficient, if sombody found a way to do it easily with pandas (it must be a way), please comment the kernel with your solution !
    processed_df['TicketSurvivor'] = pd.Series(0, index=processed_df.index)
    tickets = processed_df['Ticket'].value_counts().to_dict()
    for t, occ in tickets.items():
        if occ != 1:
            table = train_df_raw['Survived'][train_df_raw['Ticket'] == t]
            if sum(table) != 0:
                processed_df['TicketSurvivor'][processed_df['Ticket'] == t] = 1
    
    # These two columns are not useful anymore
    processed_df = processed_df.drop(['Name', 'Ticket', 'PassengerId'], 1)    
    
    return processed_df
# Let's divide the train dataset in two datasets to evaluate perfomance of the machine learning models we'll use
train_df = train_df_raw.copy()
X = train_df.drop(['Survived'], 1)
Y = train_df['Survived']

X = preprocess_data(X)
# We scale our data, it is essential for a smooth working of the models. Scaling means that each columns as a 0 mean and a 1 variance
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X.values), index=X.index, columns=X.columns)
    
# Split dataset for model testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

X_train.head()
# Create and train model on train data sample
lg = LogisticRegression(solver='lbfgs', random_state=42)
lg.fit(X_train, Y_train)

# Predict for test data sample
logistic_prediction = lg.predict(X_test)

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_test, logistic_prediction)
display_confusion_matrix(Y_test, logistic_prediction, score=score)
dt = DecisionTreeClassifier(min_samples_split=15, min_samples_leaf=20, random_state=42)
dt.fit(X_train, Y_train)
dt_prediction = dt.predict(X_test)

score = metrics.accuracy_score(Y_test, dt_prediction)
display_confusion_matrix(Y_test, dt_prediction, score=score)
visualize_tree(dt, X_test.columns)
! dot -Tpng dt.dot > dt.png
svm = SVC(gamma='auto', random_state=42)
svm.fit(X_train, Y_train)
svm_prediction = svm.predict(X_test)

score = metrics.accuracy_score(Y_test, svm_prediction)
display_confusion_matrix(Y_test, svm_prediction, score=score)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, Y_train)
rf_prediction = rf.predict(X_test)

score = metrics.accuracy_score(Y_test, rf_prediction)
display_confusion_matrix(Y_test, rf_prediction, score=score)
def build_ann(optimizer='adam'):
    
    # Initializing our ANN
    ann = Sequential()
    
    # Adding the input layer and the first hidden layer of our ANN with dropout
    ann.add(Dense(units=32, kernel_initializer='glorot_uniform', activation='relu', input_shape=(13,)))
    
    # Add other layers, it is not necessary to pass the shape because there is a layer before
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))
    ann.add(Dropout(rate=0.5))
    ann.add(Dense(units=64, kernel_initializer='glorot_uniform', activation='relu'))
    ann.add(Dropout(rate=0.5))
    
    # Adding the output layer
    ann.add(Dense(units=1, kernel_initializer='glorot_uniform', activation='sigmoid'))
    
    # Compiling the ANN
    ann.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    
    return ann
opt = optimizers.Adam(lr=0.001)
ann = build_ann(opt)
# Training the ANN
history = ann.fit(X_train, Y_train, batch_size=16, epochs=30, validation_data=(X_test, Y_test))
# Predicting the Test set results
ann_prediction = ann.predict(X_test)
ann_prediction = (ann_prediction > 0.5) # convert probabilities to binary output

# Compute error between predicted data and true response and display it in confusion matrix
score = metrics.accuracy_score(Y_test, ann_prediction)
display_confusion_matrix(Y_test, ann_prediction, score=score)
n_folds = 10
cv_score_lg = cross_val_score(estimator=lg, X=X_train, y=Y_train, cv=n_folds, n_jobs=-1)
cv_score_dt = cross_val_score(estimator=dt, X=X_train, y=Y_train, cv=n_folds, n_jobs=-1)
cv_score_svm = cross_val_score(estimator=svm, X=X_train, y=Y_train, cv=n_folds, n_jobs=-1)
cv_score_rf = cross_val_score(estimator=rf, X=X_train, y=Y_train, cv=n_folds, n_jobs=-1)
cv_score_ann = cross_val_score(estimator=KerasClassifier(build_fn=build_ann, batch_size=16, epochs=20, verbose=0),
                                 X=X_train, y=Y_train, cv=n_folds, n_jobs=-1)
cv_result = {'lg': cv_score_lg, 'dt': cv_score_dt, 'svm': cv_score_svm, 'rf': cv_score_rf, 'ann': cv_score_ann}
cv_data = {model: [score.mean(), score.std()] for model, score in cv_result.items()}
cv_df = pd.DataFrame(cv_data, index=['Mean_accuracy', 'Variance'])
cv_df
plt.figure(figsize=(20,8))
plt.plot(cv_result['lg'])
plt.plot(cv_result['dt'])
plt.plot(cv_result['svm'])
plt.plot(cv_result['rf'])
plt.plot(cv_result['ann'])
plt.title('Models Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Trained fold')
plt.xticks([k for k in range(n_folds)])
plt.legend(['logreg', 'tree', 'randomforest', 'ann', 'svm'], loc='upper left')
plt.show()
class EsemblingClassifier:
    
    def __init__(self, verbose=True):
        self.ann = build_ann(optimizer=optimizers.Adam(lr=0.001))
        self.rf = RandomForestClassifier(n_estimators=300, max_depth=11, random_state=42)
        self.svm = SVC(random_state=42)
        self.trained = False
        self.verbose = verbose
        
    def fit(self, X, y):
        if self.verbose:
            print('-------- Fitting models --------')
        self.ann.fit(X, y, epochs=30, batch_size=16, verbose=0)
        self.rf.fit(X, y)
        self.svm.fit(X, y)
        self.trained = True
    
    def predict(self, X):
        if self.trained == False:
            raise NotFittedError('Please train the classifier before making a prediction')
        if self.verbose:
            print('-------- Making and combining predictions --------')
        predictions = list()
        pred_ann = self.ann.predict(X)
        pred_ann = (pred_ann > 0.5)*1
        pred_rf = self.rf.predict(X)
        pred_svm = self.svm.predict(X)
        for n in range(len(pred_ann)):
            combined = pred_ann[n] + pred_rf[n] + pred_svm[n]
            p = 0 if combined == 1 or combined == 0 else 1
            predictions.append(p)
        return predictions
ens = EsemblingClassifier()
ens.fit(X_train, Y_train)
ens_prediction = ens.predict(X_test)
score = metrics.accuracy_score(Y_test, ens_prediction)
display_confusion_matrix(Y_test, ens_prediction, score=score)
test_df_raw = pd.read_csv(path_test)
test = test_df_raw.copy()
test = preprocess_data(test)
test = pd.DataFrame(sc.fit_transform(test.values), index=test.index, columns=test.columns)
test.head()
# Create and train model on train data sample
model_test = EsemblingClassifier()
model_test.fit(X, Y)

# Predict for test data sample
prediction = model_test.predict(test)

result_df = test_df_raw.copy()
result_df['Survived'] = prediction
result_df.to_csv('submission.csv', columns=['PassengerId', 'Survived'], index=False)
result_df[['PassengerId', 'Survived']].head(10)
import scipy.stats as stats

plt.figure(figsize=(20, 8))
sub = 2*[0.77511] + 3*[0.77990] + 10*[0.78468] + 11*[0.78947] + 2*[0.79425] + 2*[0.79904]
sns.distplot(sub)
plt.show()