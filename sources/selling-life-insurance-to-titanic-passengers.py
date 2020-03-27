
# Silencing warnings from scikit-learn
import warnings
warnings.filterwarnings("ignore")

# Importing libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import model_selection, linear_model, metrics
# Importing data
data = pd.read_csv('../input/train.csv')
data.head()
# Selecting the target column and the feature matrix
y = data['Survived']
X = data.drop(['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
X.head()
# Checking the types of the columns and if they contain missing values
X.info()
# Separating the feature matrix according to numeric and categorical features
num_feat = X.select_dtypes('number').columns.values
cat_feat = X.select_dtypes('object').columns.values
X_num = X[num_feat]
X_cat = X[cat_feat]
# Scaling the numeric features using standardization
X_num = (X_num - X_num.mean()) / X_num.std()
X_num = X_num.fillna(X_num.mean())
X_num.head()
# Encoding the categorical features using one-hot encoding
X_cat = pd.get_dummies(X_cat)
X_cat.head()
# Combining processed numeric and categorical features in one feature matrix
X = pd.concat([X_num, X_cat], axis=1)
X.head()
# Splitting the final dataset into training and testing datasets
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=0)
# Fitting the model and printing the accuracy score
model = linear_model.SGDClassifier(loss='log', max_iter=2000, random_state=0) # Random state is fixed for reproducibility
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
metrics.accuracy_score(y_test, y_pred)
# Checking the convergence of the model
n_iter = np.linspace(1, 3000)
scores = np.array([])
for n in n_iter:
    model = linear_model.SGDClassifier(loss='log', max_iter=n, random_state=0)
    model.fit(X_train, y_train)
    scores = np.append(scores, model.score(X_test, y_test))
    
plt.plot(n_iter, scores)
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.show()
# Printing the confusion matrix
metrics.confusion_matrix(y_test, y_pred)
# Printing the precision score for survived passengers
metrics.precision_score(y_test, y_pred)
# Plotting the Precision-Recall curve
y_proba_train = model.predict_proba(X_train)[:, 1]
p, r, t = metrics.precision_recall_curve(y_train, y_proba_train)

plt.plot(r, p)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.show()
# Printing the last 5 combinations of precision-recall-threshold
prt = np.array(list(zip(p, r, t)))
prt_df = pd.DataFrame(data=prt, columns=['Precision', 'Recall', 'Threshold'])
prt_df.tail()
# Counting the number of predicted survivors
y_proba_test = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba_test >= 0.973211).astype(int)
np.count_nonzero(y_pred)
# Printing the last 50 combinations of precision-recall-threshold
prt_df.tail(50)
# Counting the number of predicted survivors using the new threshold
y_proba_test = model.predict_proba(X_test)[:, 1]
y_pred = (y_proba_test >= 0.932465).astype(int)
np.count_nonzero(y_pred)
# Pairs of target labels and predicted probabilities above the new threshold
yy = np.array(list(zip(y_test, y_proba_test)))
indices = np.where(yy[:,1] >= 0.932465)
yy[indices]