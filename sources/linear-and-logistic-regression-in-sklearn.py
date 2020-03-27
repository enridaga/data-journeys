
!pip install sklearn
import matplotlib.pyplot as plt
import numpy as np

# generate training data
# generate 2000 training samples from gaussian distribution as a 2000x1 vector
X_train = np.random.randn(2000, 1)
# generate gaussian noise for each samples (noise same shape with input)
noise = np.random.randn(2000, 1)
# create y = 4 + 3X and add noise
Y_train = 4+3*X_train+noise

# generate test data
# generate 10 test samples from gaussian distribution as a 10x1 vector
X_test = np.random.randn(10, 1)
# generate gaussian noise for each samples (noise same shape with input)
noise = np.random.randn(10, 1)
# create y = 4 + 3X and add noise
Y_test = 4+3*X_test+noise
# visualize data
# plot training samples as points ("o"), index 0 to get scalar values, transparency 0.2
plt.plot(X_train[:, 0], Y_train[:, 0], "o", alpha=0.2, label="train")
# plot test samples as points ("o")
plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")
# set plot title 
plt.title("Data points")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top right
plt.legend()
# show plot
plt.show()
from sklearn import linear_model

# create LinearRegression model
regressor = linear_model.LinearRegression()

# train the regression model using fit()
regressor.fit(X_train, Y_train)
# visualize the trained results
# learned regressor line will have W as .coef_ and b as .intercept_ of the regressor
print(f"Learned parameters- W: {regressor.coef_} - b:{regressor.intercept_}")
# plot training samples as points ("o"), index 0 to get scalar values, transparency 0.2
plt.plot(X_train[:, 0], Y_train[:, 0], "o", alpha=0.2, label="train")
# plot test samples as points ("o")
plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")
# plot regression model as a line (X as x axis and projection on regressor line as y axis)
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, label="regression model")
# set plot title
plt.title("Data points")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()
# use the function predict() of regressor to predict with X_test
Y_preds = regressor.predict(X_test)
# sklearn LinearRegressor() provide score() function to estimate loss of learned regression model
print(f"Test score: {regressor.score(X_test, Y_test)}")
# visualize the trained results
# plot predicted samples as points ("o")
plt.plot(X_test[:, 0], Y_preds[:, 0], "o", label="prediction")
# plot generated test samples as points ("o")
plt.plot(X_test[:, 0], Y_test[:, 0], "o", label="test")
# plot regression model as a line (X as x axis and projection on regressor line as y axis), transparency 0.2
plt.plot(X_train, regressor.coef_ * X_train + regressor.intercept_, label="regression model", alpha=0.2)
# set plot title
plt.title("Data points")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()
cov = [[1, 0], [0, 1]]
mean0 = [2, 2]
mean1 = [5, 7]
N = 1000

# generate train set
# class 0
# generate class 0 samples x axis and y axis values from gaussian distribution
# numpy provides multivariate_normal() function to generate multiple values at once
X0_train = np.random.multivariate_normal(mean0, cov, N)
# class 0 have labels 0
Y0_train = np.zeros(N)
# class 1
# generate class 1 samples x axis and y axis values from gaussian distribution
X1_train = np.random.multivariate_normal(mean1, cov, N)
# class 1 have labels 1
Y1_train = np.ones(N)
# concatenate the training data for each class to create training set with both class.
X_train = np.concatenate([X0_train, X1_train], axis=0)
Y_train = np.concatenate([Y0_train, Y1_train], axis=0)
# generate test set
M = 10

X0_test = np.random.multivariate_normal(mean0, cov, M)
Y0_test = np.zeros(M)
# class 1
X1_test = np.random.multivariate_normal(mean1, cov, M)
Y1_test = np.ones(M)
# concatenate the training data for each class to create test set with both class.
X_test = np.concatenate([X0_test, X1_test], axis=0)
Y_test = np.concatenate([Y0_test, Y1_test], axis=0)
# visualize generated numbers
# plot class 0 samples as red points ("ro"), transparency = 0.2
plt.plot(X0_train[:, 0], X0_train[:, 1], "ro", alpha=0.2, label="train 0")
# plot class 0 samples as blue points ("bo"), transparency = 0.2
plt.plot(X1_train[:, 0], X1_train[:, 1], "bo", alpha=0.2, label="train 1")
# plot test samples as black point ("o", color="black")
plt.plot(X_test[:, 0], X_test[:, 1], "o", color="black", label="test")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()
from sklearn import linear_model

# create LogisticRegression model
classifier = linear_model.LogisticRegression()

# train the regression model with fit() function
classifier.fit(X_train, Y_train)
# visualize the trained results
# learned regressor line will have W as .coef_ and b as .intercept_ of the regressor
print(f"Learned parameters- W: {classifier.coef_} - b:{classifier.intercept_}")
# plot class 0 samples as red points ("ro"), transparency = 0.2
plt.plot(X0_train[:, 0], X0_train[:, 1], "ro", alpha=0.2, label="train 0")
# plot class 0 samples as blue points ("bo"), transparency = 0.2
plt.plot(X1_train[:, 0], X1_train[:, 1], "bo", alpha=0.2, label="train 1")
# plot test samples as black point ("o", color="black")
plt.plot(X_test[:, 0], X_test[:, 1], "o", color="black", label="test")
# plot classification boundary line
plt.plot(X_train[:, 0], (0-classifier.intercept_ - classifier.coef_[0, 0]*X_train[:, 0])/classifier.coef_[0, 1], label="classification boundary")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()
# create LogisticRegression model
Y_preds = classifier.predict(X_test)
print(f"Test score: {classifier.score(X_test, Y_test)}")
# visualize the trained results
# get test samples from each class
X_test_class0 = X_test[Y_preds == 0]
X_test_class1 = X_test[Y_preds == 1]
# plot class 0 test samples as red points ("ro")
plt.plot(X_test_class0[:, 0], X_test_class0[:, 1], "ro", label="test 0")
# plot class 1 test samples as blue points ("ro")
plt.plot(X_test_class1[:, 0], X_test_class1[:, 1], "bo", label="test 1")
# plot decision boundary
plt.plot(X_train[:, 0], (0-classifier.intercept_ - classifier.coef_[0, 0]*X_train[:, 0])/classifier.coef_[0, 1], label="classification boundary")
# set axis name
plt.xlabel("X")
plt.ylabel("Y")
# create legend on top left
plt.legend()
# show plot
plt.show()
