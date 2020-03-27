
import matplotlib.patches as patch
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import linalg
from numpy import poly1d
from sklearn import svm
import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import sys
import os
#%matplotlib inline
%precision 4
plt.style.use('ggplot')
np.set_printoptions(suppress=True)
# let see how to create a multi dimentional Array with Numpy
a = np.zeros((2, 3, 4))
#l = [[[ 0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.],
     #     [ 0.,  0.,  0.,  0.]],
     #     [[ 0.,  0.,  0.,  0.],
    #      [ 0.,  0.,  0.,  0.],
     #     [ 0.,  0.,  0.,  0.]]]
print(a)
print(a.shape)

# Declaring Vectors

x = [1, 2, 3]
y = [4, 5, 6]

print(type(x))

# This does'nt give the vector addition.
print(x + y)

# Vector addition using Numpy

z = np.add(x, y)
print(z)
print(type(z))

# Vector Cross Product
mul = np.cross(x, y)
print(mul)
# initializing matrices 
x = np.array([[1, 2], [4, 5]]) 
y = np.array([[7, 8], [9, 10]])
# using add() to add matrices 
print ("The element wise addition of matrix is : ") 
print (np.add(x,y)) 
# using subtract() to subtract matrices 
print ("The element wise subtraction of matrix is : ") 
print (np.subtract(x,y)) 
# using divide() to divide matrices 
print ("The element wise division of matrix is : ") 
print (np.divide(x,y)) 
# using multiply() to multiply matrices element wise 
print ("The element wise multiplication of matrix is : ") 
print (np.multiply(x,y))
x = [1, 2, 3]
y = [4, 5, 6]
np.cross(x, y)
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
print("x:", x)
print("y:", y)
np.dot(x, y)
np.dot(y, x)
print("x:", x)
x.shape = (4, 1)
print("xT:", x)
print("y:", y)
y.shape = (4, 1)
print("yT:", y)
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
print("x:", x)
print("y:", y)
print("xT:", x.T)
print("yT:", y.T)
x = np.array([[1, 2, 3, 4]])
y = np.array([[5, 6, 7, 8]])
print("x:", x)
print("y:", y)
print("xT:", x.T)
print("yT:", y.T)

print("x:", x)
print("y:", y.T)
np.dot(x, y.T)
print("x:", x.T)
print("y:", y)
np.dot(y, x.T)
np.dot(y, x.T)[0][0]
x = np.array([[1, 2, 3, 4]])
print("x:", x)
print("xT:", np.reshape(x, (4, 1)))
print("xT:", x.T)
print("xT:", x.transpose())
x = np.array([[1, 2, 3, 4]])
y = np.array([[5, 6, 7, 8]])
x.T * y
np.outer(x, y)
x = np.array([1, 2, 3, 4])
y = np.array([5, 6, 7, 8])
np.outer(x, y)
a = np.array([[ 5, 1 ,3], [ 1, 1 ,1], [ 1, 2 ,1]])
b = np.array([1, 2, 3])
print (a.dot(b))
A = np.array([[4, 5, 6],
             [7, 8, 9]])
x = np.array([1, 2, 3])
A.dot(x)
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.matmul(a, b)
matrix1 = np.matrix(a)
matrix2 = np.matrix(b)
matrix1 + matrix2
matrix1 - matrix2
np.dot(matrix1, matrix2)

matrix1 * matrix2
np.identity(3)
identy = np.array([[21, 5, 7],[9, 8, 16]])
print("identy:", identy)
identy.shape
np.identity(identy.shape[1], dtype="int")
np.identity(identy.shape[0], dtype="int")
inverse = np.linalg.inv(matrix1)
print(inverse)
import numpy as np
A = np.array([[0,   1,  2,  3],
              [4,   5,  6,  7],
              [8,   9, 10, 11],
              [12, 13, 14, 15]])
np.diag(A)
np.diag(A, k=1)
np.diag(A, k=-1)
a = np.array([[1, 2], [3, 4]])
a
a.transpose()
N = 100
b = np.random.random_integers(-2000,2000,size=(N,N))
b_symm = (b + b.T)/2
np.trace(np.eye(3))
print(np.trace(matrix1))
det = np.linalg.det(matrix1)
print(det)
v = np.array([1,2,3,4])
norm.median(v)
#How to find linearly independent rows from a matrix
matrix = np.array(
    [
        [0, 1 ,0 ,0],
        [0, 0, 1, 0],
        [0, 1, 1, 0],
        [1, 0, 0, 1]
    ])

lambdas, V =  np.linalg.eig(matrix.T)
# The linearly dependent row vectors 
print (matrix[lambdas == 0,:])
import numpy as np
print("np.arange(9):", np.arange(9))
print("np.arange(9, 18):", np.arange(9, 18))
A = np.arange(9, 18).reshape((3, 3))
B = np.arange(9).reshape((3, 3))
print("A:", A)
print("B:", B)
A + B
A - B
x = np.array([[1,2],[3,4]]) 
y = np.linalg.inv(x) 
print (x )
print (y )
print (np.dot(x,y))
## based on https://stackoverflow.com/questions/38426349/how-to-create-random-orthonormal-matrix-in-python-numpy
def rvs(dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
from scipy.linalg import null_space
A = np.array([[1, 1], [1, 1]])
ns = null_space(A)
ns * np.sign(ns[0,0])  # Remove the sign ambiguity of the vector
a = np.array([[1, 2], [3, 4]])
np.linalg.det(a)
# credits: https://www.tensorflow.org/api_docs/python/tf/Variable
A = tf.Variable(np.zeros((5, 5), dtype=np.float32), trainable=False)
new_part = tf.ones((2,3))
update_A = A[2:4,2:5].assign(new_part)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
print(update_A.eval())
##based on this address: https://stackoverflow.com/questions/46511017/plot-hyperplane-linear-svm-python
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2], np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20

fig, ax = plt.subplots()
clf2 = svm.LinearSVC(C=1).fit(X, Y)

# get the separating hyperplane
w = clf2.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf2.intercept_[0]) / w[1]

# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx2, yy2 = np.meshgrid(np.arange(x_min, x_max, .2),
                     np.arange(y_min, y_max, .2))
Z = clf2.predict(np.c_[xx2.ravel(), yy2.ravel()])

Z = Z.reshape(xx2.shape)
ax.contourf(xx2, yy2, Z, cmap=plt.cm.coolwarm, alpha=0.3)
ax.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.coolwarm, s=25)
ax.plot(xx,yy)

ax.axis([x_min, x_max,y_min, y_max])
plt.show()
np.mgrid[0:5,0:5]
a=np.array([1,2,3])
b=np.array([(1+5j,2j,3j), (4j,5j,6j)])
c=np.array([[(1.5,2,3), (4,5,6)], [(3,2,1), (4,5,6)]])
np.transpose(b)
b.flatten()
np.hsplit(c,2)
p=poly1d([3,4,5])
p