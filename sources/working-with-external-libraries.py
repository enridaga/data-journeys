
import math



print("It's math! It has type {}".format(type(math)))
print(dir(math))
print("pi to 4 significant digits = {:.4}".format(math.pi))
math.log(32, 2)
help(math.log)
help(math)
import math as mt

mt.pi
import math

mt = math
from math import *

print(pi, log(32, 2))
from math import *

from numpy import *

print(pi, log(32, 2))
from math import log, pi

from numpy import asarray
import numpy

print("numpy.random is a", type(numpy.random))

print("it contains names such as...",

      dir(numpy.random)[-15:]

     )
# Roll 10 dice

rolls = numpy.random.randint(low=1, high=6, size=10)

rolls
type(rolls)
print(dir(rolls))
# What am I trying to do with this dice roll data? Maybe I want the average roll, in which case the "mean"

# method looks promising...

rolls.mean()
# Or maybe I just want to get back on familiar ground, in which case I might want to check out "tolist"

rolls.tolist()
# That "ravel" attribute sounds interesting. I'm a big classical music fan.

help(rolls.ravel)
# Okay, just tell me everything there is to know about numpy.ndarray

# (Click the "output" button to see the novel-length output)

help(rolls)
[3, 4, 1, 2, 2, 1] + 10
rolls + 10
# At which indices are the dice less than or equal to 3?

rolls <= 3
xlist = [[1,2,3],[2,4,6],]

# Create a 2-dimensional array

x = numpy.asarray(xlist)

print("xlist = {}\nx =\n{}".format(xlist, x))
# Get the last element of the second row of our numpy array

x[1,-1]
# Get the last element of the second sublist of our nested list?

xlist[1,-1]
import tensorflow as tf

# Create two constants, each with value 1

a = tf.constant(1)

b = tf.constant(1)

# Add them together to get...

a + b
print(dir(list))