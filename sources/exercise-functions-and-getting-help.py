
# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')
def round_to_two_places(num):
    """Return the given number rounded to two decimal places. 
    
    >>> round_to_two_places(3.14159)
    3.14
    """
    # Replace this body with your own code.
    # ("pass" is a keyword that does literally nothing. We used it as a placeholder
    # because after we begin a code block, Python requires at least one line of code)
    return round(num, 2)
   
    q1.check()
# verification of my code
print(round_to_two_places(2.589)) 
# Uncomment the following for a hint
# q1.hint()
# Or uncomment the following to peek at the solution
# q1.solution()
# Put your test code here
def round_to_two_places_negative_arguments(num):

    # round to two place but using a negative second argument
    # this function can be helpful when you want to round a very large number
    return round(num, -2)
  
    # checking my function
print(round_to_two_places_negative_arguments(1000000002.58)) 
#q2.solution()

def to_smash(total_candies, number_of_friends=3):
    """Return the number of leftover candies that must be smashed after distributing
    the given number of candies evenly between 3 friends.
    
    >>> to_smash(91)
    1
    """
    number_candies_tosmash = total_candies % number_of_friends
    return number_candies_tosmash
    print(number_candies_tosmash)
q3.check()
#test of the function
to_smash(31,17)

#q3.hint()
#q3.solution()
round_to_two_places(9.9999)
 x = -10
 y = 5
 # Which of the two variables above has the smallest absolute value?
 smallest_abs = min(abs(x),abs(y))
 # print the value of smallest_abs
print(smallest_abs)
def f(x):
     y = abs(x)
     return y

print(f(5))
print(f(-6))