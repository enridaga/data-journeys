
import scipy.stats as stats
import plotly.plotly as py
import pandas as pd
import numpy as np
import warnings
import scipy
import sys
import csv
import os
print('scipy: {}'.format(scipy.__version__))
print('pandas: {}'.format(pd.__version__))
print('numpy: {}'.format(np.__version__))
print('Python: {}'.format(sys.version))
warnings.filterwarnings('ignore')
### precision 2
print(os.listdir("../input/"))
hp_train=pd.read_csv('../input/train.csv')
import this
# Python is dynamically typed.
x = 2
y = .5
xy = 'Hey'
z=x+y
print(x,y,xy,z)
type(x)
type(y)
y=2
type(y)
def add_numbers(x, y):
    return x + y

add_numbers(1, 2)
def fizzbuzz(n):
    '''
    this FizzBuzz Question that asked in most of the interview.
    '''
    if n%3==0 and n%5==0:
        return('FizzBuzz')
    elif n%3==0:
        return('Fizz')
    elif n%5==0:
        return('Buzz')
    else:
        return(n)
for i in range(1,101):
    print(fizzbuzz(i))
type('This is a string')
type(None)
type(1)
type(1.0)
type(add_numbers)
tuple_sample = (1, 'a', 2, 'b')
type(tuple_sample)
list_sample = [1, 6, 2, 9]
type(list_sample)
list_sample.append(3.3)
print(list_sample)
list_sample.sort()
list_sample
list_sample.remove(1)
list_sample.reverse()
list_sample
for item in list_sample:
    print(item)
i=0
while( i != len(list_sample) ):
    print(list_sample[i])
    i = i + 1
x = 'This is a string'
print(x[0]) #first character
print(x[0:1]) #first character, but we have explicitly set the end character
print(x[0:2]) #first two characters

x = {'MJ Bahmani': 'Mohamadjavad.bahmani@gmail.com', 'irmatlab': 'irmatlab.ir@gmail.com'}
x['MJ Bahmani'] # Retrieve a value by using the indexing operator

print('MJ' + str(2))
sales_record = {
'price': 3.24,
'num_items': 4,
'person': 'MJ'}


type(sales_record)
sales_statement = '{} bought {} item(s) at a price of {} each for a total of {}'

print(sales_statement.format(sales_record['person'],
                             sales_record['num_items'],
                             sales_record['price'],
                             sales_record['num_items']*sales_record['price']))

#there are many way to import a csv file

with open('../input/train.csv') as csvfile:
    house_train = list(csv.DictReader(csvfile))
    
house_train[:1] # The first three dictionaries in our list.
type(house_train)
len(house_train)
house_train[0].keys()
sum(float(d['SalePrice']) for d in house_train) / len(house_train)
YearBuilt = set(d['YearBuilt'] for d in house_train)
print(type(YearBuilt ))
len(YearBuilt)
type(house_train[0] )
# just memorize this library
import datetime as dt
import time as tm
tm.time()
dtnow = dt.datetime.fromtimestamp(tm.time())
dtnow
dtnow.year, dtnow.month, dtnow.day, dtnow.hour, dtnow.minute, dtnow.second # get year, month, day, etc.from a datetime
delta = dt.timedelta(days = 100) # create a timedelta of 100 days
delta
today = dt.date.today()
class Person:
    department = 'School of Information' #a class variable

    def set_name(self, new_name): #a method
        self.name = new_name
    def set_location(self, new_location):
        self.location = new_location
# define an object
person = Person()
# set value for the object
person.set_name('MJ Bahmani')
person.set_location('MI, Berlin, Germany')
print('{} live in {} and works in the department {}'.format(person.name, person.location, person.department))
my_function = lambda a, b, c : a + b +c
my_function(1, 2, 3)
my_list = []
for number in range(0, 9):
    if number % 2 == 0:
        my_list.append(number)
my_list
my_list = [number for number in range(0,10) if number % 2 == 0]
my_list
class FirstClass:
    test = 'test'
    def __init__(self,name,symbol):
        self.name = name
        self.symbol = symbol
eg3 = FirstClass('Three',3)
print (eg3.test, eg3.name,eg3.symbol)
class FirstClass:
    def __init__(self,name,symbol):
        self.name = name
        self.symbol = symbol
    def square(self):
        return self.symbol * self.symbol
    def cube(self):
        return self.symbol * self.symbol * self.symbol
    def multiply(self, x):
        return self.symbol * x
eg4 = FirstClass('Five',5)
print (eg4.square())
print (eg4.cube())
eg4.multiply(2)
FirstClass.multiply(eg4,2)
class SoftwareEngineer:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def salary(self, value):
        self.money = value
        print (self.name,"earns",self.money)
a = SoftwareEngineer('Kartik',26)
a.salary(40000)
dir(SoftwareEngineer)
class Artist:
    def __init__(self,name,age):
        self.name = name
        self.age = age
    def money(self,value):
        self.money = value
        print (self.name,"earns",self.money)
    def artform(self, job):
        self.job = job
        print (self.name,"is a", self.job)
b = Artist('Nitin',20)
b.money(50000)
b.artform('Musician')
dir(Artist)


# some JSON:
x =  '{ "name":"John", "age":30, "city":"New York"}'

# parse x:
y = json.loads(x)

# the result is a Python dictionary:
print(y["age"])
# a Python object (dict):
x = {
  "name": "John",
  "age": 30,
  "city": "New York"
}

# convert into JSON:
y = json.dumps(x)

# the result is a JSON string:
print(y)
print(json.dumps({"name": "John", "age": 30}))
print(json.dumps(["apple", "bananas"]))
print(json.dumps(("apple", "bananas")))
print(json.dumps("hello"))
print(json.dumps(42))
print(json.dumps(31.76))
print(json.dumps(True))
print(json.dumps(False))
print(json.dumps(None))
x = {
  "name": "MJ",
  "age": 32,
  "married": True,
  "divorced": False,
  "children": ("Ann","Billy"),
  "pets": None,
  "cars": [
    {"model": "BMW 230", "mpg": 27.5},
    {"model": "Ford Edge", "mpg": 24.1}
  ]
}

print(json.dumps(x))
try:
  print(x)
except NameError:
  print("Variable x is not defined")
except:
  print("Something else went wrong")
try:
  print(x)
except:
  print("Something went wrong")
finally:
  print("The 'try except' is finished")
mytuple = ("apple", "banana", "cherry")
myit = iter(mytuple)

print(next(myit))
print(next(myit))
print(next(myit))
mytuple = ("apple", "banana", "cherry")

for x in mytuple:
  print(x)
thisdict =	{
  "brand": "Ford",
  "model": "Mustang",
  "year": 1964
}
print(thisdict)
thistuple = ("apple", "banana", "cherry")
print(thistuple)
house_price = {"store", "apartment", "house"}
print(house_price)
house_price = {"store", "apartment", "house"}

for x in house_price:
  print(x)
thisset = {"apple", "banana", "cherry"}

thisset.add("orange")

print(thisset)