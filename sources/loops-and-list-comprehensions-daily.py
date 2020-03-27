
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

for planet in planets:

    print(planet, end=' ') # print all on same line
multiplicands = (2, 2, 2, 3, 3, 5)

product = 1

for mult in multiplicands:

    product = product * mult

product
s = 'steganograpHy is the practicE of conceaLing a file, message, image, or video within another fiLe, message, image, Or video.'

msg = ''

# print all the uppercase letters in s, one at a time

for char in s:

    if char.isupper():

        print(char, end='')        
for i in range(5):

    print("Doing important work. i =", i)
i = 0

while i < 10:

    print(i, end=' ')

    i += 1
squares = [n**2 for n in range(10)]

squares
squares = []

for n in range(10):

    squares.append(n**2)

squares
short_planets = [planet for planet in planets if len(planet) < 6]

short_planets
# str.upper() returns an all-caps version of a string

loud_short_planets = [planet.upper() + '!' for planet in planets if len(planet) < 6]

loud_short_planets
[

    planet.upper() + '!' 

    for planet in planets 

    if len(planet) < 6

]
[32 for planet in planets]
def count_negatives(nums):

    """Return the number of negative numbers in the given list.

    

    >>> count_negatives([5, -1, -2, 0, 3])

    2

    """

    n_negative = 0

    for num in nums:

        if num < 0:

            n_negative = n_negative + 1

    return n_negative
def count_negatives(nums):

    return len([num for num in nums if num < 0])
def count_negatives(nums):

    # Reminder: in the "booleans and conditionals" exercises, we learned about a quirk of 

    # Python where it calculates something like True + True + False + True to be equal to 3.

    return sum([num < 0 for num in nums])