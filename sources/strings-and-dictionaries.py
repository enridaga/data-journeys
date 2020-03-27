
x = 'Pluto is a planet'

y = "Pluto is a planet"

x == y
print("Pluto's a planet!")

print('My dog is named "Pluto"')
'Pluto's a planet!'
'Pluto\'s a planet!'
hello = "hello\nworld"

print(hello)
triplequoted_hello = """hello

world"""

print(triplequoted_hello)

triplequoted_hello == hello
print("hello")

print("world")

print("hello", end='')

print("pluto", end='')
# Indexing

planet = 'Pluto'

planet[0]
# Slicing

planet[-3:]
# How long is this string?

len(planet)
# Yes, we can even loop over them

[char+'! ' for char in planet]
planet[0] = 'B'

# planet.append doesn't work either
# ALL CAPS

claim = "Pluto is a planet!"

claim.upper()
# all lowercase

claim.lower()
# Searching for the first index of a substring

claim.index('plan')
claim.startswith(planet)
claim.endswith('dwarf planet')
words = claim.split()

words
datestr = '1956-01-31'

year, month, day = datestr.split('-')
'/'.join([month, day, year])
# Yes, we can put unicode characters right in our string literals :)

' 👏 '.join([word.upper() for word in words])
planet + ', we miss you.'
position = 9

planet + ", you'll always be the " + position + "th planet to me."
planet + ", you'll always be the " + str(position) + "th planet to me."
"{}, you'll always be the {}th planet to me.".format(planet, position)
pluto_mass = 1.303 * 10**22

earth_mass = 5.9722 * 10**24

population = 52910390

#         2 decimal points   3 decimal points, format as percent     separate with commas

"{} weighs about {:.2} kilograms ({:.3%} of Earth's mass). It is home to {:,} Plutonians.".format(

    planet, pluto_mass, pluto_mass / earth_mass, population,

)
# Referring to format() arguments by index, starting from 0

s = """Pluto's a {0}.

No, it's a {1}.

{0}!

{1}!""".format('planet', 'dwarf planet')

print(s)
numbers = {'one':1, 'two':2, 'three':3}
numbers['one']
numbers['eleven'] = 11

numbers
numbers['one'] = 'Pluto'

numbers
planets = ['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune']

planet_to_initial = {planet: planet[0] for planet in planets}

planet_to_initial
'Saturn' in planet_to_initial
'Betelgeuse' in planet_to_initial
for k in numbers:

    print("{} = {}".format(k, numbers[k]))
# Get all the initials, sort them alphabetically, and put them in a space-separated string.

' '.join(sorted(planet_to_initial.values()))
for planet, initial in planet_to_initial.items():

    print("{} begins with \"{}\"".format(planet.rjust(10), initial))
help(dict)