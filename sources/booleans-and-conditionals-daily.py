
x = True

print(x)

print(type(x))
def can_run_for_president(age):

    """Can someone of the given age run for president in the US?"""

    # The US Constitution says you must be at least 35 years old

    return age >= 35



print("Can a 19-year-old run for president?", can_run_for_president(19))

print("Can a 45-year-old run for president?", can_run_for_president(45))
3.0 == 3
'3' == 3
def is_odd(n):

    return (n % 2) == 1



print("Is 100 odd?", is_odd(100))

print("Is -1 odd?", is_odd(-1))
def can_run_for_president(age, is_natural_born_citizen):

    """Can someone of the given age and citizenship status run for president in the US?"""

    # The US Constitution says you must be a natural born citizen *and* at least 35 years old

    return is_natural_born_citizen and (age >= 35)



print(can_run_for_president(19, True))

print(can_run_for_president(55, False))

print(can_run_for_president(55, True))
True or True and False
def inspect(x):

    if x == 0:

        print(x, "is zero")

    elif x > 0:

        print(x, "is positive")

    elif x < 0:

        print(x, "is negative")

    else:

        print(x, "is unlike anything I've ever seen...")



inspect(0)

inspect(-15)
def f(x):

    if x > 0:

        print("Only printed when x is positive; x =", x)

        print("Also only printed when x is positive; x =", x)

    print("Always printed, regardless of x's value; x =", x)



f(1)

f(0)
print(bool(1)) # all numbers are treated as true, except 0

print(bool(0))

print(bool("asf")) # all strings are treated as true, except the empty string ""

print(bool(""))

# Generally empty sequences (strings, lists, and other types we've yet to see like lists and tuples)

# are "falsey" and the rest are "truthy"
if 0:

    print(0)

elif "spam":

    print("spam")