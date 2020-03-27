
import collections

def find_most_common(values):
    """"Return the value of the most common item in a list"""
    list_counts = collections.Counter(values)
    top_two_values = list_counts.most_common(2)

    # make sure we don't have a tie for most common
    assert top_two_values[0][1] != top_two_values[1][1]\
        ,"There's a tie for most common value"
    
    return(top_two_values[0][0])
values_list = [1, 2, 3, 4, 5, 5, 5]

find_most_common(values_list)
values_list = [1, 2, 3, 4, 4, 4, 5, 5, 5]

find_most_common(values_list)