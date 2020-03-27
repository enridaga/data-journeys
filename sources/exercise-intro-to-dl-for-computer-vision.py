
# Set up code checking
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning.exercise_1 import *
print("Setup Complete")
horizontal_line_conv = [[1, 1], 
                        [-1, -1]]
# load_my_image and visualize_conv are utility functions provided for this exercise
original_image = load_my_image() 
visualize_conv(original_image, horizontal_line_conv)
vertical_line_conv = ____

q_1.check()
visualize_conv(original_image, vertical_line_conv)
#q_1.hint()
#q_1.solution()
#q_2.solution()