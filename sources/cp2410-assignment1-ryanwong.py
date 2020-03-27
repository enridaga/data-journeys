
# Import csv module to interact with the input data
import csv
import math
import sys

points_list = []
shortest_pt_distance = float('inf')
best_path = []
best_distance = float('inf')
input_data = "../input/a1-data/first3.csv"  # Enter filename of input data here.
output_data = "output.csv"  # Enter filename of output data here
starting_point = [0.0, 0.0, 0.0]  # Starting point declared here. Format: [0.0, x-coord, y-coord]
ending_point = [0.0, 0.0, 0.0]  # Ending point declared here. Format: [0.0, x-coord, y-coord]
with open(input_data, 'r') as csvfile:  # reading the csv file
        csvreader = csv.reader(csvfile)     # Creating a reader
        col_names = next(csvreader)         # Extract column names
        for point_STR in csvreader:         # Add data points into internal list
            point_FLT = []                  # Convert the data points from String to Float
            for i in point_STR:
                point_FLT.append(float(i))
            points_list.append(point_FLT)
        
def save_best_path():
    with open(output_data, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(best_path)
if sys.getrecursionlimit() < len(points_list): # Checks if recursion limit will be hit
    sys.setrecursionlimit(2*len(points_list))  # Extends it if yes
def find_dist(pt1, pt2):                # Function to find the distance between 2 points
    pt1 = int(pt1)                      # Calculation below assumes points are on flat surface
    pt2 = int(pt2)
    distance = math.sqrt(((points_list[pt1][1] - points_list[pt2][1])**2) + ((points_list[pt1][2] - points_list[pt2][2])**2))
    return distance

def find_dist_start(pt):                # Extra function used to account for starting point if provided
    pt = int(pt)                        # Calculation below assumes points are on flat surface
    distance = math.sqrt(((starting_point[1] - points_list[pt][1])**2) + ((starting_point[2] - points_list[pt][2])**2))
    return distance

def find_dist_end(pt):                  # Extra function used to account for ending point if provided
    pt = int(pt)                        # Calculation below assumes points are on flat surface
    distance = math.sqrt(((ending_point[1] - points_list[pt][1])**2) + ((ending_point[2] - points_list[pt][2])**2))
    return distance
# Recursive function to find shortest point-to-point distance
def min_point_distance():
    global shortest_pt_distance
    num_pts_left = len(points_list)
    points_left = points_list.copy()    # Creates a separate list to be sorted by coordinates
    best_pt1 = 0.0
    best_pt2 = 0.0
    for i in range(num_pts_left - 1):               # Begins a loop to check every point
        for j in range(i+1, num_pts_left):          # Begins second loop to check against other points
            if find_dist(points_left[i][0], points_left[j][0]) < shortest_pt_distance:    # Checks if better
                shortest_pt_distance = find_dist(points_left[i][0], points_left[j][0])    # Saves better difference
                best_pt1 = points_left[i][0]        # Saves point indexes
                best_pt2 = points_left[j][0]
    print("Shortest distance is between point: ", best_pt1, " and ", best_pt2)
    print("Shortest distance is: ", shortest_pt_distance)
    return shortest_pt_distance

def path_is_viable(num_pts_left, curr_distance, curr_pt):
    if (((num_pts_left) * shortest_pt_distance) + curr_distance + find_dist_end(curr_pt)) < best_distance:
        return True
    else:
        return False
    
def path_is_better(curr_path, curr_dist):
    global best_path
    global best_distance
    if curr_dist < best_distance:
        best_path = curr_path.copy()
        best_distance = curr_dist
# Recursive function to plot every possible path
def plot_path(num_pts_left, curr_path, pts_left, pts_list, curr_dist):
    if len(curr_path) == 0 and len(pts_left) == 0 and pts_list != 0:    # Makes a separate list to track unused pts
        pts_left = pts_list.copy()
    if len(curr_path) == 0:
        curr_path.append(starting_point)    # Add starting point
    for point in pts_left:                  # Begins loop for each point in data to plot each path possible
        curr_path.append(point)             # Adds point in path to keep track of points used
        pts_left.remove(point)              # Removes this point from tracking of unused points

        if len(starting_point) == 3 and len(curr_path) == 2:    # Check for starting point
            curr_dist = find_dist_start(point[0])               # Adding distance from start to first point if yes
        else:
            curr_dist += find_dist(curr_path[-2][0], point[0])  # Update distance of current path

        if num_pts_left == 1:                                   # Checks if this was the last point.
            curr_dist += find_dist_end(point[0])                # Adding distance from this point to end if yes.
            path_is_better(curr_path, curr_dist)

        if path_is_viable(num_pts_left, curr_dist, point[0]):
            plot_path(num_pts_left - 1, curr_path, pts_left, pts_list, curr_dist)

        curr_dist -= find_dist(curr_path[-2][0], point[0])
        pts_left.insert(int(point[0]), point)
        pts_left.sort()
        curr_path.remove(point)



min_point_distance()
plot_path(len(points_list), [], [], points_list, 0)
best_path.remove(starting_point)
save_best_path()
print("Best distance is: ", best_distance)
print("Using best path: ", best_path)
# Import csv module to interact with the input data
import csv
import math
import sys

points_list = []
shortest_pt_distance = float('inf')
next_point = []
best_path = []
best_distance = float('inf')
input_data = "../input/a1-data/cities10per.csv"  # Enter filename of input data here.
output_data = "output.csv"  # Enter filename of output data here
starting_point = [0.0, 0.0, 0.0]  # Starting point declared here. Format: [0.0, x-coord, y-coord]
ending_point = [0.0, 0.0, 0.0]  # Ending point declared here. Format: [0.0, x-coord, y-coord]

with open(input_data, 'r') as csvfile:  # reading the csv file
    csvreader = csv.reader(csvfile)     # Creating a reader
    col_names = next(csvreader)         # Extract column names
    for point_STR in csvreader:         # Add data points into internal list
        point_FLT = []                  # Convert the data points from String to Float
        for i in point_STR:
            point_FLT.append(float(i))
        points_list.append(point_FLT)
        
def save_best_path():
    with open(output_data, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(best_path)
        
def find_dist(pt1, pt2):                # Function to find the distance between 2 points
    pt1 = int(pt1)                      # Calculation below assumes points are on flat surface
    pt2 = int(pt2)
    distance = math.sqrt(((points_list[pt1][1] - points_list[pt2][1])**2) + ((points_list[pt1][2] - points_list[pt2][2])**2))
    return distance

def find_dist_start(pt):                # Extra function used to account for starting point if provided
    pt = int(pt)                        # Calculation below assumes points are on flat surface
    distance = math.sqrt(((starting_point[1] - points_list[pt][1])**2) + ((starting_point[2] - points_list[pt][2])**2))
    return distance

def find_dist_end(pt):                  # Extra function used to account for ending point if provided
    pt = int(pt)                        # Calculation below assumes points are on flat surface
    distance = math.sqrt(((ending_point[1] - points_list[pt][1])**2) + ((ending_point[2] - points_list[pt][2])**2))
    return distance
points_list_sortedX = []

points_list_sortedX = points_list.copy()
points_list_sortedX.sort(key=lambda x: x[1])
def find_closest_pt(curr_point, search_range):              # Recursive function to find closest point
    curr_x_index = points_list_sortedX.index(curr_point)    # Defines the index of current point on sorted list
    possible_points_1 = []
    possible_points_2 = []
    dist_holder = float('inf')
    index_holder = 0.0
    if len(points_list_sortedX) == 1:                                   # Checks if this is the last point is list
        return False                                                    # Returns False if yes
    if curr_x_index > 0:                                                # Checks if any points behind current point
        for element in points_list_sortedX[curr_x_index-1::-1]:         # If yes, iterate backwards to find points
            if abs(curr_point[1] - element[1]) < search_range:
                possible_points_1.append(element)                       # Saves points if x coords in range
            else:
                break                                                   # Ends loop if no more points in range
    for element in points_list_sortedX[curr_x_index + 1:]:              # Now iterate forwards to find points
        if abs(curr_point[1] - element[1]) < search_range:
            possible_points_1.append(element)                           # Saves points to list1 if x coords in range
        else:
            break                                                       # Ends loop if no more points in range
    if curr_point in possible_points_1:
        possible_points_1.remove(curr_point)
    if len(possible_points_1) > 0:                                      # While there are any possible points
        possible_points_1.sort(key=lambda x: x[2])                      # Sort points by Y coords
        for point2 in possible_points_1:
            if abs(curr_point[2] - point2[2]) < search_range:           # Check if points Y coords are also in range
                possible_points_2.append(point2)                        # Saves points to list2 if yes

        if len(possible_points_2) > 0:                                  # If there are still possible points
            for point3 in possible_points_2:                            # Check all point's distance for closest
                if find_dist(point3[0], curr_point[0]) < dist_holder:
                    dist_holder = find_dist(point3[0], curr_point[0])
                    index_holder = point3[0]
            return index_holder                                         # Returns index of closest point
    return find_closest_pt(curr_point, (search_range + 100)*2)          # Re-curves with bigger range if no points
def plot_path():                                                        # Function that plots the best path
    global best_path                                                    # Initiates the global variables to record
    global best_distance                                                #   best path and distance
    points_list_sortedX.insert(0, starting_point)                       # Insert the starting point to points list
    best_path.append(points_list[int(find_closest_pt(starting_point, 100))])    # Tracks first point
    best_distance = find_dist_start(find_closest_pt(starting_point, 100))       # Measures starting distance
    points_list_sortedX.remove(starting_point)                          # Removes starting point from points list

    while len(points_list_sortedX) > 1:                                 # While there are unused points
        next_pt_index = find_closest_pt(best_path[-1], 100)             # Find the closest pt to the last added pt
        points_list_sortedX.remove(best_path[-1])                       # Remove last added pt from unused pts list
        best_path.append(points_list[int(next_pt_index)])               # Add the next point to path
        best_distance += find_dist(best_path[-2][0], best_path[-1][0])
        completion_percent = len(best_path)/len(points_list)
        print("Completion percent: ", completion_percent * 100, "%")

    best_path.append(points_list_sortedX[0])                            # Adds last point to end of the path
    best_distance += find_dist(best_path[-2][0], best_path[-1][0])      # Adds distance to last point
    best_distance += find_dist_end(best_path[-1][0])
plot_path()
save_best_path()
global best_path                # Initiates the global variables to print progress into console
global best_distance
print("Best distance is: ", best_distance)
print("Using best path: ", best_path)
from IPython.display import Image
