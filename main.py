import json
from Item import Item
from Container import Container
from Draw import Draw
from shapely.geometry import Polygon,MultiPolygon,Point,MultiPoint,MultiLineString
import pymunk
import sys


from Algo import Algo
from Algo2 import Algo2
from Algo3 import Algo3
from Algo4 import Algo4
from Algo6 import Algo6
from Algo7 import Algo7
from Algo8 import Algo8
from Algo9 import Algo9
from Algo10 import Algo10
from Algo11 import Algo11
from Algo12 import Algo12
from Algo13 import Algo13
from Algo14 import Algo14
from Algo15 import Algo15
from Algo16 import Algo16
from Algo17 import Algo17
from Algo18 import Algo18
from Algo19 import Algo19
from Algo20 import Algo20
from Algo21 import Algo21
from Algo22 import Algo22
from Algo23 import Algo23
from Algo24 import Algo24
from Algo25 import Algo25
from Algo26 import Algo26
from Algo27 import Algo27
from Algo28 import Algo28
from Algo29 import Algo29
from Algo30 import Algo30
from Algo31 import Algo31
from Algo32 import Algo32
from Algo33 import Algo33
from Algo34 import Algo34
from Algo35 import Algo35
from Algo36 import Algo36
from Algo37 import Algo37
from Algo38 import Algo38
from Algo39 import Algo39
from Algo40 import Algo40
from Algo41 import Algo41
from Algo42 import Algo42
from Algo43 import Algo43
from Algo44 import Algo44
from Algo45 import Algo45
from Algo46 import Algo46
from Algo48 import Algo48
from Algo49 import Algo49
from Algo50 import Algo50











































import pygame
from pygame.math import Vector2

import random

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation


import math
import copy
from shapely.validation import explain_validity
from shapely.geometry import Polygon

from shapely import normalize, Polygon, coverage_union, overlaps,distance,LineString,hausdorff_distance
from shapely.ops import unary_union

def load_json(filename):
    with open(filename, 'r') as file:
        return json.loads(file.read())


def print_container_info(container_instance):
    print("\nContainer:")
    for i, (x, y) in enumerate(zip(container_instance.x_coords, container_instance.y_coords), start=1):
        print(f"  Point {i}: ({x}, {y})")
    slope_dict = container_instance.slope_list()
    for edge, slope in slope_dict.items():
        print(f"Edge: {edge}, Slope: {slope}")
    equations = container_instance.calculate_line_equations()
    for edge, equation in equations.items():
        print(f"{edge}: {equation}")
    middle_projections =container_instance.determine_all_middle_projection_directions()
    for i, direction in enumerate(middle_projections):
        print(f"Edge {i + 1} middle point projection: {direction}")

def print_item_info(item_instances):
    for item_index, item_instance in enumerate(item_instances):
        print("\nItem", item_index + 1, ":")
        print("  Quantity:", item_instance.quantity)
        print("  Value:", item_instance.value)
        print("  Coordinates (x, y):")
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")
        print("  Slopes:")
        slopes = item_instance.slope_list()
        for edge, slope in slopes.items():
            print(f"    Edge {edge}: Slope {slope}")
        equations = item_instance.calculate_line_equations()
        for edge, equation in equations.items():
            print(f"{edge}: {equation}")

def test(item_instances, container_instance):
    edge_start, edge_end, middle_point = random.choice(container_instance.calculate_middle_points())
    proj = container_instance.determine_projection_direction_from_middle(middle_point)
    middle_x, middle_y = middle_point

    for item_index, item_instance in enumerate(item_instances):
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")
        if proj == "left":
            x, y = item_instance.find_point_with_max_x()
            print(proj, x, y, middle_x, middle_y)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y, middle_x, middle_y)
        elif proj == "right":
            x, y = item_instance.find_point_with_min_x()
            print(proj, x, y, middle_x, middle_x)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y, middle_x, middle_y)
        elif proj == "up":
            x, y = item_instance.find_point_with_min_y()
            print(proj, x, y, middle_x, middle_y)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y, middle_x, middle_y)
        elif proj == "down":
            x, y = item_instance.find_point_with_max_y()
            print(proj, x, y,middle_x, middle_y)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y,middle_x, middle_y)
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")
        break


def item_with_max_dimensions(item_list):
        """
        Find and return the item with the maximum dimensions (either width or height) from a list of items.

        Args:
        - item_list: List of Item objects.

        Returns:
        - The Item object with the maximum dimensions.
        """
        max_item = None
        max_dimension = 0

        for item in item_list:
            max_dimension_for_item = item.calculate_total_dimensions()

            if max_dimension_for_item > max_dimension:
                max_item = item
                max_dimension = max_dimension_for_item

        return max_item, max_dimension

def item_with_min_dimensions(item_list):
        """
        Find and return the item with the minimum dimensions (either width or height) from a list of items.

        Args:
        - item_list: List of Item objects.

        Returns:
        - The Item object with the minimum dimensions.
        """
        min_item = None
        min_dimension = float('inf')

        for item in item_list:
            min_dimension_for_item = item.calculate_total_dimensions()

            if min_dimension_for_item < min_dimension:
                min_item = item
                min_dimension = min_dimension_for_item

        return min_item, min_dimension

def find_center_point(point1, point2, point3, point4):
    """
    Find the center point of a rectangle given its four corner points.

    Args:
    - point1, point2, point3, point4: Tuple (x, y) representing the four corner points of the rectangle.

    Returns:
    - The center point of the rectangle as a tuple (x, y).
    """
    x_values = [point1[0], point2[0], point3[0], point4[0]]
    y_values = [point1[1], point2[1], point3[1], point4[1]]

    center_x = sum(x_values) / 4
    center_y = sum(y_values) / 4

    return (center_x, center_y)


def check(point):
    """
           Check if a point is inside the item.

           Args:
           - point: (x, y) tuple representing the point to check.

           Returns:
           - True if the point is inside the container, False otherwise.
           """
    x, y = point
    list1= [(1620.6526169403587, 4094.7676954873227), (1620.6526169403587, 3192.7676954873227), (3965.6526169403587, 3192.7676954873227), (3965.6526169403587, 4094.7676954873227)]
    list2= [(479.3810687845348, 2069.595611747717), (2790.381068784535, 2069.595611747717), (2790.381068784535, 3088.595611747717), (1445.3810687845348, 3088.595611747717), (1445.3810687845348, 3723.595611747717), (479.3810687845348, 3723.595611747717)]
    n = len(list2)
    inside = False

    for i in range(n):
        x1, y1 = list2[i]
        x2, y2 = list2[(i + 1) % n]

        if (y1 < y <= y2 or y2 < y <= y1) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside

    return inside
def get_edge_lines(coord):
        edges = []
        num_points = len(coord)

        for i in range(num_points):
            point1 = coord[i]
            point2 = coord[(i + 1) % num_points]  # Wrap around to the first point

            line = (point1, point2)
            edges.append(line)

        return edges
def calculate_slope( x1, y1, x2, y2):
    if x1 == x2:
        # The slope is float('-inf') for vertical lines
        return float('-inf')
    elif y1 == y2:
        # The slope is float('inf') for horizontal lines
        return float('inf')
    else:
        return (y2 - y1) / (x2 - x1)

def edge_length(x1, y1, x2, y2):
    # Calculate the length of the edge using the Euclidean distance formula
    length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return length

def line_a_cross(x,y,slope):
    if slope == float('inf'):
        #horizontal = y
        #x = c
        return float('-inf'), x
    elif slope == float('-inf'):
        #vertical
        #y = c
        return float('inf'), y
    else:
        s = -1 / slope
        # y = mx+n
        # y-y1 = m(x-x1)
        # y = mx-mx1+y1
        # n = -mx1+y1
        n = y - s * x
        return s, n

def line_equation(x,y,slope):
    if slope == float('inf'):
        # The slope is for horizontal lines
        #y = c
        return float('inf'), y
    elif slope == float('-inf'):
        # The slope is for vertical lines
        #x = c
        return float('-inf'), x
    else:
        # y = mx+n
        # y-y1 = m(x-x1)
        # y = mx-mx1+y1
        # n = -mx1+y1
        n = y - slope * x
        return slope, n

def do_lines_intersect(m1, n1, m2, n2):
    # Check if slopes are equal (parallel lines)
    if m1 == float('inf'):
        # horizontal,n1 = y
        return n2, n1
    elif m1 == float('-inf'):
        # vertical,n1 = x
        return n1, n2

    # Calculate the x-coordinate of the intersection point
    x_intersection = (n2 - n1) / (m1 - m2)

    # Calculate the y-coordinate using either line equation (both should yield the same result)
    y_intersection = m1 * x_intersection + n1
    # Check if x_intersection and y_intersection are not None
    if x_intersection is not None and y_intersection is not None:
        return x_intersection, y_intersection # Intersection exists
    else:
        return float('inf'), float('inf') # No intersection

def calculate_midpoint(x1, y1, x2, y2):
    midpoint_x = (x1 + x2) / 2
    midpoint_y = (y1 + y2) / 2
    return midpoint_x, midpoint_y

def is_point_between(x, y, x1, y1, x2, y2):
    """
    Check if a point is between two other points.

    Args:
    point: A tuple (x, y) representing the point to check.
    start_point: A tuple (x1, y1) representing the starting point.
    end_point: A tuple (x2, y2) representing the ending point.

    Returns:
    True if the point is between the two reference points, False otherwise.
    """
    # Check if the x-coordinate of the point is between the x-coordinates of the start and end points,
    # and if the y-coordinate of the point is between the y-coordinates of the start and end points.
    if (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1):
        return True

    return False

def edge_cover(p1,p2,q1,q2):
    x1, y1 = p1
    x2, y2 = p2
    f1,f2 = q1
    t1,t2 = q2
    ed1 = edge_length(x1, y1, x2, y2)
    ed2 = edge_length(f1, f2, t1, t2)
    x = None
    y = None
    if ed1 > ed2:
        midx, midy = calculate_midpoint(f1, f2, t1, t2)
        print("check",Point(midx, midy).distance(LineString([p1,p2])))
        x, y = point_on_edge_closest_to_point(p1, p2, (midx, midy))
        print("mid point",midx,midy)
        print("x,y",x,y)
        return is_point_between(x,y, x1, y1, x2, y2)

    elif ed1 < ed2:
        midx, midy = calculate_midpoint(x1, y1, x2, y2)
        x, y = point_on_edge_closest_to_point(q1, q2, (midx, midy))
        print("check",Point(midx, midy).distance(LineString([q1,q2])))

        return is_point_between(x, y, f1, f2, t1,t2)

    else:
        midx, midy = calculate_midpoint(x1, y1, x2, y2)
        x, y = point_on_edge_closest_to_point(q1, q2, (midx, midy))
        return is_point_between(x, y, f1, f2, t1,t2)

def edges_alined(p1,p2,p3,p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    #check if point A of edge1 is in edge2
    test1 = is_point_between(x1, y1, x3, y3, x4, y4)
    #check if point B of edge1 is in edge2
    test2 = is_point_between(x2, y2, x3, y3, x4, y4)
    #check if point A of edge2 is in edge1
    test3 = is_point_between(x3, y3, x1, y1, x2, y2)
    # check if point B of edge2 is in edge1
    test4 = is_point_between(x4, y4, x1, y1, x2, y2)
    print(test1, p1, p3, p4)
    print(test2, p2, p3, p4)
    print(test3, p3, p1, p2)
    print(test4, p4, p1, p2)



    return (test1 and test2) or (test3 and test4)

def alined(p1,p2,q1,q2):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = q1
    x4, y4 = q2
    slope1 = calculate_slope(x1, y1, x2, y2)
    slope2 = calculate_slope(x3, y3, x4, y4)
    if slope1 == slope2:
        # which line is shorter:
        edge1 = edge_length(x1, y1, x2, y2)
        edge2 = edge_length(x3, y3, x4, y4)

        if edge1 > edge2:
            # calculate the smaller one in this case edge2
            mx, my = calculate_midpoint(x3, y3, x4, y4)
            print("middle point:","x",mx,"y", my)
            print("between the points:","x",x3,"y", y3,"and" ,"x",x4,"y", y4)
            # calculate the line equation that is perpendicular to edge1 in the middle point of it
            m, n = line_a_cross(mx, my, slope2)
            if m == float('inf'):
                print("line equation of the middle point a cross:", "y=", n)
            if m == float('-inf'):
                print("line equation of the middle point a cross:", "x=", n)


            # find line equation of edge 1

            m2, n2 = line_equation(x1, y2, slope1)
            # does the line cross edge 1
            a, b = do_lines_intersect(m, n, m2, n2)
            #print("x1:",x1,"y2:", y2,"slope:",slope1,"m:",m,"n:",n,"m2:",m2,"n2:", n2,"a:",a,"b:",b)


            return is_point_between(a, b, x1, y1, x2, y2)
        elif edge1 < edge2:
            # calculate the smaller one in this case edge2
            mx, my = calculate_midpoint(x1, y1, x2, y2)
            # calculate the line equation that is perpendicular to edge1 in the middle point of it
            m, n = line_a_cross(mx, my, slope1)
            # find line equation of edge 2
            m2, n2 = line_equation(x3, y3, slope2)
            # does the line cross edge 2
            a, b = do_lines_intersect(m, n, m2, n2)
            return is_point_between(a, b, x3, y3, x4, y4)
        else:
            # does not matter which one
            calculate_midpoint(x1, y1, x2, y2)
            mx, my = calculate_midpoint(x1, y1, x2, y2)
            # calculate the line equation that is perpendicular to edge1 in the middle point of it
            m, n = line_a_cross(mx, my, slope1)
            #find line equation of edge 2
            m2, n2 = line_equation(x3, y3, slope2)
            # does the line cross edge 2
            a, b = do_lines_intersect(m, n, m2, n2)
            return is_point_between(a, b, x3, y3, x4, y4)
    else:
        return False

def find_point_on_edge_for_90_degree_angle(point_a, edge_start, edge_end):
    # Unpack the coordinates
    x_a, y_a = point_a
    x1, y1 = edge_start
    x2, y2 = edge_end

    # Calculate the direction vector from point A to edge start
    D = (x1 - x_a, y1 - y_a)

    # Calculate the direction vector of the edge
    E = (x2 - x1, y2 - y1)

    # Normalize the vectors
    magnitude_D = math.sqrt(D[0] ** 2 + D[1] ** 2)
    D_normalized = (D[0] / magnitude_D, D[1] / magnitude_D)

    magnitude_E = math.sqrt(E[0] ** 2 + E[1] ** 2)
    E_normalized = (E[0] / magnitude_E, E[1] / magnitude_E)

    # Calculate the dot product
    dot_product = D_normalized[0] * E_normalized[0] + D_normalized[1] * E_normalized[1]

    # Find the point of intersection
    intersection_x = x_a + dot_product * magnitude_D * E_normalized[0]
    intersection_y = y_a + dot_product * magnitude_D * E_normalized[1]

    return (intersection_x, intersection_y)

# Calculate the Euclidean distance between two points
def calculate_distance_of_points(x1, y1, x2, y2):
    distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return distance

def clean_coords(mergedPolys):
    exterior_coords_list = []
    if isinstance(mergedPolys, MultiPolygon):
        # Iterate through the constituent polygons
        for polygon in mergedPolys.geoms:
            # Get the coordinates of the exterior boundary of each polygon
            exterior_coords = list(polygon.exterior.coords)
            # Append them to the exterior_coords_list
            return exterior_coords_list.extend(exterior_coords)
    else:
        # If it's a single Polygon, get its exterior coordinates directly
        return list(mergedPolys.exterior.coords)

def calculate_slope_and_angle(point1_edge1, point2_edge1, point1_edge2, point2_edge2):
    # Check if either line is horizontal (slope = 0).
    if point1_edge1[1] == point2_edge1[1]:
        slope1 = 0
        angle1 = 0
    else:
        if point2_edge1[0] == point1_edge1[0]:
            slope1 = float("inf")
            angle1 = 90
        else:
            slope1 = (point2_edge1[1] - point1_edge1[1]) / (point2_edge1[0] - point1_edge1[0])
            angle1 = math.atan(slope1)

    if point1_edge2[1] == point2_edge2[1]:
        slope2 = 0
        angle2 = 0
    else:
        if point2_edge2[0] == point1_edge2[0]:
            slope2 = float("inf")
            angle2 = 90
        else:
            slope2 = (point2_edge2[1] - point1_edge2[1]) / (point2_edge2[0] - point1_edge2[0])
            angle2 = math.atan(slope2)

    # Calculate the angle between the two lines
    angle_between_lines = abs(angle2 - angle1)
    angle_between_lines_degrees = math.degrees(angle_between_lines)

    return angle_between_lines_degrees

def calculate_angle(point, centroid):
    return (math.atan2(point[1] - centroid[1], point[0] - centroid[0]) + 2 * math.pi) % (2 * math.pi)

def is_counterclockwise(coordinates):
    # Calculate the centroid
    centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
    centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
    centroid = (centroid_x, centroid_y)

    # Sort the coordinates based on angles
    sorted_coordinates = sorted(coordinates, key=lambda point: calculate_angle(point, centroid))

    # Check if the sorted coordinates are in counterclockwise order
    for i in range(len(sorted_coordinates) - 1):
        x1, y1 = sorted_coordinates[i]
        x2, y2 = sorted_coordinates[i + 1]
        cross_product = (x2 - x1) * (y2 + y1)
        if cross_product <= 0:
            return False

    return True

def order_coordinates_counterclockwise(coordinates):
    if is_counterclockwise(coordinates):
        print("orderd")
        return coordinates
    # Calculate the centroid
    centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
    centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
    centroid = (centroid_x, centroid_y)

    # Sort the coordinates based on angles
    sorted_coordinates = sorted(coordinates, key=lambda point: calculate_angle(point, centroid))

    return sorted_coordinates

def remove_duplicate_coordinates(coordinates):
    unique_coordinates = []
    for point in coordinates:
        if point not in unique_coordinates:
            unique_coordinates.append(point)
    return unique_coordinates

def filter_middle_points(points):
    if len(points) < 3:
        return points

    result = [points[0]]
    for i in range(1, len(points) - 1):
        previous_point = points[i - 1]
        current_point = points[i]
        next_point = points[i + 1]

        if (previous_point[0] + next_point[0]) / 2 == current_point[0] and (previous_point[1] + next_point[1]) / 2 == \
                current_point[1]:
            continue  # Skip the point if it's in the middle
        else:
            result.append(current_point)

    result.append(points[-1])
    return result

def remove_point_from_list(points, point_to_remove):
    if point_to_remove in points:
        points.remove(point_to_remove)

def insert_list_at_index(list1, list2, index):
    """
    Inserts list2 into list1 at the specified index.

    :param list1: The original list of points.
    :param list2: The list of points to be inserted.
    :param index: The index at which to insert list2.
    :return: The resulting list with list2 inserted at the specified index.
    """
    return list1[:index] + list2 + list1[index:]

def check_points(p1,p2,p3,p4):
    point1 = Point(p1)
    point2 = Point(p2)
    point3 = Point(p3)
    point4 = Point(p4)
    distance1 = point1.distance(point3)
    distance2 = point1.distance(point4)
    distance3 = point2.distance(point3)
    distance4 = point2.distance(point4)
    thepo = None
    thesec = None
    ch = None
    if distance1 == 0:
        thepo = p3
        thesec = p4
        ch = point2.distance(point4)
    if distance3 == 0:
        thepo = p3
        thesec = p4
        ch = point1.distance(point4)
    if distance2 == 0:
        thepo = p4
        thesec = p3
        ch = point2.distance(point3)
    if distance4 == 0:
        thepo = p4
        thesec = p3
        ch = point1.distance(point3)

    return thesec

def point_on_edge_closest_to_point(edge_start, edge_end, original_point):
    x1, y1 = edge_start
    x2, y2 = edge_end
    x0, y0 = original_point

    # Calculate the direction vector of the line formed by the edge
    dx = x2 - x1
    dy = y2 - y1

    # Calculate the parameter (t) for the point on the line closest to the original point
    t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx**2 + dy**2)

    # Find the coordinates of the closest point on the line
    x = x1 + t * dx
    y = y1 + t * dy

    return (x, y)

def is_point_on_edge(point, endpoint1, endpoint2):
    x, y = point
    x1, y1 = endpoint1
    x2, y2 = endpoint2

    vector1 = (x - x1, y - y1)
    vector2 = (x2 - x1, y2 - y1)

    # Calculate the dot product of vector1 and vector2
    dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

    # Calculate the magnitudes of vector1 and vector2
    magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
    magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5

    # Calculate the cosine of the angle between vector1 and vector2
    cosine = dot_product / (magnitude1 * magnitude2)

    # Check if the vectors are collinear (cosine is 1 or -1)
    return abs(cosine) == 1

def distance_to_center(edge,center):
    cx, cy = center
    (p1, p2) = edge
    x1, y1  = p1
    x2, y2  = p2
    # Calculate the distance from the center to both endpoints and return the minimum
    dist1 = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
    dist2 = ((x2 - cx) ** 2 + (y2 - cy) ** 2) ** 0.5
    return min(dist1, dist2)


def are_edges_close(p1,p2, q1,q2, distance):
    x1, y1 = p1
    x2,y2 = p2
    xt1,ty1 = q1
    xt2, yt2 = q2

    edge1_size = edge_length(x1, y1,x2,y2)
    edge2_size = edge_length(xt1,ty1,xt2, yt2)

    threshold_ratio = 0.5  # 30% threshold

    # Determine the size of the larger edge
    larger_edge_size = max(edge1_size, edge2_size)

    # Calculate the threshold
    threshold = larger_edge_size * threshold_ratio
    print("threshold",threshold)
    # Compare distance to threshold
    if distance < threshold:
        return True  # Edges are close
    else:
        return False  # Edges are not close


def requirements(p1, p2, t1, t2,epsilon):
    po1 = None
    po2 = None
    po3 = None
    po4 = None
    l1 = LineString([p1, p2])
    l2 = LineString([t1, t2])
    dist = hausdorff_distance(l1, l2)
    print("epsilon", epsilon, "dist", dist)
    angle = calculate_slope_and_angle(p1, p2, t1, t2)
    print("angle", angle)
    print("points", p1, p2, t1, t2)
    the_angle = None
    the_dist = None
    # draw_instance.plot()
    if dist < epsilon and (angle < 0.01 or angle > 179) and edge_cover(p1, p2, t1, t2) and are_edges_close(p1, p2, t1,
                                                                                                           t2, dist):
        print("is point between", edge_cover(p1, p2, t1, t2))
        print("edges are close", are_edges_close(p1, p2, t1, t2, dist))
        yes2 = True
        po1 = p1
        po2 = p2
        po3 = t1
        po4 = t2
        the_angle = angle
        the_dist = dist
        return True

    return False


def custom_polygon_sort(polygon):
    value = polygon.value
    area = polygon.calculate_regular_polygon_area()
    num_coordinates = len(polygon.coordinates)

    # The expression for sorting: value / (area + number of coordinates)
    return value / (area + num_coordinates)




def main():
    filename = input("Enter the filename containing JSON data: ")
    data = load_json(filename)

    container_instance = Container(data["container"]["x"], data["container"]["y"])

    item_instances = []
    for item_data in data["items"]:
        quantity = item_data["quantity"]
        value = item_data["value"]
        x_coords = item_data["x"]
        y_coords = item_data["y"]
        for _ in range(quantity):
            item_instance = Item(quantity, value, x_coords, y_coords)
            item_instances.append(item_instance)

   # print_container_info(container_instance)
    #print_item_info(item_instances)

    #test(item_instances, container_instance)

   # list = [[(0, 0), (1000, 0), (1000, 1000), (0, 1000)]]
    #grid_coordinates = container_instance.create_grid_coordinates(1000, list)

    #rectangles = container_instance.group_grid_coordinates_into_rectangles(1000)
    """
    low to high sort of items by their dimention 
    in the order of the sort we generate points by that dimention and place the item in one of the rectangles
    make sure to check that their exist such rectangles
    we put that item in a list of points to avoid in the next iteration
    """
    """
    # Sort the items by their total dimensions in ascending order
    sorted_items = sorted(item_instances, key=lambda item: item.calculate_total_dimensions())

    # Print the sorted items
    
    list = []
    i = 0
    for index, item in enumerate(sorted_items):
        dimen = item.calculate_total_dimensions()
        grid_coordinates = container_instance.create_grid_coordinates1(dimen, list)
        print(grid_coordinates)
        rec = container_instance.create_rectangles(grid_coordinates, dimen, list)
        for a, b, c, d in rec:
            if len(rec) > 0:
                xc, yc = find_center_point(a, b, c, d)
                item.move_item(xc, yc)
                break
            break
        if not len(rec) > 0:
            break
        if item not in list:
           list.append(item)
        i = i+1
        if i > 3:
            break
            
        """
    """
    sorted_items = sorted(item_instances, key=lambda item: item.calculate_total_dimensions())
    list = []
    i = 0
    value = 0
    Algo_instance = Algo(container_instance, item_instances)

    for index, item in enumerate(sorted_items):
        x, y, flag = Algo_instance.plot(item, list)
        if flag is not False:
            list.append(item)
            value = value+item.value
        elif flag is False:
            break
        i = i+1
        print(i)
        if i == 10:
            break
    print("value", value)

    # Print the generated grid coordinates

    draw_instance = Draw(container_instance,list)
    draw_instance.plot()
    """


    Algo_instance = Algo50(container_instance, item_instances)

    Algo_instance.plot()


    """
    # Initialize Pygame
    pygame.init()

    # Constants
    SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
    TRIANGLE_SIZE = 20  # Adjust size as needed
    GRAVITY = 900
    BOUNCE_FACTOR = 0.5
    TIME_STEP = 1.0 / 60.0
    MIN_DISTANCE = TRIANGLE_SIZE

    # Initialize PyMunk
    space = pymunk.Space()
    space.gravity = (0, -GRAVITY)

    # Create a static ground body as the convex region
    convex_region = [(200, 200), (300, 100), (500, 100), (600, 200)]
    ground = pymunk.Poly(space.static_body, vertices=convex_region)
    space.add(ground)

    # Create a collision handler to handle collisions
    collision_handler = space.add_default_collision_handler()

    # Colors
    WHITE = (255, 255, 255)
    BLUE = (0, 0, 255)

    class Particle:
        def __init__(self, x, y):
            self.body = pymunk.Body(1, 1)  # Mass and moment of inertia
            self.body.position = x, y
            self.shape = pymunk.Poly.create_box(self.body, size=(TRIANGLE_SIZE, TRIANGLE_SIZE))
            self.shape.elasticity = BOUNCE_FACTOR
            space.add(self.body, self.shape)

        def apply_gravity(self):
            pass  # Gravity is handled by PyMunk

        def is_inside_convex(self):
            return True  # PyMunk handles containment within the convex region

    # Initialize the screen
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Particle Simulation")

    # Calculate the center of the convex region
    center_x = sum(x for x, _ in convex_region) / len(convex_region)
    center_y = sum(y for _, y in convex_region) / len(convex_region)

    # Create a list of particles (triangles) and position them at the center of the convex region
    particles = [Particle(center_x, center_y)]

    # Main loop
    running = True
    clock = pygame.time.Clock()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        for particle in particles:
            particle.apply_gravity()

        for particle in particles:
            particle.is_inside_convex()  # Ensure containment

        space.step(TIME_STEP)  # PyMunk automatically handles collisions

        # Draw the particles as triangles
        for particle in particles:
            x, y = int(particle.body.position.x), int(particle.body.position.y)
            points = particle.shape.get_vertices()
            transformed_points = [(p.x + x, p.y + y) for p in points]
            pygame.draw.polygon(screen, BLUE, transformed_points, 0)

        # Draw the convex region
        pygame.draw.polygon(screen, (0, 255, 0), convex_region, 1)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    """
    """
    # Create the rectangular polygon
    rectangle = Polygon([(0, 0), (0, 6), (8, 6), (8, 0), (0, 0)])

    # Create the triangular polygon inside the rectangular polygon with the tip touching the edge
    triangle = Polygon([(5, 4), (3, 6), (5, 3)])
    triangle = triangle.buffer(0.1)

    # Compute the difference to remove the triangular part from the rectangular polygon
    resulting_polygon = rectangle.difference(triangle)
    print(resulting_polygon)

    # Extract the x and y coordinates from the resulting polygon
    x, y = resulting_polygon.exterior.xy

    # Plot the original rectangle
    plt.plot(*rectangle.exterior.xy, label='Rectangle', color='blue')

    # Plot the original triangular polygon
    plt.plot(*triangle.exterior.xy, label='Triangle', color='green')

    # Plot the resulting polygon after the difference operation
    plt.fill(x, y, alpha=0.5, color='red', label='Result')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.show()
    
    """


    """
    algo 2: start
    
    i = 0
    temp = None
    First_Item = None
    pol1 = None
    list_of_items = []
    list_check = []
    mainitem = None
    center  = None
    # Example usage to sort a list of polygons
    sorted_polygons = sorted(item_instances, key=custom_polygon_sort, reverse=True)
    sorted_items = sorted(sorted_polygons, key=lambda item: item.calculate_regular_polygon_area(), reverse=False)

    for index, item in enumerate(sorted_items):
        # middle of the region
        if i == 0:
        # move first item to the middle of the region
            x, y = container_instance.calculate_centroid()
            item.move_item(x, y)
            list_of_items.append(item)
            list_check.append(item)
            First_Item = item
            temp = item
            mainitem = First_Item.coordinates
            center = (x,y)
            pol1 = Polygon(First_Item.coordinates)

        if i >= 1:
            flag = False
            z = 0
            list_of_edges = get_edge_lines(mainitem)
            sorted_edges = sorted(list_of_edges, key=lambda edge: distance_to_center(edge, center))
            for (p1, p2) in sorted_edges:
                x1, y1 = p1
                x2, y2 = p2
                j = 0
                for (q1, q2) in get_edge_lines(item.coordinates):
                    num_q1 = j
                    cord_of_current_item = item.move_to_point_value(j, x1, y1)
                    (t1, t2) = get_edge_lines(cord_of_current_item)[j]

                    shallow_copy = copy.copy(item)
                    temp.set_coordinates(mainitem)

                    list_check[0] = temp
                    shallow_copy.move_to_point(j, x1, y1)

                    list_check.append(shallow_copy)

                    pol2 = Polygon(cord_of_current_item)
                    # p1 ,p2
                    #(xey, yey) = get_edge_lines(mainitem)[z]

                    yes = False
                    pas = LineString([t1, t2])
                    inters = pas.intersection(pol1)
                    if pas.intersects(pol1):
                        if not inters.is_empty:
                            # Calculate the length of the intersection
                            intersection_len = inters.length
                            if intersection_len > 0:
                                yes = True
                    (xey, yey) = get_edge_lines(mainitem)[(z - 1) % len(get_edge_lines(mainitem))]
                    (dex, dey) = get_edge_lines(cord_of_current_item)[(j - 1) % len(cord_of_current_item)]
                    yes2 = False
                    po1 = None
                    po2 = None
                    po3 = None
                    po4 = None
                    if not yes:
                        l1 = LineString([p1, p2])
                        l2 = LineString([t1, t2])
                        epsilon = container_instance.calculate_distance_threshold()
                        dist = hausdorff_distance(l1, l2)
                        print("epsilon", epsilon, "dist", dist)
                        angle = calculate_slope_and_angle(p1, p2, t1, t2)
                        print("angle", angle)
                        print("points", p1, p2, t1, t2)
                        which_index = None
                        the_angle = None
                        the_dist = None
                        draw_instance = Draw(container_instance, list_check, p1, p2, t1, t2, None)
                        # draw_instance.plot()
                        if dist < epsilon and (angle < 0.01 or angle > 179) and edge_cover(p1, p2, t1,
                                                                                           t2) and are_edges_close(p1,
                                                                                                                   p2,
                                                                                                                   t1,
                                                                                                                   t2,
                                                                                                                   dist):
                            print("is point between", edge_cover(p1, p2, t1, t2))
                            print("edges are close", are_edges_close(p1, p2, t1, t2, dist))
                            yes2 = True
                            po1 = p1
                            po2 = p2
                            po3 = t1
                            po4 = t2
                            the_angle = angle
                            the_dist = dist
                            which_index = j
                        angle = calculate_slope_and_angle(p1, p2, dex, dey)
                        l1 = LineString([p1, p2])
                        l2 = LineString([dex, dey])
                        dist = hausdorff_distance(l1, l2)
                        print("epsilon", epsilon, "dist", dist)
                        print("angle", angle)
                        print("points", p1, p2, dex, dey)
                        draw_instance = Draw(container_instance, list_check, p1, p2, dex, dey, None)
                        # draw_instance.plot()
                        if dist < epsilon and (angle < 0.01 or angle > 179) and edge_cover(p1, p2, dex,
                                                                                           dey) and are_edges_close(p1,
                                                                                                                    p2,
                                                                                                                    dex,
                                                                                                                    dey,
                                                                                                                    dist):
                            print("is point between", edge_cover(p1, p2, dex, dey))
                            print("edges are close", are_edges_close(p1, p2, dex, dey, dist))
                            yes2 = True
                            po1 = p1
                            po2 = p2
                            po3 = dex
                            po4 = dey
                            the_angle = angle
                            the_dist = dist
                            which_index = j
                        l1 = LineString([xey, yey])
                        l2 = LineString([t1, t2])
                        dist = hausdorff_distance(l1, l2)
                        print("epsilon", epsilon, "dist", dist)
                        angle = calculate_slope_and_angle(xey, yey, t1, t2)
                        print("angle", angle)
                        print("points", xey, yey, t1, t2)
                        draw_instance = Draw(container_instance, list_check, xey, yey, t1, t2, None)
                        # draw_instance.plot()
                        if dist < epsilon and (angle < 0.01 or angle > 179) and edge_cover(xey, yey, t1,
                                                                                           t2) and are_edges_close(xey,
                                                                                                                   yey,
                                                                                                                   t1,
                                                                                                                   t2,
                                                                                                                   dist):
                            print("is point between", edge_cover(xey, yey, t1, t2))
                            print("edges are close", are_edges_close(xey, yey, t1, t2, dist))
                            yes2 = True
                            po1 = xey
                            po2 = yey
                            po3 = t1
                            po4 = t2
                            the_angle = angle
                            the_dist = dist
                            which_index = (j - 1) % len(cord_of_current_item)
                        l1 = LineString([xey, yey])
                        l2 = LineString([dex, dey])
                        dist = hausdorff_distance(l1, l2)
                        print("epsilon", epsilon, "dist", dist)
                        angle = calculate_slope_and_angle(xey, yey, dex, dey)
                        print("angle", angle)
                        print("points", xey, yey, dex, dey)
                        if dist < epsilon and (angle < 0.01 or angle > 179) and edge_cover(xey, yey, dex,
                                                                                           dey) and are_edges_close(xey,
                                                                                                                    yey,
                                                                                                                    dex,
                                                                                                                    dey,
                                                                                                                    dist):
                            print("is point between", edge_cover(xey, yey, dex, dey))
                            print("edges are close", are_edges_close(xey, yey, dex, dey, dist))
                            yes2 = True
                            po1 = xey
                            po2 = yey
                            po3 = dex
                            po4 = dey
                            the_angle = angle
                            the_dist = dist
                            which_index = (j - 1) % len(cord_of_current_item)



                    draw_instance = Draw(container_instance, list_check, xey, yey, dex, dey,None)
                    #draw_instance.plot()



                    list_check.pop()
                    print("end")



                    pol3 = Polygon(container_instance.coordinates)
                    #if not pol1.crosses(pol2) and pol1.touches(pol2) and pol2.within(pol3):
                    if not pol1.crosses(pol2) and pol1.touches(pol2) and pol2.within(pol3):
                        if yes:
                            item.move_to_point(j, x1, y1)
                            list_check.append(shallow_copy)
                            pol2 = Polygon(item.coordinates)
                            list_of_items.append(item)
                            mergedPolys = unary_union([pol1, pol2])
                            exterior_coords_list = []
                            if isinstance(mergedPolys, MultiPolygon):
                                # Iterate through the constituent polygons
                                for polygon in mergedPolys.geoms:
                                    # Get the coordinates of the exterior boundary of each polygon
                                    exterior_coords = list(polygon.exterior.coords)
                                    # Append them to the exterior_coords_list
                                    exterior_coords_list.extend(exterior_coords)
                            else:
                                # If it's a single Polygon, get its exterior coordinates directly
                                exterior_coords_list = list(mergedPolys.exterior.coords)

                            exterior_coords_list = remove_duplicate_coordinates(exterior_coords_list)

                            pol1 = Polygon(exterior_coords_list)
                            mainitem = exterior_coords_list
                            draw_instance = Draw(container_instance, list_check, (1,1), (1,1), (1,1), (1,1), None)
                            # print("inside:", "angle", the_angle, "dist:", the_dist)

                            draw_instance.plot()
                            list_check.pop()

                            flag = True
                            break
                        if yes2:
                            item.move_to_point(j, x1, y1)
                            list_check.append(shallow_copy)
                            pol2 = Polygon(item.coordinates)
                            list_of_items.append(item)
                            counter = order_coordinates_counterclockwise([po1, po2, po3, po4])
                            pol4 = Polygon(counter)


                            mergedPolys1 = unary_union([pol1, pol4])

                            #mergedPolys = unary_union([mergedPolys1, pol2])
                            exterior_coords_list1 = []
                            if isinstance(mergedPolys1, MultiPolygon):
                                # Iterate through the constituent polygons
                                for polygon in mergedPolys1.geoms:
                                    # Get the coordinates of the exterior boundary of each polygon
                                    exterior_coords = list(polygon.exterior.coords)
                                    # Append them to the exterior_coords_list
                                    exterior_coords_list1.extend(exterior_coords)
                            else:
                                # If it's a single Polygon, get its exterior coordinates directly
                                exterior_coords_list1 = list(mergedPolys1.exterior.coords)
                            exterior_coords_list1 = remove_duplicate_coordinates(exterior_coords_list1)


                            mergedPolys = unary_union([Polygon(exterior_coords_list1), pol2])
                            
                            exterior_coords_list = []
                            if isinstance(mergedPolys, MultiPolygon):
                                # Iterate through the constituent polygons
                                for polygon in mergedPolys.geoms:
                                    # Get the coordinates of the exterior boundary of each polygon
                                    exterior_coords = list(polygon.exterior.coords)
                                    # Append them to the exterior_coords_list
                                    exterior_coords_list.extend(exterior_coords)
                            else:
                                # If it's a single Polygon, get its exterior coordinates directly
                                exterior_coords_list = list(mergedPolys.exterior.coords)


                            exterior_coords_list = remove_duplicate_coordinates(exterior_coords_list)

                            pol1 = Polygon(exterior_coords_list)
                            mainitem = exterior_coords_list
                            draw_instance = Draw(container_instance, list_check, po1, po2, po3, po4, None)
                            #print("inside:", "angle", the_angle, "dist:", the_dist)

                            draw_instance.plot()
                            list_check.pop()



                            flag = True
                            break

                    j = j + 1

                if flag:
                    break
                z = z+1
        print(i)
        if i == 50:
            break
        i = i+1
    draw_instance = Draw(container_instance, list_of_items,(1,1), (1,1), (1,1), (1,1), None)
    draw_instance.plot()
     finish """















if __name__ == "__main__":
    main()
