import random
import time
from scipy.spatial import Voronoi
from shapely.geometry import Polygon
import numpy as np
from shapely.geometry import Polygon, LineString
from scipy.optimize import minimize
import random
import math
from Draw import Draw
import copy




class Algo7:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def calculate_rectangle_diagonal(self,coordinates):
        # Assuming the list of coordinates contains tuples of (x, y) for the four vertices
        if len(coordinates) != 4:
            return "Input must contain exactly 4 coordinates."

        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]
        x4, y4 = coordinates[3]

        diagonal_length = math.sqrt((x1 - x3) ** 2 + (y1 - y3) ** 2)
        return diagonal_length

    def find_intersection_point(self, polygon_coordinates, line_coords):
        # Create a polygon from the given coordinates

        polygon = Polygon(polygon_coordinates)

        exterior_ring = polygon.exterior

        # Create a LineString from the given line coordinates
        line = LineString(line_coords)

        # Find the intersection between the line and the polygon
        intersection = line.intersection(exterior_ring)

        # Check the type of the result to handle different cases
        if intersection.is_empty:

            return None  # No intersection
        elif intersection.geom_type == 'Point':
            # Only one intersection point
            return (intersection.x, intersection.y)
        elif intersection.geom_type == 'MultiPoint':
            # Multiple intersection points
            intersection_points = [(point.x, point.y) for point in intersection]
            return intersection_points
        else:
            # Other types of intersections (e.g., LineString, MultiLineString)
            return None  # Handle as needed
    def calculate_centroid(self,coordinates):
        if not coordinates:
            return None  # Handle an empty set of coordinates gracefully

        # Calculate the average of x-coordinates
        x_sum = sum(x for x, y in coordinates)
        x_avg = x_sum / len(coordinates)

        # Calculate the average of y-coordinates
        y_sum = sum(y for x, y in coordinates)
        y_avg = y_sum / len(coordinates)

        return x_avg, y_avg

    def calculate_angle(self, point, centroid):
        return (math.atan2(point[1] - centroid[1], point[0] - centroid[0]) + 2 * math.pi) % (2 * math.pi)

    def order_coordinates_counterclockwise(self,coordinates):
        # Calculate the centroid
        centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
        centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
        centroid = (centroid_x, centroid_y)

        # Sort the coordinates based on angles
        sorted_coordinates = sorted(coordinates, key=lambda point: self.calculate_angle(point, centroid))

        return sorted_coordinates

    def bounding_rectangle(self,polygon_coordinates):
        if not polygon_coordinates:
            return None  # Handle an empty input gracefully

        # Initialize min and max coordinates with the first vertex
        min_x, min_y = polygon_coordinates[0]
        max_x, max_y = polygon_coordinates[0]

        # Find the min and max coordinates
        for x, y in polygon_coordinates:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # Calculate the coordinates of the rectangle's corners
        bottom_left = (min_x, min_y)
        top_left = (min_x, max_y)
        top_right = (max_x, max_y)
        bottom_right = (max_x, min_y)

        # Return the rectangle's coordinates
        return [bottom_left, top_left, top_right, bottom_right]

    def calculate_endpoint_from_direction(self, x1, y1, dx, dy, length):
        # Calculate the end point
        x2 = x1 + length * dx
        y2 = y1 + length * dy

        return x2, y2

    def calculate_width_and_height(self, coordinates):
        if len(coordinates) < 2:
            raise ValueError("A polygon must have at least 2 vertices to calculate width and height.")

        # Initialize with the coordinates of the first vertex.
        min_x, max_x = coordinates[0][0], coordinates[0][0]
        min_y, max_y = coordinates[0][1], coordinates[0][1]

        # Iterate through the remaining vertices to find the bounding box.
        for x, y in coordinates:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

        width = max_x - min_x
        height = max_y - min_y

        return width, height

    def plot(self):
        x, y = self.container_instance.calculate_centroid()
        coordinates = None
        the_list = []
        List = []
        dime = self.container_instance.calculate_total_dimensions()
        random_direction_angle = 0
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        for dex, polygon in enumerate(sorted_items):
            copied3 = copy.deepcopy(polygon)
            polygon.box()
            polygon.move_item(x, y)
            copied3.move_item(x, y)
            if dex == 10:
                break
            if dex == 0:
                coordinates = polygon.coordinates
                the_list.append(polygon)
                List.append(copied3)
            elif dex > 0:
                vx, vy = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

                x2, y2 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime/2)
                point = self.find_intersection_point(coordinates, [(x, y), (x2, y2)])

                if point is not None:
                    x3, y3 = point
                    dis = None
                    width, height = (self.calculate_width_and_height(polygon.coordinates))
                    slant = self.calculate_rectangle_diagonal(coordinates)
                    if random_direction_angle == 0 or random_direction_angle == 180:
                        dis = width
                    if random_direction_angle == 90 or random_direction_angle == 270:
                        dis = height
                    if random_direction_angle == 45 or random_direction_angle == 135 or random_direction_angle == 225 or random_direction_angle == 315:
                        dis = slant

                    x4, y4 = self.calculate_endpoint_from_direction(x3, y3, vx, vy, dis/2)

                    polygon.move_item(x4, y4)
                    copied3.move_item(x4, y4)
                    random_direction_angle = (random_direction_angle + 45) % 360
                else:
                    print("point is None")
                    break

                coordinates = coordinates + polygon.coordinates
                coordinates = self.bounding_rectangle(coordinates)

                copied = copy.deepcopy(polygon)
                copied.set_coordinates(self.bounding_rectangle(coordinates))
                the_list.append(polygon)
                List.append(copied3)
                draw_instance = Draw(self.container_instance, the_list, (x,y), (x2,y2), (1, 1), (1, 1), None, None,
                                     None)
                draw_instance.plot()
                li = []
                li.append(copied)
                draw_instance = Draw(self.container_instance,li, (1, 1), (1, 1), (1, 1), (1, 1), None, None,
                                     None)
                draw_instance.plot()
                li.pop()

        draw_instance = Draw(self.container_instance, List, (1, 1), (1, 1), (1, 1), (1, 1), None, None,
                             None)
        draw_instance.plot()








