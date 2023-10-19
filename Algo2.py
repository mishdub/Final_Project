import random
import copy
from Draw import Draw
import math
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString
import time
import numpy as np

class Algo2:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def find_closest_point_to_convex_region(self, polygon, direction):
        def distance(point1, point2):
            return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

        # Initialize variables to keep track of the closest point and its distance
        closest_point = polygon.coordinates[0]
        closest_distance = distance(polygon.coordinates[0], self.container_instance.coordinates[0])  # Initialize with a large value

        # Find the point in the polygon closest to the convex region in the given direction

        for polygon_point in polygon.coordinates:
            for convex_point in self.container_instance.coordinates:
                dist = distance(polygon_point, convex_point)
                if dist < closest_distance:
                    closest_distance = dist
                    closest_point = polygon_point

        # Calculate a back end-point of the polygon within the polygon
        back_end_point = polygon.coordinates[0]
        farthest_distance = 0

        # Calculate the back end-point of the polygon in the given direction
        j = 0
        index = 0
        for polygon_point in polygon.coordinates:
            for convex_point in self.container_instance.coordinates:
                dist = distance(polygon_point, convex_point)
                if dist > farthest_distance:
                    farthest_distance = dist
                    back_end_point = polygon_point
                    index = j
            j = j+1

        return closest_point, back_end_point

    def find_closest_point_to_point(self, polygon, target_point):
        # Initialize variables to keep track of the closest point and its distance
        closest_point = polygon.coordinates[0]
        closest_distance = math.dist(polygon.coordinates[0], target_point)  # Using the 'math.dist' function

        # Find the point in the polygon closest to the target point
        j = 0
        index = 0
        for polygon_point in polygon.coordinates:
            distance_to_target = math.dist(polygon_point, target_point)
            if distance_to_target < closest_distance:
                closest_distance = distance_to_target
                closest_point = polygon_point
                index = j
            j = j + 1

        return index,closest_point

    def place_polygon_closest_to_boundary(self, polygon):
        # Generate a random direction angle between 0 and 360 degrees
        random_direction_angle = random.uniform(0, 360)
        random_direction_vector = (
        math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

        # Start at the middle point
        current_point = self.container_instance.calculate_centroid()

        # Move in the random direction while checking for intersections with the convex region
        print(random_direction_angle)
        while True:
            list = []
            move_distance = math.sqrt(polygon.calculate_regular_polygon_area())
            next_point = (current_point[0] + random_direction_vector[0] * move_distance,
                          current_point[1] + random_direction_vector[1] * move_distance)
            x, y = next_point
            print(x,y)

            pol_coords = polygon.move_item_value(x, y)
            copied = copy.deepcopy(polygon)
            copied.move_item(x, y)
            list.append(copied)
            draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
            draw_instance.plot()
            pol = Polygon(pol_coords)
            convex_region = Polygon(self.container_instance.coordinates)

            if pol.within(convex_region):
                current_point = next_point
            else:
                # Stop when we've reached the end of the convex region or gone outside
                x1, y1 = current_point
                polygon.move_item(x1, y1)
                break

    def place_polygon_closest_to_boundary2(self, polygon):
        # Generate a random direction angle between 0 and 360 degrees
        random_direction_angle = random.uniform(0, 360)
        random_direction_vector = (
            math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

        # Start at the middle point of the convex region
        current_point = self.container_instance.calculate_centroid()

        while True:
            list = []
            # Find the farthest point in the convex region in the given direction
            farthest_point = current_point
            max_projection = 0

            for convex_point in self.container_instance.coordinates:
                direction_vector = (convex_point[0] - current_point[0], convex_point[1] - current_point[1])
                projection = direction_vector[0] * random_direction_vector[0] + direction_vector[1] * \
                             random_direction_vector[1]

                if projection > max_projection:
                    max_projection = projection
                    farthest_point = convex_point

            #find the closest point in the polygon to the point in the convex
            index, point = self.find_closest_point_to_point(polygon, farthest_point)

            # Place the polygon at the farthest point
            pol_coords = polygon.move_to_point_value(index, farthest_point[0], farthest_point[1])
            copied = copy.deepcopy(polygon)
            copied.move_to_point(index, farthest_point[0], farthest_point[1])
            list.append(copied)
            draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
            draw_instance.plot()

            # Check if the polygon is inside the convex region
            pol = Polygon(pol_coords)
            convex_region = Polygon(self.container_instance.coordinates)

            if pol.within(convex_region):
                break  # Placement is successful

            # Move the polygon towards the center while maintaining the direction
            center = self.container_instance.calculate_centroid()
            direction_vector = (center[0] - farthest_point[0], center[1] - farthest_point[1])
            direction_magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
            move_distance = 0.1 * direction_magnitude  # Adjust as needed

            current_point = (farthest_point[0] + (direction_vector[0] / direction_magnitude) * move_distance,
                             farthest_point[1] + (direction_vector[1] / direction_magnitude) * move_distance)

        # Finalize the placement
        x1, y1 = farthest_point
        polygon.move_to_point(index, x1, y1)

    def plot(self):
        i = 0
        list = []
        for index, item in enumerate(self.item_instances):
            self.place_polygon_closest_to_boundary2(item)
            list.append(item)
            i = i+1
            if i == 1:
                break
        draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
        draw_instance.plot()



