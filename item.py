
import math
from shapely.geometry import Polygon

class Item:
    def __init__(self, quantity, value, x_coordinates, y_coordinates):
        self.coordinates = list(zip(x_coordinates, y_coordinates))
        self.quantity = quantity
        self.value = value
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.max_x = max(x_coordinates)
        self.max_y = max(y_coordinates)
        self.min_x = min(x_coordinates)
        self.min_y = min(y_coordinates)
        self.ext_size = None
        self.ext_size_for_loop = None
        self.ext_size_for_loop_rec = None

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.x_coordinates, self.y_coordinates = zip(*coordinates)  # Unpack and separate x and y coordinates
        self.x_coordinates = list(self.x_coordinates)
        self.y_coordinates = list(self.y_coordinates)
        self.max_x = max(self.x_coordinates)
        self.max_y = max(self.y_coordinates)
        self.min_x = min(self.x_coordinates)
        self.min_y = min(self.y_coordinates)

    def get_largest_dimension(self):
        # Calculate the width of the object by subtracting the minimum x value from the maximum x value
        width = self.max_x - self.min_x

        # Calculate the height of the object by subtracting the minimum y value from the maximum y value
        height = self.max_y - self.min_y

        # Determine the largest dimension (either width or height) by using the max function
        total_dimensions = max(width, height)

        # Return the largest dimension
        return total_dimensions

    def set_translation_by_center(self, new_center_x, new_center_y):
        # Calculate the current center point of the item
        convex_polygon = Polygon(self.coordinates)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid

        current_center_x = round(centroid.x)
        current_center_y = round(centroid.y)
        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coordinates = [x + translation_x for x in self.x_coordinates]
        self.y_coordinates = [y + translation_y for y in self.y_coordinates]
        self.max_x = max(self.x_coordinates)
        self.max_y = max(self.y_coordinates)
        self.min_x = min(self.x_coordinates)
        self.min_y = min(self.y_coordinates)

    def set_translation_by_point(self, point_of_pol, point_of_region):
        # Calculate the translation vector
        translation_x = point_of_region[0] - point_of_pol[0]
        translation_y = point_of_region[1] - point_of_pol[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update x and y coordinates separately
        self.x_coordinates = [x + translation_x for x in self.x_coordinates]
        self.y_coordinates = [y + translation_y for y in self.y_coordinates]

        # Update the bounding box
        self.max_x += translation_x
        self.min_x += translation_x
        self.max_y += translation_y
        self.min_y += translation_y

        # Update the new position
        self.x = point_of_region[0]
        self.y = point_of_region[1]

    def set_translation_by_distance_and_angle(self, distance, angle_degrees):
        # Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)

        # Calculate the current center point of the item
        convex_polygon = Polygon(self.coordinates)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid
        current_center_x = centroid.x
        current_center_y = centroid.y

        # Calculate the new position using polar coordinates
        new_center_x = current_center_x + distance * math.cos(angle_radians)
        new_center_y = current_center_y + distance * math.sin(angle_radians)

        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coordinates = [x + translation_x for x in self.x_coordinates]
        self.y_coordinates = [y + translation_y for y in self.y_coordinates]
        self.max_x = max(self.x_coordinates)
        self.max_y = max(self.y_coordinates)
        self.min_x = min(self.x_coordinates)
        self.min_y = min(self.y_coordinates)

    def convert_coordinates_to_int(self):
        int_coordinates = [(round(x), round(y)) for x, y in self.coordinates]
        return int_coordinates



