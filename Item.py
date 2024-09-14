
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

    def calculate_width(self):
        """
        Calculate and return the width of the item based on its coordinates.

        Returns:
        - The width of the item.
        """
        return self.max_x - self.min_x

    def calculate_height(self):
        """
        Calculate and return the height of the item based on its coordinates.

        Returns:
        - The height of the item.
        """
        return self.max_y - self.min_y

    def calculate_total_dimensions(self):
        """
        Calculate and return the total dimensions (area) of the item based on its coordinates.

        Returns:
        - The total dimensions (area) of the item.
        """
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        total_dimensions = max(width, height)
        return total_dimensions

    def calculate_thinness(self):
        """
        Calculate and return the total dimensions (area) of the item based on its coordinates.

        Returns:
        - The total dimensions (area) of the item.
        """
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        return width / height

    def calculate_centroid(self):
        num_points = len(self.coordinates)
        if num_points == 0:
            return None

        centroid_x = sum(self.x_coordinates) / num_points
        centroid_y = sum(self.y_coordinates) / num_points
        return centroid_x, centroid_y

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

    def convert_coords_to_int_self(self):
        """
        Converts a list of coordinates from float to integer by rounding.

        Parameters:
        - coords: List of tuples, where each tuple contains float coordinates.

        Returns:
        - A list of tuples with integer coordinates.
        """
        int_coordinates = [(round(x), round(y)) for x, y in self.coordinates]
        return int_coordinates



