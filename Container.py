import random
import math
import copy
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString, MultiLineString



class Container:
    def __init__(self, x_coordinates, y_coordinates):
        self.coordinates = list(zip(x_coordinates, y_coordinates))
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.max_x = max(x_coordinates)
        self.max_y = max(y_coordinates)
        self.min_x = min(x_coordinates)
        self.min_y = min(y_coordinates)


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







