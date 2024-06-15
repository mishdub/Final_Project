from Draw import Draw
import math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
import random

from shapely.ops import unary_union
import time
import warnings
import copy
from PolygonTangentFinder import PolygonTangentFinder


class Algo46:

    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances
        self.error_occurred = False  # Initialize error_occurred as False

    def find_cells(self, polygon, row_list, col_list):
        container_x_min, container_x_max = min(polygon.x_coords), max(polygon.x_coords)
        container_y_min, container_y_max = min(polygon.y_coords), max(polygon.y_coords)

        grid_cols = int(container_x_max - container_x_min)

        grid_rows = int(container_y_max - container_y_min)

        ret_row_list = row_list
        ret_col_list = col_list

        for row in range(grid_rows):  # Iterate through each row
            for col in range(grid_cols):  # Iterate through each column
                # Calculate cell boundaries for a 1x1 cell
                cell_x_min = container_x_min + col
                cell_x_max = cell_x_min + 1
                cell_y_min = container_y_min + row
                cell_y_max = cell_y_min + 1

                # Define the corners of the rectangle
                point1 = (cell_x_min, cell_y_min)  # Bottom-left
                point2 = (cell_x_min + 1, cell_y_min)  # Bottom-right
                point3 = (cell_x_min + 1, cell_y_min + 1)  # Top-right
                point4 = (cell_x_min, cell_y_min + 1)  # Top-left

                # Create a list of points defining the rectangle
                rectangle = [point1, point2, point3, point4, point1]

                if (Polygon(polygon.coordinates)).contains((Polygon(rectangle))):
                    ret_row_list.remove(row)
                    ret_col_list.remove(col)

        return ret_row_list, ret_col_list

    def plot(self):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")
        container_x_min, container_x_max = min(self.container_instance.x_coords), max(self.container_instance.x_coords)
        container_y_min, container_y_max = min(self.container_instance.y_coords), max(self.container_instance.y_coords)

        grid_cols = int(container_x_max - container_x_min)

        grid_rows = int(container_y_max - container_y_min)

        rows_list = list(range(grid_rows))
        cols_list = list(range(grid_cols))


        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        convex_region = self.container_instance.coordinates
        list_of_polygons = []

        for dex, polygon in enumerate(sorted_items):
            flag = False
            for row in rows_list:  # Iterate through each row
                for col in cols_list:  # Iterate through each column
                    print("test")
                    # Calculate cell boundaries for a 1x1 cell
                    cell_x_min = container_x_min + col
                    cell_x_max = cell_x_min + 1
                    cell_y_min = container_y_min + row
                    cell_y_max = cell_y_min + 1

                    # Optionally generate a random point within the cell, or directly use the cell itself
                    x = random.uniform(cell_x_min, cell_x_max)
                    y = random.uniform(cell_y_min, cell_y_max)

                    # Move the item to the generated coordinates
                    item_coords = polygon.move_item_value(x, y)
                    polygon.move_item(x, y)
                    list_of_polygons.append(polygon)
                    draw_instance = Draw(self.container_instance, list_of_polygons, (1, 1), (1, 1), (1, 1), (1, 1),
                                         None,
                                         None,
                                         None, None)
                    draw_instance.plot()

                    if Polygon(item_coords).within(Polygon(convex_region)):
                        polygon.move_item(x, y)
                        list_of_polygons.append(polygon)
                        draw_instance = Draw(self.container_instance, list_of_polygons, (1, 1), (1, 1), (1, 1), (1, 1),
                                             None,
                                             None,
                                             None, None)
                        draw_instance.plot()

                        rows_list, cols_list = self.find_cells(polygon, rows_list, cols_list)
                        flag = True
                        break
                if flag:
                    break







