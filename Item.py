from shapely import box
import math
from shapely.geometry import Polygon


class Item:
    def __init__(self, quantity, value, x_coords, y_coords):
        self.coordinates = list(zip(x_coords, y_coords))
        self.quantity = quantity
        self.value = value
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.max_x = max(x_coords)
        self.max_y = max(y_coords)
        self.min_x = min(x_coords)
        self.min_y = min(y_coords)
        self.ex_dime = None
        self.x = None
        self.y = None
        self.Box = None
        self.left_point = None
        self.right_point = None
        self.left_list = None
        self.curr_angle = None
        self.left_intersection_point = None
        self.right_intersection_point = None
        self.extended_pol = None
        self.extended_polygon = None
        self.left_line = None
        self.right_line = None
        self.the_point = None
        self.leftline = None
        self.rightline = None
        self.tri_index = None
        self.polygon_var2 =None
        self.left_point_temp = None
        self.p_point = None
        self.sign = None





    def extended_dime(self, coordinates):
        if coordinates:
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

            return min(width, height)



    def calculate_regular_polygon_area(self):
        if len(self.coordinates) < 3:
            return "A polygon must have at least 3 coordinates."

        # Calculate the center of the polygon
        x_coords, y_coords = zip(*self.coordinates)
        center_x = sum(x_coords) / len(self.coordinates)
        center_y = sum(y_coords) / len(self.coordinates)

        # Calculate the distance from the center to a vertex as the radius
        radius = math.sqrt((center_x - self.coordinates[0][0]) ** 2 + (center_y - self.coordinates[0][1]) ** 2)

        # Determine the number of sides in the polygon
        num_sides = len(self.coordinates)

        # Calculate the area using the regular polygon area formula
        area = (0.5 * num_sides * radius ** 2 * math.sin(2 * math.pi / num_sides))

        return area


    def get_size(self):
        # Calculate the size of the polygon (e.g., area, perimeter, or bounding box size)
        # Here's a simple example for a polygon represented by a list of (x, y) points:

        # Calculate the area of the polygon using the shoelace formula
        n = len(self.coordinates)
        area = 0
        for i in range(n):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[(i + 1) % n]
            area += (x1 * y2 - x2 * y1)
        area = abs(area) / 2.0

        # You can return the area as the "size" of the polygon
        return area
    def calculate_distance_threshold(self):
        # Find the diagonal distance of the region

        dx = self.max_x - self.min_x
        dy = self.max_y - self.min_y
        max_distance = math.sqrt(dx ** 2 + dy ** 2)
        """
        width = self.max_x - self.min_x
        height = self.max_y - self.min_y
        max_distance = max(width, height)
        """

        # Calculate the distance_threshold based on the proximity factor
        distance_threshold = max_distance * 0.1
        return distance_threshold

    def set_coordinates(self, coordinates):
        self.coordinates = coordinates
        self.x_coords, self.y_coords = zip(*coordinates)  # Unpack and separate x and y coordinates
        # Convert x_coords and y_coords to lists
        self.x_coords = list(self.x_coords)
        self.y_coords = list(self.y_coords)
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def change_point(self, index, new_x, new_y):
        if 0 <= index < len(self.coordinates):
            self.x_coords[index] = new_x
            self.y_coords[index] = new_y
            self.coordinates[index] = (new_x, new_y)
            self.max_x = max(self.x_coords)
            self.max_y = max(self.y_coords)
            self.min_x = min(self.x_coords)
            self.min_y = min(self.y_coords)
        else:
            print("Invalid index")

    def box(self):
        b = box(self.min_x, self.min_y, self.max_x, self.max_y, False)
        self.Box = b
        new_coordinates = list(b.exterior.coords)
        self.x_coords = [coord[0] for coord in new_coordinates]
        self.y_coords = [coord[1] for coord in new_coordinates]
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)
        self.coordinates = new_coordinates[:-1]

    def returnbox(self):
        return self.Box



    def get_edge_lines(self):
        edges = []
        num_points = len(self.coordinates)

        for i in range(num_points):
            point1 = self.coordinates[i]
            point2 = self.coordinates[(i + 1) % num_points]  # Wrap around to the first point

            line = (point1, point2)
            edges.append(line)

        return edges



    def is_point_inside(self, point_x, point_y):
        num_vertices = len(self.coordinates)
        is_inside = False

        for i in range(num_vertices):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[(i + 1) % num_vertices]

            if (
                    (y1 <= point_y < y2 or y2 <= point_y < y1)
                    and point_x <= (x2 - x1) * (point_y - y1) / (y2 - y1) + x1
            ):
                is_inside = not is_inside

        return is_inside


    def point_inside_item(self, point):
        """
        Check if a point is inside the item.

        Args:
        - point: (x, y) tuple representing the point to check.

        Returns:
        - True if the point is inside the container, False otherwise.
        """
        x, y = point
        n = len(self.coordinates)
        inside = False

        for i in range(n):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[(i + 1) % n]

            if (y1 < y <= y2 or y2 < y <= y1) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
                inside = not inside

        return inside

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

    def reaches_to_rec(self):
        return abs((self.max_x - self.min_x) - (self.max_y - self.min_y))


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

    def find_point_with_max_x(self):
        max_x_point = max(self.coordinates, key=lambda point: point[0])
        return max_x_point[0], max_x_point[1]  # Return x and y coordinates

    def find_point_with_max_y(self):
        max_y_point = max(self.coordinates, key=lambda point: point[1])
        return max_y_point[0], max_y_point[1]  # Return x and y coordinates

    def find_point_with_min_x(self):
        min_x_point = min(self.coordinates, key=lambda point: point[0])
        return min_x_point[0], min_x_point[1]  # Return x and y coordinates

    def find_point_with_min_y(self):
        min_y_point = min(self.coordinates, key=lambda point: point[1])
        return min_y_point[0], min_y_point[1]  # Return x and y coordinates

    def move_item1(self, old_x, old_y, new_x, new_y):
        translation_x = new_x - old_x
        translation_y = new_y - old_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]



    def move_item(self, new_center_x, new_center_y):
        # Calculate the current center point of the item
        convex_polygon = Polygon(self.coordinates)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid
        current_center_x = centroid.x
        current_center_y = centroid.y
        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_leftmost_maxy(self, new_leftmost_x, new_leftmost_y):
        # Find all leftmost points with the same x-coordinate
        leftmost_points = [(x, y) for x, y in zip(self.x_coords, self.y_coords) if x == min(self.x_coords)]

        # Find the leftmost point with the highest y-coordinate
        current_leftmost_x, current_leftmost_y = max(leftmost_points, key=lambda point: point[1])

        # Calculate the translation vectors to move the leftmost point to the new position
        translation_x = new_leftmost_x - current_leftmost_x
        translation_y = new_leftmost_y - current_leftmost_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_leftmost_miny(self, new_leftmost_x, new_leftmost_y):
        # Find all leftmost points with the same x-coordinate
        leftmost_points = [(x, y) for x, y in zip(self.x_coords, self.y_coords) if x == min(self.x_coords)]

        # Find the leftmost point with the highest y-coordinate
        current_leftmost_x, current_leftmost_y = min(leftmost_points, key=lambda point: point[1])

        # Calculate the translation vectors to move the leftmost point to the new position
        translation_x = new_leftmost_x - current_leftmost_x
        translation_y = new_leftmost_y - current_leftmost_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_rightmost_maxy(self, new_rightmost_x, new_rightmost_y):
        # Find all rightmost points with the same x-coordinate
        rightmost_points = [(x, y) for x, y in zip(self.x_coords, self.y_coords) if x == max(self.x_coords)]

        # Find the rightmost point with the highest y-coordinate
        current_rightmost_x, current_rightmost_y = max(rightmost_points, key=lambda point: point[1])

        # Calculate the translation vectors to move the rightmost point to the new position
        translation_x = new_rightmost_x - current_rightmost_x
        translation_y = new_rightmost_y - current_rightmost_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_rightmost_miny(self, new_rightmost_x, new_rightmost_y):
        # Find all rightmost points with the same x-coordinate
        rightmost_points = [(x, y) for x, y in zip(self.x_coords, self.y_coords) if x == max(self.x_coords)]

        # Find the rightmost point with the highest y-coordinate
        current_rightmost_x, current_rightmost_y = min(rightmost_points, key=lambda point: point[1])

        # Calculate the translation vectors to move the rightmost point to the new position
        translation_x = new_rightmost_x - current_rightmost_x
        translation_y = new_rightmost_y - current_rightmost_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_maxy(self, new_x, new_y):
        # Find the point with the maximum y-coordinate within the polygon
        maxy_point = max(self.coordinates, key=lambda point: point[1])

        # Calculate the translation vectors to move the polygon to the new position
        translation_x = new_x - maxy_point[0]
        translation_y = new_y - maxy_point[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update the x_coords and y_coords lists
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_miny(self, new_x, new_y):
        # Find the point with the maximum y-coordinate within the polygon
        maxy_point = min(self.coordinates, key=lambda point: point[1])

        # Calculate the translation vectors to move the polygon to the new position
        translation_x = new_x - maxy_point[0]
        translation_y = new_y - maxy_point[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update the x_coords and y_coords lists
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_maxx(self, new_x, new_y):
        # Find the point with the maximum x-coordinate within the polygon
        maxx_point = max(self.coordinates, key=lambda point: point[0])

        # Calculate the translation vectors to move the polygon to the new position
        translation_x = new_x - maxx_point[0]
        translation_y = new_y - maxx_point[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update the x_coords and y_coords lists
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_minx(self, new_x, new_y):
        # Find the point with the maximum x-coordinate within the polygon
        maxx_point = min(self.coordinates, key=lambda point: point[0])

        # Calculate the translation vectors to move the polygon to the new position
        translation_x = new_x - maxx_point[0]
        translation_y = new_y - maxx_point[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update the x_coords and y_coords lists
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)

    def move_item_to_leftmost(self, new_center_x, new_center_y):
        # Calculate the current center point of the item
        current_center_x = (self.max_x + self.min_x) / 2
        current_center_y = (self.max_y + self.min_y) / 2

        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Calculate the leftmost x-coordinate of the item
        leftmost_x = min(self.x_coords)

        # Calculate the new x-coordinate for the leftmost side
        new_leftmost_x = leftmost_x + translation_x

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update max and min coordinates
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)
        # Adjust the coordinates to move the leftmost side to the new position
        self.coordinates = [(x - (new_leftmost_x - self.min_x), y) for x, y in self.coordinates]
        self.x_coords = [x - (new_leftmost_x - self.min_x) for x in self.x_coords]
        self.min_x = new_leftmost_x

    def move_item_value(self, new_center_x, new_center_y):
        convex_polygon = Polygon(self.coordinates)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid
        current_center_x = centroid.x
        current_center_y = centroid.y

        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]


    def edge_name(self, i):
        if i == len(self.coordinates) - 1:
            return f"Edge_{len(self.coordinates)}_1"
        else:
            return f"Edge_{i + 1}_{i + 2}"

    def slope_list(self):
        slopes = {}
        for i in range(len(self.coordinates)):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[(i + 1) % len(self.coordinates)]
            if x2 - x1 == 0:
                # For vertical lines, set the slope to -0
                slope = -0.0
            elif y2 - y1 == 0:
                # For horizontal lines, set the slope to 0
                slope = 0.0
            else:
                slope = (y2 - y1) / (x2 - x1)

            edge = self.edge_name(i)
            slopes[edge] = slope
        return slopes

    def calculate_line_equations(self):
        equations = {}
        for edge, slope in self.slope_list().items():
            if isinstance(slope, float) and slope != 0.0:
                i = int(edge.split('_')[1]) - 1
                x1, y1 = self.coordinates[i]
                y_intercept = y1 - slope * x1
                equation = f"y = {slope:.2f}x + {y_intercept:.2f}"
            else:
                equation = "Vertical Line" if slope == -0.0 else "Horizontal Line"
            equations[edge] = equation
        return equations

    def calculate_centroid(self):
        num_points = len(self.coordinates)
        if num_points == 0:
            return None

        centroid_x = sum(self.x_coords) / num_points
        centroid_y = sum(self.y_coords) / num_points
        return centroid_x, centroid_y

    def move_bottom_left(self, new_top_left_x, new_top_left_y):
        # Calculate the translation vector
        translation_x = new_top_left_x - self.min_x
        translation_y = new_top_left_y - self.min_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.min_x += translation_x
        self.max_x += translation_x
        self.min_y += translation_y
        self.max_y += translation_y
        self.box()

    def move_bottom_left_value(self, new_top_left_x, new_top_left_y):
        translation_x = new_top_left_x - self.min_x
        translation_y = new_top_left_y - self.min_y

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]


    def move_bottom_right(self, new_top_right_x, new_top_right_y):
        # Calculate the translation vector
        translation_x = new_top_right_x - self.max_x
        translation_y = new_top_right_y - self.min_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.max_x += translation_x
        self.min_y += translation_y
        self.max_y += translation_y
        self.box()

    def move_bottom_right_value(self, new_top_right_x, new_top_right_y):
        translation_x = new_top_right_x - self.max_x
        translation_y = new_top_right_y - self.min_y

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]


    def move_top_left(self, new_bottom_left_x, new_bottom_left_y):
        # Calculate the translation vector
        translation_x = new_bottom_left_x - self.min_x
        translation_y = new_bottom_left_y - self.max_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.min_x += translation_x
        self.max_x += translation_x
        self.min_y += translation_y
        self.max_y += translation_y
        self.box()

    def move_top_left_value(self, new_bottom_left_x, new_bottom_left_y):
        translation_x = new_bottom_left_x - self.min_x
        translation_y = new_bottom_left_y - self.max_y

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]


    def move_top_right(self, new_bottom_right_x, new_bottom_right_y):
        # Calculate the translation vector
        translation_x = new_bottom_right_x - self.max_x
        translation_y = new_bottom_right_y - self.max_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.max_x += translation_x
        self.max_y += translation_y
        self.box()

    def move_top_right_value(self,new_bottom_right_x, new_bottom_right_y):
        # Calculate the translation vector
        translation_x = new_bottom_right_x - self.max_x
        translation_y = new_bottom_right_y - self.max_y
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]

    def move_to_point(self, reference_vertex_index, new_x, new_y):
        # Get the reference vertex
        reference_vertex = self.coordinates[reference_vertex_index]

        # Calculate the translation vector
        translation_x = new_x - reference_vertex[0]
        translation_y = new_y - reference_vertex[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.max_x += translation_x
        self.max_y += translation_y
        self.min_x += translation_x
        self.min_y += translation_y

    def move_to_point_value(self, reference_vertex_index, new_x, new_y):
        # Get the reference vertex
        reference_vertex = self.coordinates[reference_vertex_index]

        # Calculate the translation vector
        translation_x = new_x - reference_vertex[0]
        translation_y = new_y - reference_vertex[1]

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]

    def move_edges_to_point(self, reference_edge_index, new_x, new_y):
        # Get the reference edge
        edges = self.get_edge_lines()
        reference_edge = edges[reference_edge_index]

        # Calculate the translation vector
        translation_x = new_x - reference_edge[0][0]
        translation_y = new_y - reference_edge[0][1]

        # Update all edges by adding the translation vector to each edge's points
        updated_edges = [((x1 + translation_x, y1 + translation_y), (x2 + translation_x, y2 + translation_y)) for
                         (x1, y1), (x2, y2) in edges]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.max_x += translation_x
        self.max_y += translation_y
        self.min_x += translation_x
        self.min_y += translation_y

        # Return the updated edge variations
        return updated_edges

    def move_edges_to_point_value(self, reference_edge_index, new_x, new_y):
        # Get the reference edge
        edges = self.get_edge_lines()
        reference_edge = edges[reference_edge_index]

        # Calculate the translation vector
        translation_x = new_x - reference_edge[0][0]
        translation_y = new_y - reference_edge[0][1]

        # Update all edges by adding the translation vector to each edge's points
        return [((x1 + translation_x, y1 + translation_y), (x2 + translation_x, y2 + translation_y)) for
                         (x1, y1), (x2, y2) in edges]

    def move_from_to(self, edge_of_pol, edge_of_region):
        # Calculate the translation vector
        translation_x = edge_of_region[0][0] - edge_of_pol[0][0]
        translation_y = edge_of_region[0][1] - edge_of_pol[0][1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update x and y coordinates separately
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update the bounding box
        self.max_x += translation_x
        self.min_x += translation_x
        self.max_y += translation_y
        self.min_y += translation_y

        # Update the new position
        self.x = edge_of_region[0][0]
        self.y = edge_of_region[0][1]

    def move_from_to2(self, point_of_pol, point_of_region):
        # Calculate the translation vector
        translation_x = point_of_region[0] - point_of_pol[0]
        translation_y = point_of_region[1] - point_of_pol[1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update x and y coordinates separately
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update the bounding box
        self.max_x += translation_x
        self.min_x += translation_x
        self.max_y += translation_y
        self.min_y += translation_y

        # Update the new position
        self.x = point_of_region[0]
        self.y = point_of_region[1]

    def move_from_to2_f_p_value(self, point_of_pol, point_of_region):
        # Calculate the translation vector
        translation_x = point_of_region[0] - point_of_pol[0]
        translation_y = point_of_region[1] - point_of_pol[1]


        return (point_of_pol[0] + translation_x), (point_of_pol[1] + translation_y)




    def move_from_to2_value(self, point_of_pol, point_of_region):
        # Calculate the translation vector
        translation_x = point_of_region[0] - point_of_pol[0]
        translation_y = point_of_region[1] - point_of_pol[1]

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]

    def align_polygon_start_with_region_start(self, edge_of_pol, edge_of_region):
        # Calculate the translation vector to align the polygon's start edge with the region's start edge
        translation_x = edge_of_region[0][0] - edge_of_pol[0][0]
        translation_y = edge_of_region[0][1] - edge_of_pol[0][1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update x and y coordinates separately
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update the bounding box
        self.max_x += translation_x
        self.min_x += translation_x
        self.max_y += translation_y
        self.min_y += translation_y

        # Update the new position to match the start of the region edge
        self.x = edge_of_region[0][0]
        self.y = edge_of_region[0][1]

    def align_polygon_start_with_region_start_value(self, edge_of_pol, edge_of_region):
        # Calculate the translation vector to align the polygon's start edge with the region's start edge
        translation_x = edge_of_region[0][0] - edge_of_pol[0][0]
        translation_y = edge_of_region[0][1] - edge_of_pol[0][1]

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]


    def align_polygon_end_with_region_start(self, edge_of_pol, edge_of_region):
        # Calculate the translation vector to align the polygon's end edge with the region's start edge
        translation_x = edge_of_region[0][0] - edge_of_pol[1][0]
        translation_y = edge_of_region[0][1] - edge_of_pol[1][1]

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]

        # Update x and y coordinates separately
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

        # Update the bounding box
        self.max_x += translation_x
        self.min_x += translation_x
        self.max_y += translation_y
        self.min_y += translation_y

        # Update the new position to match the start of the region edge
        self.x = edge_of_region[0][0]
        self.y = edge_of_region[0][1]

    def align_polygon_end_with_region_start_value(self, edge_of_pol, edge_of_region):
        # Calculate the translation vector to align the polygon's end edge with the region's start edge
        translation_x = edge_of_region[0][0] - edge_of_pol[1][0]
        translation_y = edge_of_region[0][1] - edge_of_pol[1][1]

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]

    def move_item_by_dis_and_angle(self, distance, angle_degrees):
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
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)


    def move_item_by_dis_and_angle_value(self, distance, angle_degrees):
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
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]

    def move_item_by_dis_and_angle_value2(self, distance, angle_degrees):
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
        return [(x + translation_x, y + translation_y) for x, y in self.coordinates]

    def move_to_rectangle(self, rectangle_coords):
        # Create Shapely Polygons from the given coordinates
        item_polygon = Polygon(self.coordinates)
        rectangle = Polygon(rectangle_coords)

        # Get the centroid of the rectangle
        rectangle_centroid = rectangle.centroid

        # Calculate the translation vector
        translation_vector = (rectangle_centroid.x - item_polygon.centroid.x, rectangle_centroid.y - item_polygon.centroid.y)

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_vector[0], y + translation_vector[1]) for x, y in self.coordinates]
        self.x_coords = [x + translation_vector[0] for x in self.x_coords]
        self.y_coords = [y + translation_vector[1] for y in self.y_coords]
        self.max_x = max(self.x_coords)
        self.max_y = max(self.y_coords)
        self.min_x = min(self.x_coords)
        self.min_y = min(self.y_coords)
