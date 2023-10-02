from shapely import box

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
        self.x = None
        self.y = None
        self.Box = None


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
        current_center_x = (self.max_x + self.min_x) / 2
        current_center_y = (self.max_y + self.min_y) / 2

        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Update all coordinates by adding the translation vector
        self.coordinates = [(x + translation_x, y + translation_y) for x, y in self.coordinates]
        self.x_coords = [x + translation_x for x in self.x_coords]
        self.y_coords = [y + translation_y for y in self.y_coords]

    def move_item_value(self, new_center_x, new_center_y):
        # Calculate the current center point of the item
        current_center_x = (self.max_x + self.min_x) / 2
        current_center_y = (self.max_y + self.min_y) / 2

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



