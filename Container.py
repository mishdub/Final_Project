import random
import math
import copy
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString, MultiLineString



class Container:
    def __init__(self, x_coords, y_coords):
        self.coordinates = list(zip(x_coords, y_coords))
        self.x_coords = x_coords
        self.y_coords = y_coords
        self.max_x = max(x_coords)
        self.max_y = max(y_coords)
        self.min_x = min(x_coords)
        self.min_y = min(y_coords)
        self.grid_coordinates = []
        self.middle = None


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

    def calculate_area(self):
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

    def get_edge_lines(self):
        edges = []
        num_points = len(self.coordinates)

        for i in range(num_points):
            point1 = self.coordinates[i]
            point2 = self.coordinates[(i + 1) % num_points]  # Wrap around to the first point

            line = (point1, point2)
            edges.append(line)

        return edges

    def create_grid_coordinates(self, grid_spacing):
        """
        Create grid coordinates within the boundaries of the container.

        Args:
        - grid_spacing: Spacing between grid lines.

        Returns:
        - grid_coordinates: List of grid coordinates (x, y) as tuples.
        """

        # Determine the minimum and maximum coordinates of the container
        x_min, x_max = min(self.x_coords), max(self.x_coords)
        y_min, y_max = min(self.y_coords), max(self.y_coords)

        # Iterate through rows
        for y in range(y_min, y_max + 1, grid_spacing):
            # Iterate through columns
            for x in range(x_min, x_max + 1, grid_spacing):
                point = (x, y)
                if self.point_inside_container(point) or self.point_on_boundary(point):
                    self.grid_coordinates.append(point)

        return self.grid_coordinates

    def create_grid_coordinates1(self, grid_spacing, items):
        """
        Create grid coordinates within the boundaries of the container, excluding points within the items.

        Args:
        - grid_spacing: Spacing between grid lines.
        - items: List of items, where each item is represented as a list of (x, y) coordinates.

        Returns:
        - grid_coordinates: List of grid coordinates (x, y) as tuples.
        """

        # Determine the minimum and maximum coordinates of the container
        x_min, x_max = min(self.x_coords), max(self.x_coords)
        y_min, y_max = min(self.y_coords), max(self.y_coords)

        # Iterate through rows
        for y in range(y_min, y_max + 1, grid_spacing):
            # Iterate through columns
            for x in range(x_min, x_max + 1, grid_spacing):
                point = (x, y)

                # Check if the point is inside any of the items (if items list is not empty)
                if items and any(item.point_inside_item(point) for item in items):
                    continue

                if self.point_inside_container(point) or self.point_on_boundary(point):
                    self.grid_coordinates.append(point)

        return self.grid_coordinates

    def point_inside_container(self, point):
        """
        Check if a point is inside the container.

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

    def point_inside_rectangle(self, point, rec):
        """
        Check if a point is inside a rectangle defined by its four corner coordinates.

        Args:
        point (tuple): A tuple containing the (x, y) coordinates of the point to check.
        rec (list): A list of tuples containing the (x, y) coordinates of the four rectangle corners.

        Returns:
        bool: True if the point is inside the rectangle, False otherwise.
        """
        x, y = point
        x1, y1 = rec[0]
        x2, y2 = rec[1]
        x3, y3 = rec[2]
        x4, y4 = rec[3]

        minX = min(x1, x2, x3, x4)
        maxX = max(x1, x2, x3, x4)
        minY = min(y1, y2, y3, y4)
        maxY = max(y1, y2, y3, y4)

        if (x >= minX and x <= maxX) and (y >= minY and y <= maxY):
            return True
        else:
            return False

    def point_on_boundary(self, point):
        """
        Check if a point is on the boundary of the container.

        Args:
        - point: (x, y) tuple representing the point to check.

        Returns:
        - True if the point is on the boundary of the container, False otherwise.
        """
        x, y = point
        n = len(self.coordinates)

        for i in range(n):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[(i + 1) % n]

            # Check if the point lies on the line segment defined by (x1, y1) and (x2, y2)
            if (x == x1 and y == y1) or (x == x2 and y == y2):
                return True

            # Check if the point lies on the line segment and is within the bounding box of the segment
            if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2):
                # Calculate the cross product to check if the point is collinear with the line segment
                cross_product = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
                if abs(cross_product) < 1e-6:  # A small tolerance for floating-point comparisons
                    return True

        return False

    def group_grid_coordinates_into_rectangles(self, grid_spacing, items):
        """
        Group grid coordinates into rectangles with a specified grid_spacing.

        Args:
        - grid_spacing: Spacing between grid lines.

        Returns:
        - List of rectangles, where each rectangle is represented as a list of 4 points.
        """
        rectangles = []
        coordinets = self.create_grid_coordinates1(grid_spacing, items)
        original_coordinets = copy.deepcopy(coordinets)  # Create a deep copy

        while len(coordinets) >= 4:
            # Take the first grid coordinate as a starting point
            start_point = coordinets.pop(0)
            closest_points = [start_point]

            # Find the 3 closest points to the starting point
            for _ in range(3):
                closest_point = min(coordinets,
                                    key=lambda p: (p[0] - start_point[0]) ** 2 + (p[1] - start_point[1]) ** 2)

                # Check if the distance between the closest_point and start_point is equal to grid_spacing
                distance = math.sqrt(
                    (closest_point[0] - start_point[0]) ** 2 + (closest_point[1] - start_point[1]) ** 2)
                if math.isclose(distance, grid_spacing):
                    closest_points.append(closest_point)
                    coordinets.remove(closest_point)

            # If we found 3 closest points, add the fourth point
            if len(closest_points) == 3:
                # Calculate the direction vector from start_point to closest_points[1]
                dir_vector = (closest_points[1][0] - start_point[0], closest_points[1][1] - start_point[1])

                # Calculate the fourth point
                fourth_point = (start_point[0] + dir_vector[0], start_point[1] + dir_vector[1] + grid_spacing)
                closest_points.append(fourth_point)

                # Sort the points to ensure counter-clockwise orientation
                sorted_rectangle = self.sort_points_counter_clockwise(closest_points)
                rectangles.append(sorted_rectangle)

        return rectangles, original_coordinets

    def create_rectangles(self, coordinates, spacing, items):
        # Sort the coordinates by x and y values
        coordinates.sort(key=lambda coord: (coord[0], coord[1]))

        # Create a set of unprocessed coordinates
        unprocessed = set(coordinates)

        # Initialize a list to store the rectangles
        rectangles = []

        # Iterate through the sorted coordinates
        for i in range(len(coordinates)):
            for j in range(i + 1, len(coordinates)):
                x1, y1 = coordinates[i]
                x2, y2 = coordinates[j]

                # Calculate the width and height of the potential rectangle
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                # Check if the width and height are both equal to the spacing
                if width == spacing and height == spacing:
                    # Check if all four corners of the rectangle are in the provided list
                    if (x1, y1) in unprocessed and (x2, y1) in unprocessed and (x2, y2) in unprocessed and (
                            x1, y2) in unprocessed:

                        # Create a rectangle tuple and add it to the list
                        rectangle = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

                        # Sort the corners of the rectangle to ensure consistent ordering
                        rectangle.sort()
                        # Check if this sorted rectangle is already in the list
                        if rectangle not in rectangles:
                            rectangles.append(rectangle)
        temp = []
        # Initialize flag to False outside the loop
        flag = False
        # Iterate through the items to check if their centroids are inside the potential rectangle

        for rectangle in rectangles:
            for item in items:
                z1, z2 = item.calculate_centroid()
                flag = self.point_inside_rectangle((z1, z2), rectangle)
                if flag is True:
                    break
            if flag is False:
                temp.append(rectangle)

        return temp

    def sort_points_counter_clockwise(self, points):
        """
            Sort a list of points to have a counter-clockwise orientation.

            Args:
            - points: List of 4 points [(x1, y1), (x2, y2), (x3, y3), (x4, y4)].

            Returns:
            - List of 4 points with counter-clockwise orientation.
            """
        center_x = sum(p[0] for p in points) / 4
        center_y = sum(p[1] for p in points) / 4
        return sorted(points, key=lambda p: math.atan2(p[1] - center_y, p[0] - center_x))

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
        for i, (edge, slope) in enumerate(self.slope_list().items()):
            x1, y1 = self.coordinates[i]
            x2, y2 = self.coordinates[(i + 1) % len(self.coordinates)]

            if abs(x2 - x1) < 1e-6:  # Vertical line
                equation = f"x = {x1:.2f}"
            elif abs(y2 - y1) < 1e-6:  # Horizontal line
                equation = f"y = {y1:.2f}"
            else:  # Non-vertical line
                y_intercept = y1 - slope * x1
                equation = f"y = {slope:.2f}x + {y_intercept:.2f}"

            equations[edge] = equation
        return equations

    def random_point_on_edge(self):
        edge = random.choice(list(self.slope_list().keys()))
        slope = self.slope_list()[edge]
        x1, y1 = self.coordinates[int(edge.split('_')[1]) - 1]

        if slope == 0.0:  # Horizontal line
            x = random.uniform(min(self.x_coords), max(self.x_coords))
            y = y1
        elif slope == -0.0:  # Vertical line
            x = x1
            y = random.uniform(min(self.y_coords), max(self.y_coords))
        else:
            x = random.uniform(min(self.x_coords), max(self.x_coords))
            y = slope * x + (y1 - slope * x1)

        return x, y

    def calculate_centroid(self):
        num_points = len(self.coordinates)
        if num_points == 0:
            return None

        centroid_x = sum(self.x_coords) / num_points
        centroid_y = sum(self.y_coords) / num_points
        return centroid_x, centroid_y

    def calculate_middle_points(self):
        middle_points = []

        for i in range(len(self.coordinates)):
            edge_start = self.coordinates[i]
            edge_end = self.coordinates[(i + 1) % len(self.coordinates)]
            middle_x = (edge_start[0] + edge_end[0]) / 2
            middle_y = (edge_start[1] + edge_end[1]) / 2
            middle_point = (middle_x, middle_y)
            middle_points.append((edge_start, edge_end, middle_point))

        return middle_points

    def determine_projection_direction_from_middle(self, middle_point):
        centroid = self.calculate_centroid()
        centroid_x, centroid_y = centroid
        middle_x, middle_y = middle_point

        dx = middle_x - centroid_x
        dy = middle_y - centroid_y

        if abs(dx) > abs(dy):
            if dx > 0:
                return "left"
            else:
                return "right"
        else:
            if dy > 0:
                return "down"
            else:
                return "up"

    def determine_all_middle_projection_directions(self):
        middle_points = self.calculate_middle_points()
        projection_directions = []

        for middle_point in middle_points:
            direction = self.determine_projection_direction_from_middle(middle_point[2])
            projection_directions.append(direction)

        return projection_directions

    def create_grid_coordinates_with_equations(self, grid_spacing, items):
        """
        Create grid coordinates within the boundaries of the container, excluding points within the items.

        Args:
        - grid_spacing: Spacing between grid lines.
        - items: List of items, where each item is represented as a list of (x, y) coordinates.

        Returns:
        - grid_coordinates: List of grid coordinates (x, y) as tuples.
        """

        # Determine the minimum and maximum coordinates of the container
        x_min, x_max = min(self.x_coords), max(self.x_coords)
        y_min, y_max = min(self.y_coords), max(self.y_coords)

        # Generate grid points efficiently using line equations
        grid_coordinates = []
        for y in range(y_min, y_max + 1, grid_spacing):
            for x in range(x_min, x_max + 1, grid_spacing):
                point = (x, y)

                # Check if the point is inside any of the items using line equations
                if not items or not any(item.point_inside_item(point) for item in items):
                    if self.point_inside_container(point) or self.point_on_boundary(point):
                        grid_coordinates.append(point)

        return grid_coordinates

    def generate_points_inside_container(self, items):
        """
        Generate random points inside the container that do not intersect with points of any items.

        Args:
        - items: List of items, where each item is represented as an instance of the Item class.

        Returns:
        - points: List of generated points as tuples (x, y).
        """
        container_vertices = self.coordinates  # List of vertices defining the container

        points = []

        for _ in range(len(items)):
            while True:
                # Generate a random point within the container
                x = random.uniform(min(self.x_coords), max(self.x_coords))
                y = random.uniform(min(self.y_coords), max(self.y_coords))
                point = (x, y)

                # Check if the point intersects with any items
                intersects = False
                for item in items:
                    if any(
                            x == item_x and y == item_y for item_x, item_y in item.coordinates
                    ):
                        intersects = True
                        break

                # Check if the point is inside the container
                if not intersects and self.point_inside_container(point):
                    points.append(point)
                    break

        return points

    def generate_feasible_point(self, item, items):
        """
        Generate a feasible point inside the container for the given item.

        Args:
        - item: An instance of the Item class.
        - items: List of other items in the container.

        Returns:
        - point: A feasible point as a tuple (x, y).
        """
        x_min, x_max = min(self.x_coords), max(self.x_coords)
        y_min, y_max = min(self.y_coords), max(self.y_coords)

        while True:
            # Calculate the valid range of x and y coordinates for the item
            valid_x_range = (x_min, x_max - item.max_x + item.min_x)
            valid_y_range = (y_min, y_max - item.max_y + item.min_y)

            # Generate a random point within the valid range
            x = random.uniform(valid_x_range[0], valid_x_range[1])
            y = random.uniform(valid_y_range[0], valid_y_range[1])
            point = (x, y)

            # Check if the point intersects with any other items or goes beyond the container boundaries
            intersects = any(
                any(
                    x == other_x and y == other_y
                    for other_x, other_y in other_item.coordinates
                )
                for other_item in items
            )

            # Check if the point goes beyond the container boundaries
            if (
                    x + item.max_x > x_max
                    or y + item.max_y > y_max
                    or x < x_min
                    or y < y_min
            ):
                intersects = True

            # If the point is valid, return it
            if not intersects:
                return point

    def generate_random_point(self, item, items):
        # Ensure there are valid coordinates in the container
        if not self.coordinates:
            raise ValueError("Container is empty")
        max_attempts = 10 # Maximum number of attempts to generate a feasible point
        for _ in range(max_attempts):
            # Generate random x and y coordinates within the range of the container
            x = random.uniform(min(self.x_coords), max(self.x_coords))
            y = random.uniform(min(self.y_coords), max(self.y_coords))

            deep_copy_item = copy.deepcopy(item)
            deep_copy_item.move_item(x, y)

            # Check if the item is inside the container and does not overlap with other items
            # Check if the deep copied item is inside the container
            flag1 = True
            for (x_p, y_p) in deep_copy_item.coordinates:
                flag1 = self.point_inside_container((x_p, y_p))
                if not flag1:
                    break

            if items:
                flag2 = False
                flag3 = False
                flag4 = False

                found_flag1 = False
                for other_item in items:
                    poly1 = Polygon(other_item.coordinates)
                    for (x_point, y_point) in deep_copy_item.coordinates:
                        p1 = Point(x_point, y_point)
                        # flag2 = other_item.point_inside_item((x_point, y_point))
                        flag2 = p1.within(poly1)
                        if flag2:
                            found_flag1 = True
                            break
                    if found_flag1:
                        break

                found_flag2 = False
                for other_item in items:
                    poly2 = Polygon(deep_copy_item.coordinates)
                    for (x_point, y_point) in other_item.coordinates:
                        p2 = Point(x_point, y_point)
                        # flag3 = deep_copy_item.point_inside_item((x_point, y_point))
                        flag3 = p2.within(poly2)
                        if flag3:
                            found_flag2 = True
                            break
                    if found_flag2:
                        break

                found_flag3 = False
                found_flag4 = False
                for (a, b) in deep_copy_item.get_edge_lines():
                    line_a = LineString([a, b])
                    for other_item in items:
                        for (c, d) in other_item.get_edge_lines():
                            line_b = LineString([c, d])
                            flag4 = line_a.intersects(line_b)
                            if flag4:
                                found_flag3 = True
                                break
                        if found_flag3:
                            found_flag4 = True
                            break
                    if found_flag4:
                        break


                if flag1 and not found_flag1 and not found_flag2 and not found_flag4:
                #if flag1 and not found_flag4:
                    item.move_item(x, y)
                    return x, y, True
            elif not items:
                if flag1:
                    item.move_item(x, y)
                    return x, y, True

        return 0, 0, False
