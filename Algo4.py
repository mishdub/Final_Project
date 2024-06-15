import random
import copy
from Draw import Draw
import math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint
from shapely.ops import unary_union
import time
import warnings





class Algo4:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances
        warnings.showwarning = self.warning_handler
        self.error_occurred = False  # Initialize error_occurred as False

    # Function to calculate the support point of a convex polygon in a given direction.
    def calculate_angle(self, point, centroid):
        return (math.atan2(point[1] - centroid[1], point[0] - centroid[0]) + 2 * math.pi) % (2 * math.pi)

    def is_counterclockwise(self,coordinates):
        # Calculate the centroid
        centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
        centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
        centroid = (centroid_x, centroid_y)

        # Sort the coordinates based on angles
        sorted_coordinates = sorted(coordinates, key=lambda point: self.calculate_angle(point, centroid))

        # Check if the sorted coordinates are in counterclockwise order
        for i in range(len(sorted_coordinates) - 1):
            x1, y1 = sorted_coordinates[i]
            x2, y2 = sorted_coordinates[i + 1]
            cross_product = (x2 - x1) * (y2 + y1)
            if cross_product <= 0:
                return False

        return True

    def order_coordinates_counterclockwise(self,coordinates):
        if self.is_counterclockwise(coordinates):
            print("orderd")
            return coordinates
        # Calculate the centroid
        centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
        centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
        centroid = (centroid_x, centroid_y)

        # Sort the coordinates based on angles
        sorted_coordinates = sorted(coordinates, key=lambda point: self.calculate_angle(point, centroid))

        return sorted_coordinates

    def move_poly(self, polygon, angle, convex_region):
        list_of_points = []
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        min_distance = float('inf')
        from_p = None
        to_p = None
        for point in polygon.coordinates:
            x, y = point
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line = LineString([(x, y), (x1, y1)])
            if not line.crosses(Polygon(polygon.coordinates)):
                list_of_points.append(point)
                inter_x, inter_y = self.find_intersection_point(convex_region, [(x, y), (x1, y1)], (x, y))
                p1 = Point(x, y)
                p2 = Point(inter_x, inter_y)
                distance = p1.distance(p2)
                if distance < min_distance:
                    min_distance = distance
                    from_p = point
                    to_p = (inter_x, inter_y)
        return from_p, to_p

    def move_poly_MultiLineString(self, polygon, angle, MultiLineString):
        list_of_lines = []
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        min_distance = float('inf')
        from_p = None
        to_p = None
        for point in polygon.coordinates:
            x, y = point
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line = LineString([(x, y), (x1, y1)])
            if not line.crosses(Polygon(polygon.coordinates)):
                for line_st in MultiLineString:
                    start_point = Point(line_st.coords[0])
                    end_point = Point(line_st.coords[-1])

                    # Create a new LineString with only the start and end points
                    simplified_line = LineString([start_point, end_point])
                    list_of_lines.append((line_st.coords[0], line_st.coords[-1]))

                    in_po = self.find_intersection_point_linestring(line_st, [(x, y), (x1, y1)], (x, y))
                    if in_po is not None:
                        inter_x, inter_y = in_po
                        p1 = Point(x, y)
                        p2 = Point(inter_x, inter_y)
                        distance = p1.distance(p2)
                        if distance == 0:
                            break
                        if distance < min_distance:
                            min_distance = distance
                            from_p = point
                            to_p = (inter_x, inter_y)

        return from_p, to_p,list_of_lines

    def place_poly(self, original_polygon, extended_poly, convex_region, angle):
        point_in_convex = None
        dime = self.container_instance.calculate_total_dimensions()
        po = self.container_instance.calculate_centroid()
        convex_exterior = Polygon(convex_region)
        convex_exterior = convex_exterior.exterior
        f_p = None
        t_p = None
        list_of_lines = None
        if extended_poly.intersects(convex_exterior):
            intersection = extended_poly.intersection(convex_exterior)

            if intersection.is_empty:
                print("Polygons overlap, but no intersection.")
            else:
                if intersection.geom_type == "Polygon":
                    print("Polygon")
                    min_distance = float('inf')
                    for point in intersection.exterior.coords:
                        angle = (angle + 180) % 360
                        vx, vy = (
                            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
                        cx, cy = point
                        x2, y2 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
                        px, py = self.find_intersection_point2(original_polygon.coordinates, [(cx, cy), (x2, y2)],
                                                               (cx, cy))
                        point_from_poly = Point(px, py)
                        point_from_convex = Point(cx, cy)
                        distance = point_from_poly.distance(point_from_convex)
                        if distance < min_distance:
                            min_distance = distance
                            f_p = (px, py)
                            t_p = (cx, cy)
                elif intersection.geom_type == "LineString":
                    print("line string")

                    f_p, t_p = self.move_poly(original_polygon, angle, convex_region)

                elif intersection.geom_type == "MultiLineString":
                    print("check")
                    f_p, t_p, list_of_lines = self.move_poly_MultiLineString(original_polygon, angle, intersection.geoms)
        return f_p, t_p, list_of_lines

    def classify_points_forward(self, line_angle, line_start, points, angle_tolerance=180):
        forward_points = []

        for point in points:
            # Calculate the angle between the line and the vector from line_start to the point.
            angle_rad = math.atan2(point[1] - line_start[1], point[0] - line_start[0])
            angle_deg = math.degrees(angle_rad)

            # Determine if the point is in the forward direction based on the angle tolerance.
            angle_difference = (angle_deg - line_angle + 360) % 360  # Ensure angle_difference is positive
            if 0 <= angle_difference <= angle_tolerance:
                forward_points.append(point)

        return forward_points

    def classify_points_left_right(self, line_angle, line_start, points):
        left_side_points = []
        right_side_points = []

        for point in points:
            # Calculate the angle between the line and the vector from line_start to the point.
            angle_rad = math.atan2(point[1] - line_start[1], point[0] - line_start[0])
            angle_deg = math.degrees(angle_rad)

            # Determine if the point is on the left or right side.
            if line_angle < 0:
                if angle_deg >= line_angle and angle_deg <= line_angle + 180:
                    left_side_points.append(point)
                else:
                    right_side_points.append(point)
            else:
                if angle_deg >= line_angle and angle_deg <= line_angle + 180:
                    right_side_points.append(point)
                else:
                    left_side_points.append(point)

        return left_side_points, right_side_points
    # Function to find the farthest point within the specified angle region
    def classify_points_left_right1(self,line_angle, line_start, points):
        left_side_points = []
        right_side_points = []

        for point in points:
            # Calculate the angle between the line and the vector from line_start to the point.
            angle_rad = math.atan2(point[1] - line_start[1], point[0] - line_start[0])
            angle_deg = math.degrees(angle_rad)

            # Determine if the point is on the left or right side.
            angle_difference = (angle_deg - line_angle + 360) % 360  # Ensure angle_difference is positive
            if angle_difference < 180:
                left_side_points.append(point)
            else:
                right_side_points.append(point)

        return left_side_points, right_side_points
    def find_farthest_point_from_line(self,line_coords, points, polygon,vx, vy, dime):
        # Create a LineString object from the line coordinates.
        line = LineString(line_coords)

        max_distance = -1
        farthest_point = None

        for point_coords in points:
            # Create a Point object from the point coordinates.
            point = Point(point_coords)

            # Calculate the distance from the point to the line.
            distance = point.distance(line)
            x, y = point_coords
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line2 = LineString([(x, y), (x1, y1)])
            if distance > max_distance:
                if not line2.crosses(Polygon(polygon)):
                    max_distance = distance
                    farthest_point = point_coords
        return farthest_point

    def placement(self, angle, middle_polygon, convex_polygon):
        dime = self.container_instance.calculate_total_dimensions()
        center = self.container_instance.calculate_centroid()
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, middle_polygon)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x1, y1)]

        px1, py1 = self.find_farthest_point_from_line(line1, right, middle_polygon, vx, vy, dime)
        px2, py2 = self.find_farthest_point_from_line(line1, left, middle_polygon, vx, vy, dime)

        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        right_line = LineString([(px1, py1), p1])
        left_line = LineString([(px2, py2), p2])
        filled_polygon = Polygon(list(left_line.coords) + list(right_line.coords)[::-1])

        flag = False
        if not (filled_polygon.intersects(Polygon(convex_polygon.coordinates))):
            flag = True
        return flag, (px1, py1), p1, (px2, py2), p2, (cx, cy), (x1,y1), filled_polygon

    def extend_pol(self, angle, convex_region, polygon):
        dime = self.container_instance.calculate_total_dimensions()
        center = polygon.calculate_centroid()
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, polygon.coordinates)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x1, y1)]

        px1, py1 = self.find_farthest_point_from_line(line1, right,polygon.coordinates, vx, vy, dime)
        px2, py2 = self.find_farthest_point_from_line(line1, left, polygon.coordinates, vx, vy, dime)

        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        px, py = self.find_intersection_point2(convex_region, [(px1, py1), p1],
                                               (px1, py1))
        qx, qy = self.find_intersection_point2(convex_region, [(px2, py2), p2],
                                               (px2, py2))
        pol_coord = [(px1, py1), (px, py), (qx, qy), (px2, py2)]
        order_coordinates = self.order_coordinates_counterclockwise(pol_coord)
        pol = Polygon(order_coordinates)
        origin_pol = Polygon(polygon.coordinates)
        mergedPolys = unary_union([origin_pol, pol])
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

        return exterior_coords_list

    def perpendicular_angle(self, angle_degrees):
        # Calculate the angle that is perpendicular to the given angle in degrees
        perpendicular_angle_degrees = (angle_degrees + 90) % 360  # Adding 90 degrees
        return perpendicular_angle_degrees

    def opposite_angle(self, angle_degrees):
        # Calculate the angle that is in the opposite direction of the given angle in degrees
        opposite_angle_degrees = (angle_degrees - 90) % 360
        return opposite_angle_degrees

    def find_vectors(self, angle, polygon, convex_polygon):
        cx, cy = self.container_instance.calculate_centroid()
        dime = self.container_instance.calculate_total_dimensions()

        random_direction_angle = self.perpendicular_angle(angle)
        vx, vy = (
            math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
        x2, y2 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        point1 = self.find_intersection_point(polygon, [(cx, cy), (x2, y2)],(cx, cy))

        random_direction_angle = self.opposite_angle(angle)
        vx, vy = (
            math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
        x2, y2 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        point2 = self.find_intersection_point(polygon, [(cx, cy), (x2, y2)],(cx, cy))
        px1, py1 = point1
        px2, py2 = point2
        random_direction_angle = angle
        vx, vy = (
            math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        line1 = LineString([point1, p1])
        line2 = LineString([point2, p2])
        flag = False
        if not (line1.intersects(Polygon(convex_polygon.coordinates))) and not (line2.intersects(Polygon(convex_polygon.coordinates))):
            flag = True

        return point1,p1,point2,p2,flag


    def calculate_distance(self,point1, point2):
        x1, y1 = point1
        x2, y2 = point2
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def find_closest_point(self,point, multi_point):
        if multi_point is None:
            return None  # Return None if there are no intersection points

        distances = [point.distance(p) for p in multi_point]
        closest_point_index = distances.index(min(distances))
        closest_point = multi_point[closest_point_index]
        return closest_point

    def find_intersection_point(self, polygon_coordinates, line_coords, po):
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
            closest_point = None
            min_distance = float('inf')
            given_point = Point(po)
            for point in intersection.geoms:
                distance = given_point.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point

            # Find the closest point
            return closest_point.x, closest_point.y
        else:
            # Other types of intersections (e.g., LineString, MultiLineString)
            return None  # Handle as needed

    def find_intersection_point2(self, polygon_coordinates, line_coords, po):
        # Create a polygon from the given coordinates

        polygon = Polygon(polygon_coordinates)

        exterior_ring = polygon.exterior

        # Create a LineString from the given line coordinates
        line = LineString(line_coords)

        # Find the intersection between the line and the polygon
        intersection = line.intersection(exterior_ring)

        # Check the type of the result to handle different cases
        if intersection.is_empty:
            line = line.buffer(0.01)
            intersection = line.intersection(exterior_ring)

        if intersection.is_empty:
            return None  # No intersection
        elif intersection.geom_type == 'Point':
            # Only one intersection point
            return (intersection.x, intersection.y)
        elif intersection.geom_type == 'MultiPoint':
            # Multiple intersection points
            closest_point = None
            min_distance = float('inf')
            given_point = Point(po)
            for point in intersection.geoms:
                distance = given_point.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
                    # Find the closest point
            return closest_point.x, closest_point.y
        elif intersection.geom_type == 'LineString':
            # Multiple intersection points
            closest_point = None
            min_distance = float('inf')
            given_point = Point(po)
            for point in list(intersection.coords):
                point = Point(point)
                distance = given_point.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
                    # Find the closest point
            return closest_point.x, closest_point.y

        else:
            print(intersection.geom_type)

            # Other types of intersections (e.g., LineString, MultiLineString)
            return None  # Handle as needed

    def find_intersection_point_linestring(self, line_string, line_coords, po):
        # Create a polygon from the given coordinates

        # Create a LineString from the given line coordinates
        line = LineString(line_coords)

        # Find the intersection between the line and the polygon
        intersection = line.intersection(line_string)

        # Check the type of the result to handle different cases
        if intersection.is_empty:
            line = line.buffer(0.01)
            intersection = line.intersection(line_string)

        if intersection.is_empty:
            return None  # No intersection
        elif intersection.geom_type == 'Point':
            # Only one intersection point
            return (intersection.x, intersection.y)
        elif intersection.geom_type == 'MultiPoint':
            # Multiple intersection points
            closest_point = None
            min_distance = float('inf')
            given_point = Point(po)
            for point in intersection.geoms:
                distance = given_point.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
                    # Find the closest point
            return closest_point.x, closest_point.y
        elif intersection.geom_type == 'LineString':
            # Multiple intersection points
            closest_point = None
            min_distance = float('inf')
            given_point = Point(po)
            for point in list(intersection.coords):
                point = Point(point)
                distance = given_point.distance(point)
                if distance < min_distance:
                    min_distance = distance
                    closest_point = point
                    # Find the closest point
            return closest_point.x, closest_point.y

        else:
            print(intersection.geom_type)

            # Other types of intersections (e.g., LineString, MultiLineString)
            return None  # Handle as needed
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

        return max(width, height)
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

        return index, closest_point

    def for_edges_that_intersect(self, pol1, pol2):
        buffered_result = pol2.buffer(0.1)

        mergedPolys = pol1.difference(buffered_result)
        #print("before",mergedPolys)

        exterior_coords_list = []
        if isinstance(mergedPolys, MultiPolygon):
            largest_polygon = None
            largest_area = 0
            # Iterate through the constituent polygons
            for polygon in mergedPolys.geoms:
                # Get the coordinates of the exterior boundary of each polygon
                if isinstance(polygon, Polygon):
                    # Calculate the area of the polygon
                    area = polygon.area
                    if area > largest_area:
                        largest_area = area
                        largest_polygon = polygon
            return list(largest_polygon.exterior.coords)

        else:
            # If it's a single Polygon, get its exterior coordinates directly
            return list(mergedPolys.exterior.coords)

    def calculate_next_angle(self, current_angle, current_polygon, next_polygon, vector):
        threshold = self.container_instance.calculate_distance_threshold()
        cx,cy = self.container_instance.calculate_centroid()
        dx, dy = vector
        pol_area = current_polygon.calculate_total_dimensions()
        next_pol_area = next_polygon.calculate_total_dimensions()
        convex_area = self.container_instance.calculate_total_dimensions()
        # Calculate the distance adjustment based on both width and height
        distance = (threshold/(pol_area/2)) + (threshold/(next_pol_area/2))
        #+ (threshold/(next_pol_area/2))

        radius = math.sqrt(dx ** 2 + dy ** 2)
        new_angle = current_angle + math.atan2(dy, dx) + (distance / radius)
        flag = False
        print(distance)
        while not flag:
            current_angle = (current_angle + distance) % 360
            vx, vy = (
                math.cos(math.radians(current_angle)), math.sin(math.radians(current_angle)))
            # Calculate the endpoint of the line
            x1 = cx + self.container_instance.calculate_total_dimensions() * vx
            y1 = cy + self.container_instance.calculate_total_dimensions() * vy

            line1 = LineString([(cx, cy), (x1, y1)])
            if not (line1.intersects(Polygon(current_polygon.coordinates))):
                print("does it", line1.intersects(Polygon(current_polygon.coordinates)))
                flag = True
                new_angle = current_angle

            # Ensure the angle is within [0, 360] degrees
        return new_angle

    def plot(self):
        i = 0
        new_region = self.container_instance.coordinates
        for dex, polygon in enumerate(self.item_instances):
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            polygon.move_item(x, y)
            while True:
                # Generate a random direction angle between 0 and 360 degrees
                random_direction_angle = random.uniform(0, 360)
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
                print(random_direction_angle)
                # Start at the middle point of the convex region

                list = []
                # Initialize farthest_point and max_projection with None and -inf, respectively
                farthest_point = None
                max_projection = float('-inf')

                # Iterate through all points that define the boundary of the convex region
                for i in range(len(new_region)):
                    current_coord = new_region[i]
                    next_coord = new_region[(i + 1) % len(new_region)]

                    # Calculate the direction vector of the boundary edge
                    edge_vector = (next_coord[0] - current_coord[0], next_coord[1] - current_coord[1])

                    # Calculate the projection of the direction vector onto the random_direction_vector
                    projection = edge_vector[0] * random_direction_vector[0] + edge_vector[1] * random_direction_vector[
                        1]

                    if projection > max_projection:
                        # Update max_projection and farthest_point if the projection is greater
                        max_projection = projection
                        farthest_point = current_coord

                # find the closest point in the polygon to the point in the convex
                polygon.box()
                index, point = self.find_closest_point_to_point(polygon, farthest_point)
                (X, Y) = polygon.coordinates[index]
                (X2, Y2) = polygon.coordinates[(index-1) % len(polygon.coordinates)]
                (X3, Y3) = polygon.coordinates[(index+1) % len(polygon.coordinates)]


                list.append(polygon)
                draw_instance = Draw(self.container_instance, list, (X, Y), (X2, Y2), (X3, Y3), (1, 1), None)
                draw_instance.plot()
                list.pop()
                # Place the polygon at the farthest point
                pol_coords = polygon.move_to_point_value(index, farthest_point[0], farthest_point[1])
                copied = copy.deepcopy(polygon)
                copied2 = copy.deepcopy(polygon)
                copied.move_to_point(index, farthest_point[0], farthest_point[1])
                list.append(copied)
                copied2.set_coordinates(new_region)
                list.append(copied2)
                draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                draw_instance.plot()
                list.pop()
                list.pop()


                # Check if the polygon is inside the convex region
                pol = Polygon(pol_coords)
                print(pol_coords)
                convex_region = Polygon(new_region)
                if pol.within(convex_region):
                    polygon.move_to_point(index, farthest_point[0], farthest_point[1])
                    list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                       Polygon(copied.coordinates))
                    copied.set_coordinates(list_of_new_region)
                    list.append(copied)

                    draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                    draw_instance.plot()
                    list.pop()
                    new_region = list_of_new_region
                    break  # Placement is successful


            if i == 1:
                break
            i = i+ 1


    def calculate_area(self,coordinates):
        if len(coordinates) < 3:
            return "A polygon must have at least 3 coordinates."

        # Calculate the center of the polygon
        x_coords, y_coords = zip(*coordinates)
        center_x = sum(x_coords) / len(coordinates)
        center_y = sum(y_coords) / len(coordinates)

        # Calculate the distance from the center to a vertex as the radius
        radius = math.sqrt((center_x - coordinates[0][0]) ** 2 + (center_y - coordinates[0][1]) ** 2)

        # Determine the number of sides in the polygon
        num_sides = len(coordinates)

        # Calculate the area using the regular polygon area formula
        area = (0.5 * num_sides * radius ** 2 * math.sin(2 * math.pi / num_sides))

        return area
    def plot2(self):
        new_region = self.container_instance.coordinates
        max_iterations = 100  # Maximum number of iterations for the inner loop
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        start_time = time.time()

        for dex, polygon in enumerate(sorted_items):
            print(dex)
            if dex == 240:
                break
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            for _ in range(max_iterations):
                polygon.move_item(x, y)
                # Generate a random direction angle between 0 and 360 degrees
                random_direction_angle = random.uniform(0,  360)
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

                # Initialize the binary search parameters
                min_distance = 0.0
                max_distance_x = self.container_instance.calculate_width()
                max_distance_y = self.container_instance.calculate_height()
                max_distance = max(max_distance_x, max_distance_y)

                tolerance = 0.1  # A small value to stop the binary search
                list = []
                moved = False

                for _ in range(max_iterations):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    polygon.move_item(new_x, new_y)
                    list.append(polygon)
                    copied = copy.deepcopy(polygon)
                    copied.set_coordinates(new_region)
                    list.append(copied)
                    #draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None,None,None)
                    #draw_instance.plot()
                    list.pop()
                    list.pop()

                    # Check if the polygon is inside the convex region
                    pol = Polygon(polygon.coordinates)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    # Reset the position back to the original
                    polygon.move_item(x, y)
                    if abs(max_distance - min_distance) < tolerance:
                        if pol.within(convex_region):
                            break  # Exit the loop if it's converging

                # If the polygon has not moved, break the loop
                if not moved:
                    break

                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                copied = copy.deepcopy(polygon)
                polygon.move_item(final_x, final_y)
                the_list.append(polygon)
                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))

                copied.set_coordinates(list_of_new_region)
                list.append(copied)
                if dex == 300:
                    draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                    draw_instance.plot()

                list.pop()
                new_region = list_of_new_region
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None,None,None)
        draw_instance.plot()
        print("polygons",len(the_list),elapsed_time)



    def plot3(self):
        new_region = self.container_instance.coordinates
        max_iterations = 100  # Maximum number of iterations for the inner loop
        random_direction_angle = 0
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        for dex, polygon in enumerate(sorted_items):
            if dex == 100:
                break
            print(dex)
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            for k in range(720):
                polygon.move_item(x, y)
                random_direction_angle = k+polygon.calculate_total_dimensions()
                # Generate a random direction angle between 0 and 360 degrees
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
                # Initialize the binary search parameters
                min_distance = 0.0
                max_distance_x = self.container_instance.calculate_width()
                max_distance_y = self.container_instance.calculate_height()
                max_distance = max(max_distance_x, max_distance_y)

                tolerance = 0.1  # A small value to stop the binary search
                list = []
                moved = False

                for _ in range(max_iterations):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    polygon.move_item(new_x, new_y)
                    list.append(polygon)
                    copied = copy.deepcopy(polygon)
                    copied.set_coordinates(new_region)
                    list.append(copied)
                    draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                    # draw_instance.plot()
                    list.pop()
                    list.pop()

                    # Check if the polygon is inside the convex region
                    pol = Polygon(polygon.coordinates)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    # Reset the position back to the original
                    polygon.move_item(x, y)
                    print(max_distance - min_distance, dex)
                    if abs(max_distance - min_distance) < tolerance:
                        if pol.within(convex_region):
                            break  # Exit the loop if it's converging

                # If the polygon has not moved, break the loop
                if not moved:
                    #random_direction_angle = (random_direction_angle+polygon.calculate_total_dimensions()) % 720
                    break

                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                copied = copy.deepcopy(polygon)
                polygon.move_item(final_x, final_y)
                the_list.append(polygon)
                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))

                copied.set_coordinates(list_of_new_region)
                list.append(copied)

                draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                #draw_instance.plot()
                list.pop()
                new_region = list_of_new_region
                #random_direction_angle = (random_direction_angle) % 360
                break
        draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None)
        draw_instance.plot()
        print(len(the_list))

    def plot4(self):
        new_region = self.container_instance.coordinates
        max_iterations = 100  # Maximum number of iterations for the inner loop
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        start_time = time.time()

        for dex, polygon in enumerate(sorted_items):
            print(dex)
            if dex == 300:
                break
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            for _ in range(max_iterations):
                polygon.move_item(x, y)
                # Generate a random direction angle between 0 and 360 degrees
                random_direction_angle = random.uniform(0, 1440)
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

                # Initialize the binary search parameters
                min_distance = 0.0
                max_distance_x = self.container_instance.calculate_width()
                max_distance_y = self.container_instance.calculate_height()
                max_distance = min(max_distance_x, max_distance_y)

                tolerance = 0.1  # A small value to stop the binary search
                list = []
                moved = False

                for _ in range(20):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    polygon.move_item(new_x, new_y)
                    list.append(polygon)
                    copied = copy.deepcopy(polygon)
                    copied.set_coordinates(new_region)
                    list.append(copied)
                    draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                    # draw_instance.plot()
                    list.pop()
                    list.pop()

                    # Check if the polygon is inside the convex region
                    pol = Polygon(polygon.coordinates)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    # Reset the position back to the original
                    polygon.move_item(x, y)
                    print(max_distance - min_distance, dex)
                    if abs(max_distance - min_distance) < tolerance:
                        if pol.within(convex_region):
                            break  # Exit the loop if it's converging

                # If the polygon has not moved, break the loop
                if not moved:
                    break

                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                copied = copy.deepcopy(polygon)
                polygon.move_item(final_x, final_y)
                the_list.append(polygon)
                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))

                copied.set_coordinates(list_of_new_region)
                list.append(copied)
                if dex == 300:
                    draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None)
                    draw_instance.plot()

                list.pop()
                new_region = list_of_new_region
                break
        draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None)
        draw_instance.plot()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("polygons", len(the_list), elapsed_time)

    def plot5(self):
        new_region = self.container_instance.coordinates
        max_iterations = 100  # Maximum number of iterations for the inner loop
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        start_time = time.time()
        random_direction_angle = 0

        random_direction_vector = (
            math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
        angle1 = None
        angle2 = None

        for dex, polygon in enumerate(sorted_items):
            if dex == 10:
                break
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point

            for _ in range(max_iterations):
                polygon.move_item(x, y)
                # Generate a random direction angle between 0 and 360 degrees

                # Initialize the binary search parameters
                min_distance = 0.0
                max_distance_x = self.container_instance.calculate_width()
                max_distance_y = self.container_instance.calculate_height()
                max_distance = max(max_distance_x, max_distance_y)

                tolerance = 0.1  # A small value to stop the binary search
                list = []
                moved = False

                for _ in range(max_iterations):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    polygon.move_item(new_x, new_y)

                    list.append(polygon)
                    draw_instance = Draw(self.container_instance, list, (x, y), (1000, 1000), (x, y),
                                         (1000, 1000), None, random_direction_angle, None)
                    #draw_instance.plot()
                    copied = copy.deepcopy(polygon)
                    copied.set_coordinates(new_region)
                    list.append(copied)

                    list.pop()
                    list.pop()

                    # Check if the polygon is inside the convex region
                    pol = Polygon(polygon.coordinates)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    # Reset the position back to the original
                    lis = []
                    lis.append(polygon)
                    polygon.move_item(x, y)
                    if abs(max_distance - min_distance) < tolerance:
                        if pol.within(convex_region):
                            break  # Exit the loop if it's converging


                # If the polygon has not moved, break the loop
                if not moved:
                    break

                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                copied = copy.deepcopy(polygon)
                polygon.move_item(final_x, final_y)

                the_list.append(polygon)
                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))

                copied.set_coordinates(list_of_new_region)
                list.append(copied)
                list.pop()
                new_region = list_of_new_region
                next_polygon = None
                if dex < len(sorted_items) - 1:
                    # Get the next polygon
                    next_polygon = sorted_items[dex + 1]
                if dex == 0:
                    angle1 = random_direction_angle

                #random_direction_angle = (random_direction_angle+polygon.calculate_total_dimensions() + next_polygon.calculate_total_dimensions()) % 360
                random_direction_angle = self.calculate_next_angle(random_direction_angle,polygon,next_polygon,random_direction_vector)
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

                if dex == 0:
                    angle2 = random_direction_angle

                draw_instance = Draw(self.container_instance, the_list, (1,1),(1,1),(1,1),(1,1),None,None,None)
                draw_instance.plot()
                break

        x, y =self.container_instance.calculate_centroid()
        print(angle1,angle2)
        draw_instance = Draw(self.container_instance, the_list, (x, y), (1000, 1000), (x, y), (1000, 1000), None,angle1,angle2)
        draw_instance.plot()
        end_time = time.time()
        elapsed_time = end_time - start_time
        print("polygons",len(the_list),elapsed_time)

    # Define a warning handler to print warnings
    def warning_handler(self,message, category, filename, lineno, file=None, line=None):
        self.error_occurred = True
        print(f"Warning: {category.__name__}: {message}")




    def plot6(self):
        new_region = self.container_instance.coordinates
        max_iterations = 100  # Maximum number of iterations for the inner loop
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        start_time = time.time()
        infinity = float('inf')

        for dex, polygon in enumerate(sorted_items):
            print(dex)
            if dex == 200:
                break

            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            polygon.box()
            for _ in range(100):
                polygon.move_item(x, y)
                # Generate a random direction angle between 0 and 360 degrees
                random_direction_angle = random.uniform(0,  360)
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

                # Initialize the binary search parameters
                min_distance = 0.0
                max_distance_x = self.container_instance.calculate_width()
                max_distance_y = self.container_instance.calculate_height()
                max_distance = min(max_distance_x, max_distance_y)

                tolerance = 0.1  # A small value to stop the binary search
                list = []
                moved = False
                for j in range(max_iterations):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    polygon.move_item(new_x, new_y)
                    list.append(polygon)
                    copied = copy.deepcopy(polygon)
                    #copied.set_coordinates(new_region)
                    #list.append(copied)
                    draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None, None,
                                         None)
                    #draw_instance.plot()

                    #list.pop()
                    list.pop()
                    # Check if the polygon is inside the convex region
                    pol = Polygon(polygon.coordinates)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    # Reset the position back to the original
                    copied2 = copy.deepcopy(polygon)

                    polygon.move_item(x, y)
                    if abs(max_distance - min_distance) < tolerance:
                        try:
                            if pol.within(convex_region):
                                break  # Exit the loop if it's converging
                        except Exception as error:

                            # Handle the warning here
                            print("Caught an exception:", error)
                        finally:
                            if self.error_occurred:
                                #copied.set_coordinates(new_region)
                                print(new_region)
                                #list.append(copied)
                                list.append(copied2)

                                draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1),
                                                     None,
                                                     None,
                                                     None)
                                draw_instance.plot()
                                #list.pop()
                                list.pop()
                # If the polygon has not moved, break the loop
                if not moved:
                    break

                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                copied = copy.deepcopy(polygon)
                polygon.move_item(final_x, final_y)
                the_list.append(polygon)
                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))
                #copied.set_coordinates(list_of_new_region)
                #list.append(copied)

                #list.pop()
                new_region = list_of_new_region

                #self.container_instance.set_coordinates(list_of_new_region)
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None,None,None)
        draw_instance.plot()
        print("polygons",len(the_list),elapsed_time)

    def plot7(self):
        new_region = self.container_instance.coordinates
        new_region2 = self.container_instance.coordinates
        new_region3 = self.container_instance.coordinates


        dime = self.container_instance.calculate_total_dimensions()
        max_iterations = 100  # Maximum number of iterations for the inner loop
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        another_list = []

        value = 0
        start_time = time.time()
        random_direction_angle = 0
        random_direction_vector = (
            math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
        max_distance_x = self.container_instance.calculate_width()
        max_distance_y = self.container_instance.calculate_height()
        dist = min(max_distance_x, max_distance_y)
        for dex, polygon in enumerate(sorted_items):
            if dex == 120:
                break
            print(dex)
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            i = 0
            max_val = 0
            for i in range(max_iterations):
                polygon.move_item(x, y)
                # Generate a random direction angle between 0 and 360 degrees
                min_distance = 0.0
                max_distance = dist

                tolerance = 0.1  # A small value to stop the binary search
                moved = False
                j = 0
                for j in range(max_iterations):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    polygon.move_item(new_x, new_y)
                    li = []
                    li.append(polygon)
                    #draw_instance = Draw(self.container_instance, li, (1, 1), (1, 1), (1, 1), (1, 1), None, None,None)
                    #draw_instance.plot()

                    li.pop()
                    # Check if the polygon is inside the convex region
                    pol = Polygon(polygon.coordinates)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    polygon.move_item(x, y)
                    if abs(max_distance - min_distance) < tolerance:
                        if pol.within(convex_region):
                            break  # Exit the loop if it's converging
                # If the polygon has not moved, break the loop

                if not moved:
                    break
                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                polygon.move_item(final_x, final_y)
                the_list.append(polygon)

                value = value + polygon.value
                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))
                new_region = list_of_new_region
                copied = copy.deepcopy(polygon)
                copied.set_coordinates(new_region)
                li = []
                li.append(polygon)
                li.append(copied)
                #draw_instance = Draw(self.container_instance, li, (1, 1), (1, 1), (1, 1), (1, 1), None, None,None,)
                #draw_instance.plot()
                li.pop()
                li.pop()
                flag = False
                while not flag:
                    random_direction_angle = (random_direction_angle + 0.5) % 360
                    random_direction_vector = (
                        math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))
                    next_polygon = None
                    if dex < len(sorted_items) - 1:
                        # Get the next polygon
                        next_polygon = sorted_items[dex + 1]
                    copied4 = copy.deepcopy(next_polygon)
                    copied4.move_item(x, y)
                    flag, d1, d2, d3, d4, d5, d6, extended_polygon = self.placement(random_direction_angle, next_polygon.move_item_value(x, y), polygon)
                    flag2 = False
                    if flag:
                        copied6 = copy.deepcopy(copied4)

                        copied6.set_coordinates(new_region3)

                        f_p, t_p, list_of_lines = self.place_poly(copied4, extended_polygon, new_region3, random_direction_angle)

                        copied4.move_from_to2(f_p, t_p)
                        #li = self.extend_pol(random_direction_angle, new_region3, copied4)

                        new_region3 = self.for_edges_that_intersect(Polygon(new_region3),
                                                                    Polygon(copied4.coordinates))
                        another_list.append(copied4)
                        #another_list.append(copied6)





                        #list_of_new = self.for_edges_that_intersect(Polygon(new_region2),Polygon(li))

                        #new_region2 = list_of_new
                        #another_list.append(copied4)
                        #copied4.set_coordinates(list_of_new)
                        if dex >= 200:
                            #print("len",len(list_of_lines))
                            draw_instance = Draw(self.container_instance, another_list, (1,1), (1,1), (1, 1), (1, 1), None,
                                                 None,
                                                 None, list_of_lines)
                            #draw_instance.plot()
                        another_list.pop()
                        #another_list.pop()

                    if flag:
                        break
                break
        end_time = time.time()
        elapsed_time = end_time - start_time
        draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None,None,None, None)
        draw_instance.plot()
        print("num of polygons", len(the_list),"out of",len(self.item_instances), "time", elapsed_time, "value", value)

    def plot8(self):
        new_region = self.container_instance.coordinates
        max_iterations = 100  # Maximum number of iterations for the inner loop
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        the_list = []
        value = 0
        start_time = time.time()
        for dex, polygon in enumerate(sorted_items):
            print(dex)
            print(len(the_list))
            if dex == 10:
                break
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            i = 0
            for i in range(max_iterations):
                polygon.move_item(x, y)
                # Generate a random direction angle between 0 and 360 degrees
                random_direction_angle = random.uniform(0,  360)
                random_direction_vector = (
                    math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

                # Initialize the binary search parameters
                min_distance = 0.0
                max_distance_x = self.container_instance.calculate_width()
                max_distance_y = self.container_instance.calculate_height()
                max_distance = max(max_distance_x, max_distance_y)

                tolerance = 1  # A small value to stop the binary search
                moved = False
                j = 0
                for j in range(max_iterations):  # Limit the inner loop to a maximum number of iterations
                    # Calculate the middle distance
                    mid_distance = (min_distance + max_distance) / 2

                    # Calculate the new position
                    new_x = x + mid_distance * random_direction_vector[0]
                    new_y = y + mid_distance * random_direction_vector[1]

                    # Move the polygon to the new position
                    co = polygon.move_item_value(new_x, new_y)
                    # Check if the polygon is inside the convex region
                    pol = Polygon(co)
                    convex_region = Polygon(new_region)

                    if pol.within(convex_region):
                        min_distance = mid_distance
                        moved = True
                    else:
                        max_distance = mid_distance

                    if abs(max_distance - min_distance) < tolerance:
                        if pol.within(convex_region):
                            break  # Exit the loop if it's converging
                #print("j",j)
                # If the polygon has not moved, break the loop
                if not moved:
                    break
                # Move the polygon to the final position
                final_x = x + min_distance * random_direction_vector[0]
                final_y = y + min_distance * random_direction_vector[1]
                polygon.move_item(final_x, final_y)
                the_list.append(polygon)
                value = value + polygon.value

                list_of_new_region = self.for_edges_that_intersect(Polygon(new_region),
                                                                   Polygon(polygon.coordinates))
                new_region = list_of_new_region
                #print("i", i)
                break

        end_time = time.time()
        elapsed_time = end_time - start_time
        draw_instance = Draw(self.container_instance, the_list, (1, 1), (1, 1), (1, 1), (1, 1), None,None,None,None)
        draw_instance.plot()
        print("num of polygons", len(the_list),"out of",len(self.item_instances), "time", elapsed_time, "value", value)
