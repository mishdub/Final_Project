import math
from shapely.geometry import MultiPolygon, Polygon, GeometryCollection, Point, LineString
from shapely.ops import unary_union, split
from math import sqrt
import time
import copy


class Algorithm:

    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def calculate_angle(self, point, centroid):
        return (math.atan2(point[1] - centroid[1], point[0] - centroid[0]) + 2 * math.pi) % (2 * math.pi)

    def order_coordinates_counterclockwise(self, coordinates):
        # Calculate the centroid
        centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
        centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
        centroid = (centroid_x, centroid_y)

        # Sort the coordinates based on angles
        sorted_coordinates = sorted(coordinates, key=lambda point: self.calculate_angle(point, centroid))

        return sorted_coordinates

    def find_min_intersection_dis_from_boundary_side_to_polygon(self, polygon_coordinates, angle, line_geometry,
                                                                distance):
        # Adjust the angle to get the reverse direction
        reversed_angle = (angle + 180) % 360
        direction_x, direction_y = math.cos(math.radians(reversed_angle)), math.sin(math.radians(reversed_angle))

        min_distance = float('inf')  # Initialize minimum distance to infinity
        closest_polygon_point = None
        closest_line_point = None

        # Ensure that line_geometry is iterable (either MultiLineString or LineString)
        if line_geometry.geom_type == "LineString":
            line_geometries = [line_geometry]
        elif line_geometry.geom_type == "MultiLineString":
            line_geometries = line_geometry.geoms
        else:
            raise TypeError("line_geometry must be a LineString or MultiLineString")

        # Process each line segment
        for line in line_geometries:
            for line_point in line.coords:
                line_x, line_y = line_point
                endpoint_x, endpoint_y = self.calculate_endpoint_from_direction(line_x, line_y, direction_x,
                                                                                direction_y, distance)
                intersection_point = self.find_closest_intersection_point(polygon_coordinates,
                                                                          [(line_x, line_y), (endpoint_x, endpoint_y)],
                                                                          (line_x, line_y), True)

                if intersection_point is not None:
                    intersection_x, intersection_y = intersection_point
                    point_on_line = Point(line_x, line_y)
                    point_on_polygon = Point(intersection_x, intersection_y)
                    distance_to_polygon = point_on_line.distance(point_on_polygon)

                    if distance_to_polygon < min_distance:
                        min_distance = distance_to_polygon
                        closest_polygon_point = (intersection_x, intersection_y)
                        closest_line_point = (line_x, line_y)

        return closest_polygon_point, closest_line_point, min_distance

    def find_min_intersection_dis_from_polygon_side_to_boundary(self, polygon_coordinates, angle, line_geometry,
                                                                distance):
        # Calculate the direction vector based on the given angle
        direction_x, direction_y = math.cos(math.radians(angle)), math.sin(math.radians(angle))

        min_distance = float('inf')  # Initialize the minimum distance to infinity
        closest_polygon_point = None
        closest_boundary_point = None

        # Ensure line_geometry is either LineString or MultiLineString
        if line_geometry.geom_type == "LineString":
            line_geometries = [line_geometry]
        elif line_geometry.geom_type == "MultiLineString":
            line_geometries = line_geometry.geoms
        else:
            raise TypeError("line_geometry must be a LineString or MultiLineString")

        # Process each point on the polygon
        for polygon_point in polygon_coordinates:
            polygon_x, polygon_y = polygon_point
            endpoint_x, endpoint_y = self.calculate_endpoint_from_direction(polygon_x, polygon_y, direction_x,
                                                                            direction_y, distance)
            line_segment = LineString([(polygon_x, polygon_y), (endpoint_x, endpoint_y)])

            # Ensure the line segment does not cross the polygon
            if not line_segment.crosses(Polygon(polygon_coordinates)):
                for line in line_geometries:
                    intersection_point = self.find_closest_intersection_point(line, [(polygon_x, polygon_y),
                                                                                     (endpoint_x, endpoint_y)],
                                                                              (polygon_x, polygon_y), False)
                    if intersection_point is not None:
                        intersection_x, intersection_y = intersection_point
                        point_on_polygon = Point(polygon_x, polygon_y)
                        point_on_boundary = Point(intersection_x, intersection_y)
                        distance_to_boundary = point_on_polygon.distance(point_on_boundary)

                        if distance_to_boundary == 0:
                            break  # No need to continue if distance is zero
                        if distance_to_boundary < min_distance:
                            min_distance = distance_to_boundary
                            closest_polygon_point = (polygon_x, polygon_y)
                            closest_boundary_point = (intersection_x, intersection_y)

        return closest_polygon_point, closest_boundary_point, min_distance

    def find_best_placement_for_polygon(self, original_polygon, extended_poly, convex_region, angle, right_line, left_line, dime):
        convex_exterior = Polygon(convex_region).exterior
        min_dis_from_intersection = float('inf')
        f_p, t_p = None, None

        # Step 1: Check if extended_poly intersects with the convex exterior
        if extended_poly.intersects(convex_exterior):
            intersection = extended_poly.intersection(convex_exterior)
            if intersection.is_empty:
                raise TypeError("Polygons overlap, but no intersection.")

            # Step 2: Handle LineString and MultiLineString intersections
            if intersection.geom_type in ["LineString", "MultiLineString"]:
                f_p, t_p, min_dis_from_intersection = self.handle_line_intersection(
                    original_polygon, angle, intersection, dime
                )
        else:
            raise TypeError("No intersection found")

        # Step 3: Check the minimum distance from the boundaries - right line and left line
        f_p, t_p = self.check_boundary_distances(
            convex_exterior, right_line, left_line, f_p, t_p, min_dis_from_intersection
        )

        return f_p, t_p

    def handle_line_intersection(self, original_polygon, angle, intersection, dime):
        # Find distances from both polygon side and boundary side
        from_poly, to_poly, dis_poly = self.find_min_intersection_dis_from_polygon_side_to_boundary(
            original_polygon, angle, intersection, dime
        )
        from_boundary, to_boundary, dis_boundary = self.find_min_intersection_dis_from_boundary_side_to_polygon(
            original_polygon, angle, intersection, dime
        )

        # Determine which distance is smaller and return the corresponding points
        if intersection.geom_type == "LineString":
            if dis_poly < dis_boundary:
                return from_poly, to_poly, dis_poly
            else:
                return from_boundary, to_boundary, dis_boundary

        elif intersection.geom_type == "MultiLineString":
            if dis_boundary < dis_poly:
                return from_boundary, to_boundary, dis_boundary
            else:
                return from_poly, to_poly, dis_poly

    def check_boundary_distances(self, convex_exterior, right_line, left_line, closest_start_point, closest_end_point,
                                 min_distance):
        # Final step - check the minimum distance from the boundaries - left line and right line.

        # Extract starting points from the right and left lines
        right_start_point = Point((list(right_line.coords))[0])
        left_start_point = Point((list(left_line.coords))[0])

        # Convert right and left lines to lists of coordinates
        right_line_coordinates = list(right_line.coords)
        left_line_coordinates = list(left_line.coords)

        # Extract the first points of the right and left lines
        right_initial_point = right_line_coordinates[0]
        left_initial_point = left_line_coordinates[0]

        # Find intersection points between the convex exterior and the lines
        right_intersection = self.find_closest_intersection_point(convex_exterior.coords, right_line_coordinates,
                                                                      right_initial_point, True)
        left_intersection = self.find_closest_intersection_point(convex_exterior.coords, left_line_coordinates,
                                                                     left_initial_point, True)

        if left_intersection is None and right_intersection is not None:
            right_intersection = Point(right_intersection)
            right_distance = right_start_point.distance(right_intersection)
            if right_distance < min_distance:
                closest_start_point = (right_start_point.x, right_start_point.y)
                closest_end_point = (right_intersection.x, right_intersection.y)
        elif right_intersection is None and left_intersection is not None:
            left_intersection = Point(left_intersection)
            left_distance = left_start_point.distance(left_intersection)
            if left_distance < min_distance:
                closest_start_point = (left_start_point.x, left_start_point.y)
                closest_end_point = (left_intersection.x, left_intersection.y)
        elif right_intersection is not None and left_intersection is not None:
            right_intersection = Point(right_intersection)
            left_intersection = Point(left_intersection)

            right_distance = right_start_point.distance(right_intersection)
            left_distance = left_start_point.distance(left_intersection)
            if right_distance < left_distance:
                if right_distance < min_distance:
                    closest_start_point = (right_start_point.x, right_start_point.y)
                    closest_end_point = (right_intersection.x, right_intersection.y)
            else:
                if left_distance < min_distance:
                    closest_start_point = (left_start_point.x, left_start_point.y)
                    closest_end_point = (left_intersection.x, left_intersection.y)

        return closest_start_point, closest_end_point

    def classify_points_left_right(self, line_angle, line_start, points):
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

    def find_farthest_point_from_line(self, line_coordinates, candidate_points, polygon_vertices, vector_x, vector_y,
                                      distance_multiplier):
        # Create a LineString object from the line coordinates.
        reference_line = LineString(line_coordinates)

        max_distance = -1
        farthest_point = None

        polygon_shape = Polygon(polygon_vertices)

        for point_coordinates in candidate_points:
            # Create a Point object from the point coordinates.
            current_point = Point(point_coordinates)

            # Calculate the distance from the point to the line.
            distance_to_line = current_point.distance(reference_line)
            current_x, current_y = point_coordinates

            # Calculate the endpoint in the direction of the vector.
            end_point = self.calculate_endpoint_from_direction(current_x, current_y, vector_x, vector_y,
                                                                            distance_multiplier)
            extended_line = LineString([(current_x, current_y), end_point])

            if distance_to_line > max_distance:
                if not extended_line.crosses(polygon_shape):
                    max_distance = distance_to_line
                    farthest_point = point_coordinates

        return farthest_point

    def find_closest_intersection_point(self, geometry, line_coordinates, reference_point, is_polygon=False):
        # If geometry is a polygon, use its exterior ring
        if is_polygon:
            polygon = Polygon(geometry)
            geometry = polygon.exterior

        # Create a LineString from the given line coordinates
        line = LineString(line_coordinates)

        # Find the intersection between the line and the geometry
        intersection = line.intersection(geometry)

        # Check the type of the result to handle different cases
        if intersection.is_empty:
            return None  # No intersection
        elif intersection.geom_type == 'Point':
            # Only one intersection point
            return (intersection.x, intersection.y)
        elif intersection.geom_type == 'MultiPoint':
            # Multiple intersection points (for both Polygon and LineString cases)
            closest_point = None
            minimum_distance = float('inf')
            reference_geom_point = Point(reference_point)
            for point in intersection.geoms:
                distance = reference_geom_point.distance(point)
                if distance < minimum_distance:
                    minimum_distance = distance
                    closest_point = point
            return closest_point.x, closest_point.y
        elif intersection.geom_type == 'LineString':
            # Multiple intersection points (for both Polygon and LineString cases)
            closest_point = None
            minimum_distance = float('inf')
            reference_geom_point = Point(reference_point)
            for coordinate in list(intersection.coords):
                current_point = Point(coordinate)
                distance = reference_geom_point.distance(current_point)
                if distance < minimum_distance:
                    minimum_distance = distance
                    closest_point = current_point
            return closest_point.x, closest_point.y
        else:
            print(intersection.geom_type)
            return None  # Handle as needed

    def calculate_endpoint_from_direction(self, x1, y1, dx, dy, length):
        # Calculate the end point
        x2 = x1 + length * dx
        y2 = y1 + length * dy

        return x2, y2

    def calculate_angle_in_degrees(self, point, centroid):
        angle_radians = math.atan2(centroid[1] - point[1], centroid[0] - point[0])
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def check_if_line_crosses_the_polygon(self, convex_polygon, polygon):
        # Initialize an empty list to store points that pass the line-crossing check
        new_list = []

        # Create a Shapely Polygon object from the given polygon points
        pol = Polygon(polygon)

        # Calculate the centroid of the convex polygon
        center_con = self.calculate_centroid(convex_polygon)

        # Iterate through each point in the given polygon
        for point in polygon:
            # Create a line from the centroid of the convex polygon to the current point
            line = LineString([center_con, point])

            # Check if the line does not cross the polygon
            if not line.crosses(pol):
                # If the point is not already in the new list, add it
                if point not in new_list:
                    new_list.append(point)

        # Return the list of points whose lines do not cross the polygon
        return new_list

    def calculate_centroid(self, coordinates):
        # Create a convex polygon from the given coordinates
        convex_polygon = Polygon(coordinates)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid

        return centroid.x, centroid.y

    def polygon_to_rectangle(self, coordinates):
        # Check if the input list of coordinates is not empty
        if not coordinates:
            raise ValueError("The list of coordinates cannot be empty.")

        # Find the minimum and maximum coordinates of the polygon
        min_x = min(coord[0] for coord in coordinates)
        min_y = min(coord[1] for coord in coordinates)
        max_x = max(coord[0] for coord in coordinates)
        max_y = max(coord[1] for coord in coordinates)

        # Create rectangle coordinates using min and max values
        rect_coordinates = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

        return rect_coordinates

    def find_most_left_or_right_point(self, angle, polygon_coordinates, new_center, size_extend, for_rec, dime):
        # Extend the polygon by a specified size
        extended_polygon = self.extend_polygon(polygon_coordinates, size_extend)

        # If the polygon needs to be treated as a rectangle, convert it into a rectangle
        if for_rec:
            extended_polygon = self.polygon_to_rectangle(extended_polygon)

        # If a new center is not provided, calculate the centroid of the extended polygon
        if new_center is None:
            center = self.calculate_centroid(extended_polygon)
        else:
            # Use the provided new center
            center = new_center

        # Find the left and right points along with their endpoints based on the given center, angle, and dimension (dime)
        left_point, right_point, left_endpoint, right_endpoint = self.find_left_and_right_points(center, angle,
                                                                                                 extended_polygon, dime)

        # Return the most left and right points of the polygon
        return left_point, right_point

    def intersection_of_lines(self, vertical_line_coordinates, horizontal_line_coordinates, angle, start_point,
                              distance):
        reverse_angle = (angle + 180) % 360  # Reverse the given angle by 180 degrees
        direction_x, direction_y = (
            math.cos(math.radians(reverse_angle)), math.sin(math.radians(reverse_angle)))  # Calculate direction vector

        start_x, start_y = start_point  # Decompose the start point into x and y coordinates
        end_point = self.calculate_endpoint_from_direction(start_x, start_y, direction_x, direction_y,
                                                           distance)  # Get the second point

        vertical_line = LineString(vertical_line_coordinates)  # Convert the coordinates to a vertical line object
        horizontal_line = LineString(horizontal_line_coordinates)  # Convert the coordinates to a horizontal line object
        main_line = LineString(
            [(start_x, start_y), end_point])  # Create the main line from the start point to the calculated endpoint

        vertical_intersection = vertical_line.intersection(main_line)  # Get intersection with the vertical line
        horizontal_intersection = horizontal_line.intersection(main_line)  # Get intersection with the horizontal line

        # If both intersections are valid (not empty), compare distances and return the closest one
        if not vertical_intersection.is_empty and not horizontal_intersection.is_empty:
            start_point_geom = Point((start_x, start_y))
            vertical_point = Point((vertical_intersection.x, vertical_intersection.y))
            horizontal_point = Point((horizontal_intersection.x, horizontal_intersection.y))
            vertical_distance = start_point_geom.distance(vertical_point)  # Distance to vertical intersection
            horizontal_distance = start_point_geom.distance(horizontal_point)  # Distance to horizontal intersection

            if vertical_distance < horizontal_distance:
                return vertical_intersection.x, vertical_intersection.y  # Return closer vertical intersection
            else:
                return horizontal_intersection.x, horizontal_intersection.y  # Return closer horizontal intersection
        elif not vertical_intersection.is_empty:
            return vertical_intersection.x, vertical_intersection.y  # Return vertical intersection if only one exists
        elif not horizontal_intersection.is_empty:
            return horizontal_intersection.x, horizontal_intersection.y  # Return horizontal intersection if only one exists
        else:
            return False  # Return False if no intersections are found

    def create_lines(self, vertices):
        # Find the center of the convex region
        center_x, center_y = self.calculate_centroid(vertices)

        # Find the minimum and maximum coordinates
        min_x, min_y = min(x for x, y in vertices), min(y for x, y in vertices)
        max_x, max_y = max(x for x, y in vertices), max(y for x, y in vertices)

        # Create LineString objects for the vertical and horizontal lines
        vertical_line = [(center_x, min_y), (center_x, max_y)]
        horizontal_line = [(min_x, center_y), (max_x, center_y)]

        return vertical_line, horizontal_line

    def merge_polygon_with_boundary(self, pol1, pol2, ext_size=None):
        # Extend pol2 if ext_size is provided
        if ext_size is not None:
            big_p = self.extend_polygon(pol2, ext_size)
            buffered_result = Polygon(big_p)
        else:
            buffered_result = Polygon(pol2)

        mergedPolys = pol1.difference(buffered_result)

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

        elif isinstance(mergedPolys, GeometryCollection):

            largest_geom = None

            largest_size = 0

            # Iterate over each geometry in the GeometryCollection

            for geom in mergedPolys.geoms:

                if isinstance(geom, Polygon):

                    # Calculate the area of the Polygon

                    size = geom.area

                    if size > largest_size:
                        largest_size = size

                        largest_geom = geom

                elif isinstance(geom, LineString):

                    # Calculate the length of the LineString

                    size = geom.length

                    if size > largest_size:
                        largest_size = size

                        largest_geom = geom

                elif isinstance(geom, Point):

                    # Points don't have a size, so just choose the first one encountered

                    if largest_geom is None:
                        largest_geom = geom

                else:

                    # Handle other geometry types if necessary

                    print(f"Found an unhandled geometry type: {type(geom)}")

            # Return the largest geometry based on area for polygons or length for lines

            if isinstance(largest_geom, Polygon):

                return list(largest_geom.exterior.coords)

            elif isinstance(largest_geom, LineString):

                return list(largest_geom.coords)

            elif isinstance(largest_geom, Point):

                return (largest_geom.x, largest_geom.y)

            else:

                raise ValueError("No valid geometry found")  # or a more specific/custom error message

        else:
            # If it's a single Polygon, get its exterior coordinates directly
            return list(mergedPolys.exterior.coords)

    def extend_polygon_with_angle(self, angle, convex_region, polygon, extension_size, dime):
        # Create the original extended polygon, the new extended polygon, and ordered coordinates based on the given angle
        original_extended_polygon, extended_polygon, ordered_coordinates = self.create_extended_polygon(angle,
                                                                                                        convex_region,
                                                                                                        polygon,
                                                                                                        extension_size,
                                                                                                        dime)

        # Initialize a list to hold the exterior coordinates of the merged polygons
        exterior_coordinates_list = []

        try:
            # Merge the original and new extended polygons
            merged_polygons = unary_union([original_extended_polygon, extended_polygon])

            # Check if the merged result is a MultiPolygon
            if isinstance(merged_polygons, MultiPolygon):
                # Iterate through each individual polygon in the MultiPolygon
                for individual_polygon in merged_polygons.geoms:
                    # Get the exterior coordinates and add them to the list
                    exterior_coordinates = list(individual_polygon.exterior.coords)
                    exterior_coordinates_list.extend(exterior_coordinates)
            else:
                # If not a MultiPolygon, directly get the exterior coordinates
                exterior_coordinates_list = list(merged_polygons.exterior.coords)

            # Validate the merged polygon; if invalid, convert it to a rectangle
            if not Polygon(exterior_coordinates_list).is_valid:
                print("Polygon is not valid; converting to rectangle.")
                exterior_coordinates_list = self.polygon_to_rectangle(exterior_coordinates_list)
        except Exception as e:
            # In case of an error, combine the original polygon and ordered coordinates
            combined_coordinates = polygon + ordered_coordinates
            # Order the combined coordinates counterclockwise
            combined_coordinates = self.order_coordinates_counterclockwise(combined_coordinates)
            # Convert the combined coordinates to a rectangle
            exterior_coordinates_list = self.polygon_to_rectangle(combined_coordinates)
            print(f"An error occurred: {str(e)}")

        # Return the list of exterior coordinates of the extended polygon
        return exterior_coordinates_list

    def create_extended_polygon(self, angle, convex_region, polygon, extension_size, tolerance):
        # Calculate the centroid of the given polygon
        centroid = self.calculate_centroid(polygon)

        # Find the left and right points from the centroid based on the specified angle and tolerance
        (left_x, left_y), (right_x, right_y), left_endpoint, right_endpoint = self.find_left_and_right_points(
            centroid, angle, polygon, tolerance)

        # Find the closest intersection point on the convex region for the right side
        right_intersection = self.find_closest_intersection_point(
            convex_region, [(right_x, right_y), right_endpoint], (right_x, right_y), True)

        # Use the intersection point or the original right point if no intersection is found
        right_intersection_x, right_intersection_y = (
            right_intersection if right_intersection is not None else (right_x, right_y)
        )

        # Find the closest intersection point on the convex region for the left side
        left_intersection = self.find_closest_intersection_point(
            convex_region, [(left_x, left_y), left_endpoint], (left_x, left_y), True)

        # Use the intersection point or the original left point if no intersection is found
        left_intersection_x, left_intersection_y = (
            left_intersection if left_intersection is not None else (left_x, left_y)
        )

        # Calculate the direction vector based on the given angle
        direction_vector_x, direction_vector_y = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle))
        )

        # Calculate the endpoint for the right intersection point based on the direction vector
        (right_intersection_x, right_intersection_y) = self.calculate_endpoint_from_direction(
            right_intersection_x, right_intersection_y, direction_vector_x, direction_vector_y, 2
        )

        # Calculate the endpoint for the left intersection point based on the direction vector
        (left_intersection_x, left_intersection_y) = self.calculate_endpoint_from_direction(
            left_intersection_x, left_intersection_y, direction_vector_x, direction_vector_y, 2
        )

        # Create a list of polygon coordinates using the right and left intersection points and original points
        polygon_coordinates = [
            (right_x, right_y),
            (right_intersection_x, right_intersection_y),
            (left_intersection_x, left_intersection_y),
            (left_x, left_y)
        ]

        # Order the coordinates counterclockwise for proper polygon representation
        ordered_coordinates = self.order_coordinates_counterclockwise(polygon_coordinates)

        # Create extended polygons based on the ordered coordinates and the original polygon
        extended_polygon = Polygon(self.extend_polygon(ordered_coordinates, extension_size))
        original_extended_polygon = Polygon(self.extend_polygon(polygon, extension_size))

        # Return the original extended polygon, the newly extended polygon, and the ordered coordinates
        return original_extended_polygon, extended_polygon, ordered_coordinates

    def extend_polygon(self, coordinates, buffer_distance):
        original_polygon = Polygon(coordinates)

        # Buffer the original polygon
        buffered_polygon = original_polygon.buffer(buffer_distance*2, join_style="mitre")

        return list(buffered_polygon.exterior.coords)

    def placement(self, angle, middle_polygon, convex_polygon, dime):
        # Calculate the centroid of the middle polygon
        center = self.calculate_centroid(middle_polygon)

        # Find the left and right points, as well as their endpoints based on the center and given angle
        left_point, right_point, end_left_point, end_right_point = self.find_left_and_right_points(center, angle,
                                                                                                   middle_polygon, dime)

        # Create lines representing the left and right extensions
        right_line = LineString([right_point, end_right_point])
        left_line = LineString([left_point, end_left_point])

        # Create a filled polygon by combining the left and right lines
        filled_polygon = Polygon(list(left_line.coords) + list(right_line.coords)[::-1])

        # Generate the convex hull for the middle polygon
        convex_hull_for_pol = Polygon(middle_polygon).convex_hull

        # Unite the convex hull with the filled polygon to create a combined shape
        unite_pols = convex_hull_for_pol.union(filled_polygon)

        # Update filled_polygon with the united shape
        filled_polygon = unite_pols

        # Initialize a flag to check intersection with the convex polygon
        flag = False
        if convex_polygon is not None:
            # Get the extended size of the convex polygon
            big_p = convex_polygon.ext_size
            big_p = Polygon(big_p)

            # Check if the filled polygon does not intersect with the convex polygon
            if not (filled_polygon.intersects(big_p)):
                flag = True

        # Return the flag indicating intersection status, the filled polygon, and the left and right lines
        return flag, filled_polygon, right_line, left_line

    def shrink_polygon(self, distance, coordinates):
        centroid = (Polygon(coordinates)).centroid
        new_points = []
        current_center_x = centroid.x
        current_center_y = centroid.y
        for point in coordinates:
            # Create a vector from the centroid to the current vertex
            vector_x = point[0] - current_center_x
            vector_y = point[1] - current_center_y
            # Calculate the current distance from the centroid to the vertex
            current_distance = math.sqrt(vector_x ** 2 + vector_y ** 2)
            # Calculate the new distance
            new_distance = current_distance - distance
            if new_distance < 0:
                new_distance = 0  # Prevents the new distance from becoming negative
            # Scale the vector
            new_x = current_center_x + (vector_x / current_distance * new_distance)
            new_y = current_center_y + (vector_y / current_distance * new_distance)
            # Append the new point
            new_points.append((new_x, new_y))
        return new_points

    def find_potential_tangent_point(self, angle, previous_polygon, point, most_left_right_point_func, ext_const,
                                     diameter, is_rectangle=False, is_backward=False):
        # Record the start time for timeout handling
        start_time = time.time()
        time_limit = 5  # Set a time limit of 5 seconds for the operation

        # Calculate the centroid of the previous polygon to use as a starting point
        center = self.calculate_centroid(previous_polygon.coordinates)
        potential_point = center  # Initialize the potential point to the center

        # Continuously attempt to find the tangent point
        while True:
            # Depending on whether we are moving backward or forward, call the left/right point function
            if is_backward:
                temp_point_v1, potential_point = most_left_right_point_func(angle, previous_polygon.coordinates,
                                                                            potential_point, ext_const, False, diameter)
            else:
                potential_point, temp_point_v2 = most_left_right_point_func(angle, previous_polygon.coordinates,
                                                                            potential_point, ext_const, is_rectangle,
                                                                            diameter)

            # Recalculate the angle between the reference point and the current potential point
            angle = self.calculate_angle_in_degrees(point, potential_point)

            # Unpack the reference point coordinates
            point_x, point_y = point

            # Adjust the angle slightly based on whether the function is running in reverse or forward mode
            if is_backward:
                adjusted_angle = (angle - 0.01 % 360)
            else:
                adjusted_angle = (angle + 0.01 % 360)

            # Calculate the direction vector using the adjusted angle
            vector_x, vector_y = (
                math.cos(math.radians(adjusted_angle)),
                math.sin(math.radians(adjusted_angle))
            )

            # Find the endpoint by extending the direction vector from the point
            endpoint_x, endpoint_y = self.calculate_endpoint_from_direction(point_x, point_y, vector_x, vector_y,
                                                                            diameter)

            # Create lines from the reference point: one to the endpoint and another to the potential point
            line_to_endpoint = LineString([point, (endpoint_x, endpoint_y)])
            line_to_potential_point = LineString([point, potential_point])

            # Check if the polygon should be treated as a rectangle or not and create the bounding polygon accordingly
            if is_rectangle:
                bounding_polygon = previous_polygon.ext_size_for_loop_rec
                bounding_polygon = self.polygon_to_rectangle(bounding_polygon)
            else:
                bounding_polygon = previous_polygon.ext_size_for_loop

            bounding_polygon = Polygon(bounding_polygon)  # Ensure the bounding polygon is a valid Polygon object

            # Check if the time spent exceeds the defined time limit (5 seconds)
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time > time_limit:
                return (None, None, True)  # Return None if the operation times out

            # Check if the line extending from the point crosses the polygon's boundary
            if not line_to_endpoint.crosses(bounding_polygon):
                # If the line to the potential point touches the bounding polygon, return the angle and point
                if line_to_potential_point.touches(bounding_polygon):
                    if is_backward:
                        return angle, potential_point, False
                    else:
                        return angle, potential_point, False

    def find_tangent_points(self, previous_polygon, polygon, ext_size_for_loop, diameter, is_rectangle, is_backward):
        # Check if the previous polygon should be treated as a rectangle or not
        if is_rectangle:
            # Convert the previous polygon's coordinates into a rectangle
            rec_cor = self.polygon_to_rectangle(previous_polygon.coordinates)
            # Find points where the line crosses the polygon's boundary using the rectangle coordinates
            points = self.check_if_line_crosses_the_polygon(rec_cor, polygon.coordinates)
        else:
            # Find points where the line crosses the polygon's boundary using the original polygon coordinates
            points = self.check_if_line_crosses_the_polygon(previous_polygon.coordinates, polygon.coordinates)

        # Iterate through all points where the line crosses the polygon
        for tangent_point_from in points:
            # Calculate the angle between the tangent point and the leftmost point of the previous polygon
            angle = self.calculate_angle_in_degrees(tangent_point_from, previous_polygon.left_point)

            # Call the find_potential_tangent_point function to get the adjusted angle and the next tangent point
            angle, tangent_point_to, break_from_all = self.find_potential_tangent_point(
                angle,
                previous_polygon,
                tangent_point_from,
                self.find_most_left_or_right_point,  # Function to find the most left/right point
                ext_size_for_loop,
                diameter,
                is_rectangle,
                is_backward
            )

            # If the find_potential_tangent_point function signals a timeout or failure, return early
            if break_from_all:
                return 1, None, None, None, None, None, False  # Signal failure with a flag

            # Adjust the angle slightly depending on whether we are moving backward or forward
            if is_backward:
                angle = (angle - 0.01 % 360)
            else:
                angle = (angle + 0.01 % 360)

            # Try to place the polygon based on the calculated angle and other parameters
            flag, extended_poly, right_li, left_li = self.placement(
                angle,
                polygon.coordinates,
                previous_polygon,
                diameter
            )

            # If the placement was successful (flag is True), return all relevant information
            if flag:
                return angle, tangent_point_from, tangent_point_to, extended_poly, right_li, left_li, True

        # If no valid tangent points were found or placement failed, return failure signal
        return None, None, None, None, None, None, False

    def update_convex_region(self, angle, convex_region, less_detailed_convex_region, polygon, extension_size, midpoint,
                             target_point, current_angle, is_blue_region_active, is_pink_region_active,
                             half_detailed_convex_region, dimension, first_polygon_flag):

        # Calculate the angle and normalize it to the range [0, 360)
        normalized_angle = self.calculate_angle_in_degrees(midpoint, target_point) % 360

        # Extend the polygon based on the angle
        extended_polygon_coordinates = self.extend_polygon_with_angle(angle, convex_region, polygon, extension_size,
                                                                      dimension)
        # Generate the new regions for both detailed and less detailed versions
        new_detailed_region = self.merge_polygon_with_boundary(Polygon(convex_region), polygon, extension_size)
        new_less_detailed_region = self.merge_polygon_with_boundary(Polygon(less_detailed_convex_region),
                                                                    extended_polygon_coordinates)

        # Update the convex region based on the angle
        # If angle is in the pink region (0 <= angle < 180)
        if 0 <= normalized_angle < 180:
            if is_pink_region_active:
                convex_region = new_detailed_region
                half_detailed_convex_region = self.merge_polygon_with_boundary(Polygon(half_detailed_convex_region),
                                                                               polygon, extension_size)
            else:
                if is_blue_region_active:
                    half_detailed_convex_region = self.merge_polygon_with_boundary(Polygon(half_detailed_convex_region),
                                                                                   polygon, extension_size)
                    convex_region = half_detailed_convex_region
                    is_blue_region_active = False
                else:
                    convex_region = new_detailed_region

                half_detailed_convex_region = self.merge_polygon_with_boundary(Polygon(less_detailed_convex_region),
                                                                               polygon, extension_size)
                if not first_polygon_flag:
                    is_pink_region_active = True

        # If angle is in the blue region (180 <= angle < 360)
        else:
            if is_blue_region_active:
                convex_region = new_detailed_region
                half_detailed_convex_region = self.merge_polygon_with_boundary(Polygon(half_detailed_convex_region),
                                                                               polygon, extension_size)
            else:
                half_detailed_convex_region = self.merge_polygon_with_boundary(Polygon(half_detailed_convex_region),
                                                                               polygon, extension_size)
                convex_region = half_detailed_convex_region
                half_detailed_convex_region = self.merge_polygon_with_boundary(Polygon(less_detailed_convex_region),
                                                                               polygon, extension_size)
                is_blue_region_active = True
                is_pink_region_active = False

        less_detailed_convex_region = new_less_detailed_region
        return convex_region, less_detailed_convex_region, current_angle, is_blue_region_active, is_pink_region_active, half_detailed_convex_region

    def polygon_dimensions(self, vertices):
        # Create a Polygon object
        polygon = Polygon(vertices)

        # Get the minimum rotated rectangle
        min_rotated_rect = polygon.minimum_rotated_rectangle

        # Get the coordinates of the rectangle's vertices
        rect_coordinates = list(min_rotated_rect.exterior.coords)[:-1]  # Last point is the same as the first, so we skip it

        # Define the sides of the minimum rotated rectangle
        width_points = rect_coordinates[0], rect_coordinates[1]
        height_points = rect_coordinates[1], rect_coordinates[2]

        # Calculate the width and height
        width = sqrt((width_points[1][0] - width_points[0][0]) ** 2 + (width_points[1][1] - width_points[0][1]) ** 2)
        height = sqrt(
            (height_points[1][0] - height_points[0][0]) ** 2 + (height_points[1][1] - height_points[0][1]) ** 2)

        # Return the width, height, and the start and end points for width and height
        return width, height

    def polygon_best_angle_for_rec(self, vertices):
        # Create a Polygon object
        polygon = Polygon(vertices)

        # Get the minimum rotated rectangle
        min_rotated_rect = polygon.minimum_rotated_rectangle

        # Get the coordinates of the rectangle's vertices
        rect_coordinates = list(min_rotated_rect.exterior.coords)[:-1]  # Last point is the same as the first, so we skip it

        # Define the corners of the minimum rotated rectangle
        width_points = rect_coordinates[0], rect_coordinates[1]
        height_points = rect_coordinates[1], rect_coordinates[2]

        # Calculate the width and height
        width = sqrt((width_points[1][0] - width_points[0][0]) ** 2 + (width_points[1][1] - width_points[0][1]) ** 2)
        height = sqrt(
            (height_points[1][0] - height_points[0][0]) ** 2 + (height_points[1][1] - height_points[0][1]) ** 2)

        # Calculate the angles for width and height in degrees
        width_angle = self.calculate_angle_in_degrees(width_points[0], width_points[1])
        height_angle = self.calculate_angle_in_degrees(height_points[0], height_points[1])

        # Determine whether width or height is larger and return the corresponding angle
        if width > height:
            return width_angle
        else:
            return height_angle

    def is_close_to_square(self, vertices, tolerance_difference=0.35):
        width, height = self.polygon_dimensions(vertices)

        # Check the difference and ratio
        relative_difference = abs(width - height) / max(width, height)

        # Check if the difference and ratio are within the tolerance
        return relative_difference <= tolerance_difference

    def find_polygon_diameter_angle(self, vertices):
        # Create a Polygon object
        polygon = Polygon(vertices)

        # Ensure the polygon is valid
        if not polygon.is_valid:
            raise ValueError("The input polygon must be valid.")

        # Get the exterior coordinates of the convex polygon
        points = list(polygon.exterior.coords[:-1])  # The last point is a repeat of the first one

        # Number of vertices
        n = len(points)

        # Initialize the maximum distance (diameter) and the points that form it
        max_distance = 0
        start_point = None
        end_point = None

        # Rotating Calipers to find the diameter
        k = 1
        for i in range(n):
            while True:
                next_k = (k + 1) % n
                current_line = LineString([points[i], points[k]])
                current_distance = current_line.length
                next_distance = LineString([points[i], points[next_k]]).length

                if next_distance > current_distance:
                    k = next_k
                else:
                    break

            if current_distance > max_distance:
                max_distance = current_distance
                start_point = points[i]
                end_point = points[k]

        angle = self.calculate_angle_in_degrees(start_point, end_point)

        return angle

    def find_perpendicular_line(self, angle, through_point, length):
        # Normalize the angle to be within [0, 360) degrees
        angle = angle % 360

        # Find the perpendicular angle (90 degrees clockwise or counterclockwise)
        perpendicular_angle = (angle + 90) % 360  # Perpendicular angle

        # Convert the perpendicular angle to radians for trigonometric functions
        perpendicular_angle_rad = math.radians(perpendicular_angle)

        # Calculate the half-length for easier computation
        half_length = length / 2

        # Calculate the change in x and y using the perpendicular angle
        dx = half_length * math.cos(perpendicular_angle_rad)
        dy = half_length * math.sin(perpendicular_angle_rad)

        # The through_point is the midpoint, so calculate the two endpoints
        x1 = through_point[0] - dx
        y1 = through_point[1] - dy
        x2 = through_point[0] + dx
        y2 = through_point[1] + dy

        # Return the endpoints of the perpendicular line
        return [(x1, y1), (x2, y2)]

    def polygon_best_angle_for_none_rec(self, polygon_coordinates, initial_diameter):
        # Calculate the centroid of the polygon
        centroid = self.calculate_centroid(polygon_coordinates)
        expanded_diameter = initial_diameter * 2

        # Initialize variables for tracking the best angle and minimal difference
        minimal_difference = float('inf')
        optimal_angle = 0
        valid_split_found = False

        # Iterate through each edge of the polygon
        for i in range(len(polygon_coordinates)):
            start_vertex = polygon_coordinates[i]
            end_vertex = polygon_coordinates[(i + 1) % len(polygon_coordinates)]
            edge_angle = self.calculate_angle_in_degrees(start_vertex, end_vertex)

            perpendicular_line = self.find_perpendicular_line(edge_angle, centroid, expanded_diameter)

            split_result = split(Polygon(polygon_coordinates), LineString(perpendicular_line))
            convex_polygons = []

            # Check if the result is a GeometryCollection
            if split_result.geom_type == 'GeometryCollection':
                for geometry in split_result.geoms:
                    if isinstance(geometry, Polygon):
                        convex_polygons.append(list(geometry.exterior.coords))

            # Check if the resulting polygons are close to square
            is_first_polygon_square = self.is_close_to_square(convex_polygons[0])
            is_second_polygon_square = self.is_close_to_square(convex_polygons[1])

            if is_first_polygon_square and is_second_polygon_square:
                valid_split_found = True

                first_polygon_dimensions = self.polygon_dimensions(convex_polygons[0])
                second_polygon_dimensions = self.polygon_dimensions(convex_polygons[1])

                width_difference = abs(first_polygon_dimensions[0] - second_polygon_dimensions[0])
                height_difference = abs(first_polygon_dimensions[1] - second_polygon_dimensions[1])

                overall_difference = abs(width_difference - height_difference)

                if overall_difference < minimal_difference:
                    minimal_difference = overall_difference
                    optimal_angle = edge_angle

        # Check the largest angle and diameter angle
        largest_angle = self.polygon_best_angle_for_rec(polygon_coordinates)
        diameter_angle = self.find_polygon_diameter_angle(polygon_coordinates)

        angles_to_evaluate = [largest_angle, diameter_angle]
        for angle in angles_to_evaluate:
            perpendicular_line = self.find_perpendicular_line(angle, centroid, expanded_diameter)
            split_result = split(Polygon(polygon_coordinates), LineString(perpendicular_line))
            convex_polygons = []

            # Check if the result is a GeometryCollection
            if split_result.geom_type == 'GeometryCollection':
                for geometry in split_result.geoms:
                    if isinstance(geometry, Polygon):
                        convex_polygons.append(list(geometry.exterior.coords))

            # Check if the resulting polygons are close to square
            is_first_polygon_square = self.is_close_to_square(convex_polygons[0])
            is_second_polygon_square = self.is_close_to_square(convex_polygons[1])

            if is_first_polygon_square and is_second_polygon_square:
                valid_split_found = True

                first_polygon_dimensions = self.polygon_dimensions(convex_polygons[0])
                second_polygon_dimensions = self.polygon_dimensions(convex_polygons[1])

                width_difference = abs(first_polygon_dimensions[0] - second_polygon_dimensions[0])
                height_difference = abs(first_polygon_dimensions[1] - second_polygon_dimensions[1])

                overall_difference = abs(width_difference - height_difference)

                if overall_difference < minimal_difference:
                    minimal_difference = overall_difference
                    optimal_angle = angle

        # If no valid split was found, return the largest angle
        if not valid_split_found:
            optimal_angle = self.polygon_best_angle_for_rec(polygon_coordinates)

        return optimal_angle

    def recursive_split(self, polygon_coordinates, max_depth, diameter, current_depth=0, use_other=False):
        # Check if the polygon is already close to a square or if the recursion has reached the maximum depth
        split_flag = self.is_close_to_square(polygon_coordinates)

        # If the polygon is square-like or maximum recursion depth is reached, return the current polygon
        if split_flag or current_depth >= max_depth:
            return [polygon_coordinates]
        else:
            # Determine the method for calculating the best angle for splitting
            if use_other:
                # Use non-recursive method to calculate the best angle for splitting
                angle_for_split = self.polygon_best_angle_for_none_rec(polygon_coordinates, diameter)
            else:
                # Use recursive method to calculate the best angle for splitting
                angle_for_split = self.polygon_best_angle_for_rec(polygon_coordinates)

            # Calculate the centroid of the polygon to find the splitting point
            middle_point = self.calculate_centroid(polygon_coordinates)

            # Find a perpendicular line based on the calculated angle and the centroid for the split
            per_line = self.find_perpendicular_line(angle_for_split, middle_point, diameter)

            # Split the polygon along the calculated perpendicular line
            split_polygons = split(Polygon(polygon_coordinates), LineString(per_line))
            list_of_convex_regions = []

            # Check if the split operation returned a GeometryCollection (multiple geometries)
            if split_polygons.geom_type == 'GeometryCollection':
                for geom in split_polygons.geoms:
                    if isinstance(geom, Polygon):
                        # Recursively apply the split operation to each resulting polygon
                        list_of_convex_regions.extend(
                            self.recursive_split(list(geom.exterior.coords), max_depth, diameter, current_depth + 1,
                                                 use_other))

            # Return the list of split polygons (convex regions)
            return list_of_convex_regions

    def find_left_and_right_points(self, center, angle, middle_polygon, dime):
        # Extract x and y coordinates of the center point
        center_x, center_y = center

        # Calculate direction vector based on the given angle
        direction_x, direction_y = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))

        # Classify points of the middle polygon as left or right based on the angle and center point
        left_points, right_points = self.classify_points_left_right(angle, center, middle_polygon)

        # Calculate the endpoint of the line extending in the direction of the angle from the center point
        endpoint_left_x, endpoint_left_y = self.calculate_endpoint_from_direction(center_x, center_y, direction_x,
                                                                                  direction_y, dime)

        # Calculate the opposite angle (180 degrees away) to find the other direction
        opposite_angle = (angle + 180) % 360

        # Calculate direction vector based on the opposite angle
        opposite_direction_x, opposite_direction_y = (
            math.cos(math.radians(opposite_angle)), math.sin(math.radians(opposite_angle)))

        # Calculate the endpoint of the line extending in the opposite direction from the center point
        endpoint_right_x, endpoint_right_y = self.calculate_endpoint_from_direction(center_x, center_y,
                                                                                    opposite_direction_x,
                                                                                    opposite_direction_y, dime)

        # Create a line segment using the left and right endpoints
        line_segment = [(endpoint_right_x, endpoint_right_y), (endpoint_left_x, endpoint_left_y)]

        # Find the farthest point on the right side of the polygon from the line segment
        farthest_right_x, farthest_right_y = self.find_farthest_point_from_line(line_segment, right_points,
                                                                                middle_polygon, direction_x,
                                                                                direction_y, dime)

        # Find the farthest point on the left side of the polygon from the line segment
        farthest_left_x, farthest_left_y = self.find_farthest_point_from_line(line_segment, left_points, middle_polygon,
                                                                              direction_x, direction_y, dime)

        # Calculate the endpoint of a line extending from the farthest right point in the direction of the angle
        right_endpoint = self.calculate_endpoint_from_direction(farthest_right_x, farthest_right_y, direction_x,
                                                                direction_y, dime)

        # Calculate the endpoint of a line extending from the farthest left point in the direction of the angle
        left_endpoint = self.calculate_endpoint_from_direction(farthest_left_x, farthest_left_y, direction_x,
                                                               direction_y, dime)

        # Set the left and right points to the farthest left and right points found
        left_point = (farthest_left_x, farthest_left_y)
        right_point = (farthest_right_x, farthest_right_y)

        # Return the left and right points along with their corresponding endpoints
        return left_point, right_point, left_endpoint, right_endpoint

    def handle_polygon_movement(self, polygon, from_point, to_point, angle, middle_point, convex_region_pol_object,
                                previous_polygon, convex_region, x, y, dime):
        # Create vertical and horizontal lines from the convex region
        vertical_line, horizontal_line = self.create_lines(convex_region)

        # Determine the target point by intersecting lines with the given angle and dimensions
        intersection_point = self.intersection_of_lines(vertical_line, horizontal_line, angle, to_point, dime)

        # Nested function to handle the movement of the polygon to a target point
        def attempt_polygon_move(target_point):
            proposed_translation = self.get_translation_by_point(polygon.coordinates, from_point, target_point)

            # Check if the polygon stays within the allowed region after the move
            if Polygon(proposed_translation).within(convex_region_pol_object):
                polygon.set_translation_by_point(from_point, target_point)
            else:
                # If not, adjust using x, y offsets
                polygon.set_translation_by_center(x, y)

        # If the intersection_point is valid (a tuple)
        if isinstance(intersection_point, tuple):
            movement_line = LineString([intersection_point, to_point])
            proposed_move = self.get_translation_by_point(polygon.coordinates, from_point, intersection_point)

            # Create an enlarged version of the previous polygon for intersection checks
            enlarged_previous_polygon = Polygon(previous_polygon.ext_size_for_loop)
            is_intersecting = enlarged_previous_polygon.intersects(Polygon(proposed_move))

            # Check movement conditions before applying the move
            if (Polygon(proposed_move).within(convex_region_pol_object) and
                    not movement_line.crosses(convex_region_pol_object) and
                    Polygon(polygon.coordinates).within(convex_region_pol_object) and
                    not is_intersecting):
                polygon.set_translation_by_point(from_point, intersection_point)
            else:
                # Use middle_point as fallback if conditions fail
                attempt_polygon_move(middle_point)
        else:
            # If no valid intersection, move to the middle point
            attempt_polygon_move(middle_point)

    def find_best_point(self, polygon, previous_polygon, right_line, from_point, to_point):

        # Move the polygon based on right_li and angle a
        proposed_translation = self.get_translation_by_point(polygon.coordinates, list(right_line.coords)[0], to_point)

        # Check for intersection with the previous polygon's extended size
        check_inter = Polygon(previous_polygon.ext_size_for_loop).intersects(Polygon(proposed_translation))

        # Determine the point to move to
        point = list(right_line.coords)[0] if not check_inter else from_point

        return point

    def move_back_and_check_intersection(self, polygon, previous_polygon, middle_point, angle):
        # Calculate the initial distance
        distance_from_middle_to_pol = Point(middle_point).distance(Polygon(polygon.coordinates))

        # Move the polygon by the calculated distance and angle
        polygon.set_translation_by_distance_and_angle(distance_from_middle_to_pol, angle)

        # Adjust the angle by 180 degrees
        new_angle = (angle + 180) % 360

        # Calculate the second distance
        distance_from_current_pol_to_previous_pol = Polygon(polygon.coordinates).distance(Polygon(previous_polygon.coordinates))

        # Move the polygon by the new distance and angle
        proposed_translation = self.get_translation_by_distance_and_angle(polygon.coordinates, distance_from_current_pol_to_previous_pol, new_angle)

        # Create an expanded polygon from the previous polygon's ext_size_for_loop
        ext_polygon = Polygon(previous_polygon.ext_size_for_loop)

        # Check if the expanded polygon intersects with the moved polygon
        intersection_exists = ext_polygon.intersects(Polygon(proposed_translation))

        if intersection_exists:
            # Recalculate the distance if there is an intersection
            distance_from_current_pol_to_previous_extended_pol = Polygon(polygon.coordinates).distance(Polygon(previous_polygon.ext_size_for_loop))
            # Move the polygon by the new distance and angle
            polygon.set_translation_by_distance_and_angle(distance_from_current_pol_to_previous_extended_pol, new_angle)
        else:
            # Move the polygon by the previously calculated distance and angle
            polygon.set_translation_by_distance_and_angle(distance_from_current_pol_to_previous_pol, new_angle)

    def final_polygon_update(self, polygon, f_p, t_p, value, list_of_polygons):
        # Set the translation of the polygon from the first point (f_p) to the target point (t_p)
        polygon.set_translation_by_point(f_p, t_p)

        # Create a deep copy of the polygon to avoid modifying the original polygon
        copied = copy.deepcopy(polygon)

        # Convert the polygon's coordinates to integers for more efficient handling
        con = copied.convert_coordinates_to_int()

        # Update the copied polygon's coordinates with the converted integer values
        copied.set_coordinates(con)

        # Add the updated polygon copy to the list of polygons
        list_of_polygons.append(copied)

        # Update the value by adding the polygon's value
        value = value + polygon.value

        # Return the updated value and the modified list of polygons
        return value, list_of_polygons

    def get_translation_by_center(self, coordinates, new_center_x, new_center_y):
        convex_polygon = Polygon(coordinates)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid
        current_center_x = centroid.x
        current_center_y = centroid.y

        # Calculate the translation vector to move the center to the new position
        translation_x = new_center_x - current_center_x
        translation_y = new_center_y - current_center_y

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in coordinates]

    def get_translation_by_point(self, coordinates, point_of_pol, point_of_region):
        # Calculate the translation vector
        translation_x = point_of_region[0] - point_of_pol[0]
        translation_y = point_of_region[1] - point_of_pol[1]

        # Update all coordinates by adding the translation vector
        return [(x + translation_x, y + translation_y) for x, y in coordinates]

    def get_translation_by_distance_and_angle(self, coordinates, distance, angle_degrees):
        # Convert angle from degrees to radians
        angle_radians = math.radians(angle_degrees)

        # Calculate the current center point of the item
        convex_polygon = Polygon(coordinates)

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
        return [(x + translation_x, y + translation_y) for x, y in coordinates]

    def run_algorithm(self):
        # Record the start time for performance tracking
        start_time = time.time()

        # Sort the items by their value-to-dimension ratio in descending order
        sorted_items = sorted(self.item_instances, key=lambda item: (item.value / item.get_largest_dimension()),
                              reverse=True)

        # Calculate the total dimensions of the container
        diameter = self.container_instance.get_largest_dimension()

        # Check if the container is close to being a square
        split_flag = self.is_close_to_square(self.container_instance.coordinates)

        # If the container is square-like, directly plot the items
        if split_flag:
            list_of_polygons, value = self.packing_algorithm(self.container_instance.coordinates, sorted_items, diameter)
        else:
            # If not square-like, check the number of polygon vertices (4 = rectangle/square)
            len_pol = len(self.container_instance.coordinates)

            # Perform recursive splitting based on the number of polygon vertices
            if len_pol == 4:
                split_list = self.recursive_split(self.container_instance.coordinates, 100, diameter * 2, False)
            else:
                split_list = self.recursive_split(self.container_instance.coordinates, 100, diameter * 2, True)

            # Get the number of resulting split polygons
            split_num = len(split_list)
            n = split_num

            # Initialize a list of empty lists to store the sorted items into split regions
            list_of_lists = [[] for _ in range(n)]

            # Distribute the sorted items among the split polygons using modulo indexing
            for dex, polygon in enumerate(sorted_items):
                list_of_lists[dex % n].append(polygon)

            # Initialize variables to store the final list of polygons and the total value
            final_lists = []
            sum_val = 0

            # Plot items into each split polygon and update the total value
            for i in range(n):
                final_list, val = self.packing_algorithm(split_list[i], list_of_lists[i], diameter)
                sum_val = val + sum_val  # Accumulate the value from each split
                final_lists.extend(final_list)  # Extend the final list with the polygons from this iteration

            # Store the final polygons and value after all splits are processed
            list_of_polygons = final_lists
            value = sum_val

        # Calculate the elapsed time for the algorithm
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Print the number of polygons processed, total items, execution time, and total value
        print(f"Number of polygons: {len(list_of_polygons)} out of {len(self.item_instances)}, "
              f"Time: {elapsed_time:.2f} seconds, Value: {value}")

        return list_of_polygons

    def packing_algorithm(self, convex_region, list_pol, diameter):
        # Shrink the convex region slightly for better packing efficiency
        convex_region_shrink = self.shrink_polygon(1, convex_region)

        # Initialize angle and other relevant variables
        angle = 0
        current_angle = 0  # Starting angle for placing polygons
        ext_size = 1
        ext_size_for_loop = 3
        ext_size_for_loop_rec = 4

        # Calculate the centroid (middle point) of the shrunken convex region
        middle_point = self.calculate_centroid(convex_region_shrink)

        # Initialize different levels of detail for the convex region
        convex_region = convex_region_shrink
        convex_region_original = convex_region_shrink
        convex_region_less_detailed = convex_region_shrink
        half_detailed_convex_region = convex_region_shrink

        # Initialize flags and tracking variables
        pink_in = False
        blue_in = False
        first_polygon = True
        result_list = []
        value = 0
        previous_polygon = None

        # Iterate through the list of polygons to place them
        for dex, polygon in enumerate(list_pol):
            can_place_polygon = False
            extended_polygon = None
            right_line = None
            left_line = None

            print(dex)  # Debugging statement to track index

            # Set polygon translation to the center of the region
            x, y = middle_point
            polygon.set_translation_by_center(x, y)

            # Create polygon objects from the coordinates for geometric operations
            current_polygon = Polygon(polygon.coordinates)
            convex_region_polygon = Polygon(convex_region)

            # Check if the polygon fits within the current convex region
            if current_polygon.within(convex_region_polygon):
                if first_polygon:
                    # Handle placement of the first polygon
                    first_polygon = False
                    _, extended_polygon, right_line, left_line = self.placement(angle, polygon.coordinates, None,
                                                                                diameter)
                    can_place_polygon = True
                    first_polygon_flag = True

                else:
                    # Handle subsequent polygons
                    first_polygon_flag = False
                    previous_polygon.ext_size_for_loop = self.extend_polygon(previous_polygon.coordinates,
                                                                             ext_size_for_loop)
                    rec_pol = self.polygon_to_rectangle(previous_polygon.ext_size_for_loop)
                    intersects = (Polygon(polygon.coordinates)).intersects((Polygon(rec_pol)))

                    if not intersects:
                        # Further extend the previous polygon for the operations
                        previous_polygon.ext_size = self.extend_polygon(previous_polygon.coordinates, ext_size)
                        previous_polygon.ext_size_for_loop_rec = self.extend_polygon(previous_polygon.coordinates,
                                                                                     ext_size_for_loop_rec)
                        move_back = False
                        within = False

                        # Loop to find the best tangent points and placement angle
                        for j_index in range(3):
                            if j_index == 0:
                                angle, right_point, left_point, extended_poly, right_li, left_li, found = self.find_tangent_points(
                                    previous_polygon, polygon, ext_size_for_loop_rec, diameter, True, False)
                            else:
                                angle, right_point, left_point, extended_poly, right_li, left_li, found = self.find_tangent_points(
                                    previous_polygon, polygon, ext_size_for_loop, diameter, False, False)

                            if found:
                                # If tangent points are found, update the extended polygon and lines
                                extended_polygon = extended_poly
                                right_line = right_li
                                left_line = left_li

                                if j_index == 0:
                                    # Move the polygon to the best found points
                                    right_point = self.find_best_point(polygon, previous_polygon, right_line,
                                                                       right_point, left_point)
                                    polygon.set_translation_by_point(right_point, left_point)

                                    if Polygon(polygon.coordinates).within(convex_region_polygon):
                                        within = True
                                        continue
                                    else:
                                        # Handle cases where the polygon is partially within the original region
                                        go_to = False
                                        if Polygon(polygon.coordinates).within(
                                                Polygon(convex_region_original)) and not Polygon(
                                                polygon.coordinates).within(convex_region_polygon):
                                            go_to = True
                                        angle, right_point, left_point, extended_poly, right_li, left_li, found = self.find_tangent_points(
                                            previous_polygon, polygon, ext_size_for_loop, diameter, False, True)
                                        if found:
                                            # Check for intersection and possibly move the polygon back
                                            self.move_back_and_check_intersection(polygon, previous_polygon,
                                                                                  middle_point, angle)
                                            if go_to:
                                                move_back = True
                                            else:
                                                if not Polygon(polygon.coordinates).within(convex_region_polygon):
                                                    move_back = True
                                            continue

                                if j_index == 1 and move_back:
                                    # Handle polygon movement and intersection checks
                                    self.handle_polygon_movement(polygon, right_point, left_point, angle, middle_point,
                                                                 convex_region_polygon, previous_polygon, convex_region,
                                                                 x, y, diameter)
                                elif j_index == 1 and within:
                                    can_place_polygon = True
                                    break
                                elif j_index == 2:
                                    can_place_polygon = True

                            if not found:
                                break

                if can_place_polygon:
                    # Determine the best placement for the polygon and update the final list
                    from_point, to_point = self.find_best_placement_for_polygon(polygon.coordinates, extended_polygon,
                                                                                convex_region, angle, right_line,
                                                                                left_line, diameter)
                    value, result_list = self.final_polygon_update(polygon, from_point, to_point, value, result_list)

                    # Find the most extreme left point for further processing
                    left_point, _ = self.find_most_left_or_right_point(angle, polygon.coordinates, None,
                                                                       ext_size_for_loop, False, diameter)
                    polygon.left_point = left_point

                    # Update the convex region and its detailed versions
                    convex_region, convex_region_less_detailed, current_angle, blue_in, pink_in, half_detailed_convex_region = self.update_convex_region(
                        angle, convex_region, convex_region_less_detailed, polygon.coordinates, ext_size, middle_point,
                        to_point, current_angle, blue_in, pink_in, half_detailed_convex_region, diameter, first_polygon_flag)

                    # Set the previous polygon to the current one for the next iteration
                    previous_polygon = polygon

                    # Update the middle point based on the new convex region
                    middle_point = self.calculate_centroid(convex_region)
                    print("placed")

        # Return the final list of placed polygons and the total value
        return result_list, value



