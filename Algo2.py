import random
import copy
from Draw import Draw
import math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point
import numpy as np

class Algo2:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def calculate_midpoint(self, x1, y1, x2, y2):
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        return midpoint_x, midpoint_y

    def is_point_between(self, x, y, x1, y1, x2, y2):
        """
        Check if a point is between two other points.

        Args:
        point: A tuple (x, y) representing the point to check.
        start_point: A tuple (x1, y1) representing the starting point.
        end_point: A tuple (x2, y2) representing the ending point.

        Returns:
        True if the point is between the two reference points, False otherwise.
        """
        # Check if the x-coordinate of the point is between the x-coordinates of the start and end points,
        # and if the y-coordinate of the point is between the y-coordinates of the start and end points.
        if (x1 <= x <= x2 or x2 <= x <= x1) and (y1 <= y <= y2 or y2 <= y <= y1):
            return True

        return False
    def point_on_edge_closest_to_point(self,edge_start, edge_end, original_point):
        x1, y1 = edge_start
        x2, y2 = edge_end
        x0, y0 = original_point

        # Calculate the direction vector of the line formed by the edge
        dx = x2 - x1
        dy = y2 - y1

        # Calculate the parameter (t) for the point on the line closest to the original point
        t = ((x0 - x1) * dx + (y0 - y1) * dy) / (dx ** 2 + dy ** 2)

        # Find the coordinates of the closest point on the line
        x = x1 + t * dx
        y = y1 + t * dy

        return (x, y)

    def point_on_edge_closest_to_point1(self,start_point, end_point, given_point):
        # Calculate the direction vector of the edge.
        edge_vector = (end_point[0] - start_point[0], end_point[1] - start_point[1])

        # Calculate the vector from the start point to the given point.
        start_to_given_vector = (given_point[0] - start_point[0], given_point[1] - start_point[1])

        # Calculate the length of the edge vector.
        edge_length = math.dist(start_point, end_point)

        if edge_length == 0:
            # The edge is just a point, return that point.
            return start_point

        # Calculate the distance from the start point to the given point.
        distance_to_start = math.dist(start_point, given_point)

        # Calculate the dot product to project start_to_given_vector onto the edge_vector.
        dot_product = (start_to_given_vector[0] * edge_vector[0] + start_to_given_vector[1] * edge_vector[
            1]) / edge_length ** 2

        if dot_product <= 0:
            # The closest point is the start point.
            return start_point
        elif dot_product >= 1:
            # The closest point is the end point.
            return end_point

        # Calculate the closest point on the edge using the distance.
        closest_point = (
            start_point[0] + dot_product * edge_vector[0],
            start_point[1] + dot_product * edge_vector[1]
        )

        return closest_point


    def edge_cover(self, p1, p2, q1, q2):
        x1, y1 = p1
        x2, y2 = p2
        f1, f2 = q1
        t1, t2 = q2
        ed1 = self.edge_length(x1, y1, x2, y2)
        ed2 = self.edge_length(f1, f2, t1, t2)
        x = None
        y = None
        if ed1 > ed2:
            midx, midy = self.calculate_midpoint(f1, f2, t1, t2)
            x, y = self.point_on_edge_closest_to_point(p1, p2, (midx, midy))

            return self.is_point_between(x, y, x1, y1, x2, y2)

        elif ed1 < ed2:
            midx, midy = self.calculate_midpoint(x1, y1, x2, y2)
            x, y = self.point_on_edge_closest_to_point(q1, q2, (midx, midy))


            return self.is_point_between(x, y, f1, f2, t1, t2)

        else:
            midx, midy = self.calculate_midpoint(x1, y1, x2, y2)
            x, y = self.point_on_edge_closest_to_point(q1, q2, (midx, midy))
            return self.is_point_between(x, y, f1, f2, t1, t2)
    def edge_length(self,x1, y1, x2, y2):
        # Calculate the length of the edge using the Euclidean distance formula
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length
    def the_closest_edges(self, polygon, edge_of_region):
        min_distance = float('inf')
        (q1, q2) = edge_of_region
        l2 = LineString([q1, q2])
        min_edge = None
        for (p1, p2) in polygon.get_edge_lines():
            l1 = LineString([p1, p2])
            dist = l1.distance(l2)
            if min_distance > dist:
                min_distance = dist
                min_edge = (p1, p2)
        return min_edge

    def closest_points_on_edges(self,edge1, edge2):
        # Convert the edges to numpy arrays for vector operations
        edge1 = np.array([edge1])
        edge2 = np.array([edge2])

        # Calculate the vector from the first point of edge1 to the first point of edge2
        v1 = edge2[0] - edge1[0]

        # Calculate the vector along edge1
        edge1_direction = edge1[1] - edge1[0]

        # Calculate the projection of v1 onto edge1_direction
        t1 = np.dot(v1, edge1_direction) / np.dot(edge1_direction, edge1_direction)

        # Ensure the projection is within the bounds of edge1
        t1 = max(0, min(1, t1))

        # Calculate the closest point on edge1
        closest_point1 = edge1[0] + t1 * edge1_direction

        # Calculate the vector from the first point of edge2 to the closest point on edge1
        v2 = closest_point1 - edge2[0]

        # Calculate the vector along edge2
        edge2_direction = edge2[1] - edge2[0]

        # Calculate the projection of v2 onto edge2_direction
        t2 = np.dot(v2, edge2_direction) / np.dot(edge2_direction, edge2_direction)

        # Ensure the projection is within the bounds of edge2
        t2 = max(0, min(1, t2))

        # Calculate the closest point on edge2
        closest_point2 = edge2[0] + t2 * edge2_direction

        return closest_point1, closest_point2

    def is_counterclockwise(self, coordinates):
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
            return coordinates
        # Calculate the centroid
        centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
        centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
        centroid = (centroid_x, centroid_y)

        # Sort the coordinates based on angles
        sorted_coordinates = sorted(coordinates, key=lambda point: self.calculate_angle(point, centroid))

        return sorted_coordinates

    def for_almost_close_edges(self,po1, po2, po3,pol1,pol2):
        counter = self.order_coordinates_counterclockwise([po1, po2, po3])

        pol4 = Polygon(counter)
        buffered_result = pol4.buffer(0.01)

        # Assuming pol1 and pol4 are Shapely Polygon objects

        mergedPolys1 = pol1.difference(buffered_result)

        exterior_coords_list1 = []
        if isinstance(mergedPolys1, MultiPolygon):
            # Iterate through the constituent polygons
            for polygon in mergedPolys1.geoms:
                # Get the coordinates of the exterior boundary of each polygon
                exterior_coords = list(polygon.exterior.coords)
                # Append them to the exterior_coords_list
                exterior_coords_list1.extend(exterior_coords)
        else:
            # If it's a single Polygon, get its exterior coordinates directly
             exterior_coords_list1 = list(mergedPolys1.exterior.coords)
        #exterior_coords_list1 = self.remove_duplicate_coordinates(exterior_coords_list1)


        mergedPolys = Polygon(exterior_coords_list1).difference(pol2)

        exterior_coords_list = []
        if isinstance(mergedPolys, MultiPolygon):
            # Iterate through the constituent polygons
            for polygon in mergedPolys.geoms:
                # Get the coordinates of the exterior boundary of each polygon
                exterior_coords = list(polygon.exterior.coords)
                # Append them to the exterior_coords_list
                return exterior_coords_list.extend(exterior_coords)
        else:
            # If it's a single Polygon, get its exterior coordinates directly
            return list(mergedPolys.exterior.coords)

    def for_edges_that_intersect(self, pol1, pol2):
        mergedPolys = pol1.difference(pol2)
        exterior_coords_list = []
        if isinstance(mergedPolys, MultiPolygon):
            # Iterate through the constituent polygons
            for polygon in mergedPolys.geoms:
                # Get the coordinates of the exterior boundary of each polygon
                exterior_coords = list(polygon.exterior.coords)
                # Append them to the exterior_coords_list
                return exterior_coords_list.extend(exterior_coords)
        else:
            # If it's a single Polygon, get its exterior coordinates directly
            return list(mergedPolys.exterior.coords)

    def get_edge_lines(self,coord):
        edges = []
        num_points = len(coord)

        for i in range(num_points):
            point1 = coord[i]
            point2 = coord[(i + 1) % num_points]  # Wrap around to the first point

            line = (point1, point2)
            edges.append(line)

        return edges

    def smaller_angle_points(self,edge1, edge2):
        # Calculate the unit vectors for each edge
        vector1 = (edge1[1][0] - edge1[0][0], edge1[1][1] - edge1[0][1])
        vector2 = (edge2[1][0] - edge2[0][0], edge2[1][1] - edge2[0][1])

        # Calculate the angle between the two vectors
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]
        norm1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        norm2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)
        cosine_similarity = dot_product / (norm1 * norm2)
        angle = math.acos(cosine_similarity)

        # Determine which pair of points has the smaller angle
        if angle < math.pi / 2:  # Check if the angle is acute
            return edge1[1], edge2[1]
        else:
            return edge1[0], edge2[0]
    def find_good_edge(self,polygon,convex_region_coords):
        index = 0
        edges_intersect = False
        edges_close = False
        for (p1, p2) in polygon.get_edge_lines():
            for (q1, q2) in self.get_edge_lines(convex_region_coords):
                edge1 = (p1, p2)
                edge2 = (q1, q2)
                angle = self.calculate_slope_and_angle(p1, p2, q1, q2)
                edge3 = self.the_closest_edges(polygon, edge2)
                x1,y1 = p1
                x2,y2 = p2
                x3,y3 = q1
                x4,y4 = q2
                len_edge1 = self.edge_length(x1,y1,x2,y2)
                len_edge2 = self.edge_length(x3,y3,x4,y4)

                if (angle == 0.0 or angle == 180) and (edge3 == edge1) and (len_edge1 <= len_edge2):
                    edges_intersect = True
                    return edges_intersect, edges_close, index, edge1, edge2
                elif (angle <= 50 or angle >= 130) and (edge3 == edge1) and (len_edge1 <= len_edge2):
                    edges_close = True
                    return edges_intersect, edges_close, index, edge1, edge2
            index = index + 1

        return None
    def calculate_slope_and_angle1(self, point1_edge1, point2_edge1, point1_edge2, point2_edge2):
        # Check if either line is horizontal (slope = 0).
        if point1_edge1[1] == point2_edge1[1]:
            slope1 = 0
            angle1 = 0
        else:
            if point2_edge1[0] == point1_edge1[0]:
                slope1 = float("inf")
                angle1 = 90
            else:
                slope1 = (point2_edge1[1] - point1_edge1[1]) / (point2_edge1[0] - point1_edge1[0])
                angle1 = math.atan(slope1)

        if point1_edge2[1] == point2_edge2[1]:
            slope2 = 0
            angle2 = 0
        else:
            if point2_edge2[0] == point1_edge2[0]:
                slope2 = float("inf")
                angle2 = 90
            else:
                slope2 = (point2_edge2[1] - point1_edge2[1]) / (point2_edge2[0] - point1_edge2[0])
                angle2 = math.atan(slope2)

        # Calculate the angle between the two lines
        angle_between_lines = abs(angle2 - angle1)
        angle_between_lines_degrees = math.degrees(angle_between_lines)

        return angle_between_lines_degrees

    def calculate_slope_and_angle(self, point1_edge1, point2_edge1, point1_edge2, point2_edge2):
        # Calculate the direction vectors of the two lines.
        vector1 = (point2_edge1[0] - point1_edge1[0], point2_edge1[1] - point1_edge1[1])
        vector2 = (point2_edge2[0] - point1_edge2[0], point2_edge2[1] - point1_edge2[1])

        # Calculate the magnitudes of the direction vectors.
        magnitude1 = math.sqrt(vector1[0] ** 2 + vector1[1] ** 2)
        magnitude2 = math.sqrt(vector2[0] ** 2 + vector2[1] ** 2)

        if magnitude1 == 0 or magnitude2 == 0:
            return 0  # Handle the case where one or both lines have no length.

        # Calculate the dot product of the direction vectors.
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # Clamp cos_theta to the valid range [-1, 1]
        cos_theta = max(-1.0, min(1.0, dot_product / (magnitude1 * magnitude2)))

        # Use the clamped cos_theta to find the angle between the two lines.
        angle_between_lines = math.acos(cos_theta)
        angle_between_lines_degrees = math.degrees(angle_between_lines)

        return angle_between_lines_degrees

    def calculate_angle(self,point, centroid):
        return (math.atan2(point[1] - centroid[1], point[0] - centroid[0]) + 2 * math.pi) % (2 * math.pi)

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

    def calculate_extent_outside(self, polygon):
        # Calculate the area of the intersection between the polygon and the convex region
        convex_region = Polygon(self.container_instance.coordinates)
        intersection_area = convex_region.intersection(polygon).area

        # Calculate the area of the polygon
        polygon_area = polygon.area

        # Calculate the extent outside by subtracting the intersection area from the polygon area
        extent_outside = polygon_area - intersection_area

        return extent_outside

    def place_polygon_closest_to_boundary2(self, polygon):
        while True:
            # Generate a random direction angle between 0 and 360 degrees
            random_direction_angle = random.uniform(0, 360)
            random_direction_vector = (
                math.cos(math.radians(random_direction_angle)), math.sin(math.radians(random_direction_angle)))

            # Start at the middle point of the convex region
            current_point = self.container_instance.calculate_centroid()
            x, y = current_point
            polygon.move_item(x, y)
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
            list.pop()
            list_of_new_region = self.for_edges_that_intersect(Polygon(self.container_instance.coordinates), Polygon(copied.coordinates))
            copied.set_coordinates(list_of_new_region)
            list.append(copied)

            draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
           # draw_instance.plot()
            list.pop()

            # Check if the polygon is inside the convex region
            pol = Polygon(pol_coords)
            convex_region = Polygon(self.container_instance.coordinates)

            if pol.within(convex_region):
                break  # Placement is successful

            # Move the polygon towards the center while maintaining the direction
            center = self.container_instance.calculate_centroid()
            direction_vector = (center[0] - farthest_point[0], center[1] - farthest_point[1])
            direction_magnitude = math.sqrt(direction_vector[0] ** 2 + direction_vector[1] ** 2)
            # Calculate the extent of the polygon outside the convex region
            #extent_outside = self.calculate_extent_outside(pol)
            # Calculate the move distance based on the extent

        # Finalize the placement
        x1, y1 = farthest_point
        polygon.move_to_point(index, x1, y1)

    def alined_edges(self,polygon,p1,p2,q1,q2,index,convex_region_pol):
        temp1 = polygon.move_from_to2_value(p1, q1)
        (e1, e2) = self.get_edge_lines(temp1)[index]
        if self.edge_cover(e1, e2, q1, q2) and Polygon(temp1).within(convex_region_pol):
            return True, p1, q1
        temp2 = polygon.move_from_to2_value(p1, q2)
        (e1, e2) = self.get_edge_lines(temp2)[index]
        if self.edge_cover(e1, e2, q1, q2) and Polygon(temp2).within(convex_region_pol):
            return True, p1, q2
        temp3 = polygon.move_from_to2_value(p2, q1)
        (e1, e2) = self.get_edge_lines(temp3)[index]
        if self.edge_cover(e1, e2, q1, q2) and Polygon(temp3).within(convex_region_pol):
            return True,p2, q1
        temp4 = polygon.move_from_to2_value(p2, q2)
        (e1, e2) = self.get_edge_lines(temp4)[index]
        if self.edge_cover(e1, e2, q1, q2) and Polygon(temp4).within(convex_region_pol):
            return True, p2, q2
        return False, (0, 0), (0, 0)

    def place_polygon_closest_to_boundary3(self):
        cx, cy = self.container_instance.calculate_centroid()
        convex_region_coords = self.container_instance.coordinates
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        thelist = []
        while len(sorted_items) != 0:
            for dex, polygon in enumerate(sorted_items):
                polygon.move_item(cx, cy)
                edges = self.find_good_edge(polygon, convex_region_coords)
                if edges is not None:
                    list1 = []
                    edges_intersect, edges_close, index, edge_of_pol, edge_of_region = edges
                    (p1, p2) = edge_of_pol
                    (q1, q2) = edge_of_region
                    pol1 = Polygon(convex_region_coords)

                    flag, e1, e2 = self.alined_edges(polygon, p1, p2, q1, q2, index, pol1)
                    temp = copy.copy(polygon)
                    temp.move_from_to2(e1, e2)
                    list1.append(temp)
                    draw_instance = Draw(self.container_instance, list1, p1, p2, q1, q2, None)
                    # draw_instance.plot()
                    list1.pop()

                    if flag:
                        print(len(sorted_items))
                        polygon.move_from_to2(e1, e2)
                        pol2 = Polygon(polygon.coordinates)
                        (t1, t2) = self.get_edge_lines(polygon.coordinates)[index]
                        point1 = None
                        point2 = None
                        point3 = None

                        if (int(t1[0]), int(t1[1])) == (int(e2[0]), int(e2[1])):
                            edge = LineString([q1, q2])
                            target_point = Point(t2)
                            closest_point = edge.interpolate(edge.project(target_point))
                            point1 = closest_point.x, closest_point.y
                            point2 = t1
                            point3 = t2
                        elif (int(t2[0]), int(t2[1])) == (int(e2[0]), int(e2[1])):
                            edge = LineString([q1, q2])
                            target_point = Point(t1)
                            closest_point = edge.interpolate(edge.project(target_point))
                            point1 = closest_point.x, closest_point.y
                            point2 = t2
                            point3 = t1
                        list1.append(polygon)
                        draw_instance = Draw(self.container_instance, list1, point1, point2, point3, (80, 80), None)
                        # draw_instance.plot()
                        list1.pop()
                        list_of_new_region = None
                        if edges_close:
                            list_of_new_region = self.for_almost_close_edges(point1, point2, point3, pol1, pol2)
                        elif edges_intersect:
                            list_of_new_region = self.for_edges_that_intersect(pol1, pol2)
                        item = copy.copy(polygon)
                        item.set_coordinates(list_of_new_region)
                        list1.append(item)

                        if len(sorted_items) <= 960:

                            draw_instance = Draw(self.container_instance, list1, point1, point2, point3, (80, 80), None)
                            draw_instance.plot()
                            list1.pop()
                            simplified_polygon = (Polygon(item.coordinates))
                            simplified_coords = simplified_polygon.exterior.coords
                            list_of = list(simplified_coords)
                            item.set_coordinates(list_of)
                            list1.append(item)
                            draw_instance = Draw(self.container_instance, list1, point1, point2, point3, (80, 80), None)
                            draw_instance.plot()


                        thelist.append(polygon)
                        convex_region_coords = list_of_new_region
                        sorted_items.remove(polygon)
        draw_instance = Draw(self.container_instance, thelist, (1,1), (1,1), (1,1), (1,1), None)
        draw_instance.plot()
        print(len(thelist))

    def plot1(self):
        cx, cy = self.container_instance.calculate_centroid()
        convex_region_coords = self.container_instance.coordinates
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        thelist = []
        for dex, polygon in enumerate(sorted_items):
            polygon.move_item(cx, cy)
            edges = self.find_good_edge(polygon, convex_region_coords)
            if edges is not None:
                list1 = []
                edges_intersect, edges_close, index, edge_of_pol, edge_of_region = edges
                (p1, p2) = edge_of_pol
                (q1, q2) = edge_of_region
                pol1 = Polygon(convex_region_coords)

                flag, e1, e2 = self.alined_edges(polygon, p1, p2, q1, q2, index, pol1)
                temp = copy.copy(polygon)
                temp.move_from_to2(e1, e2)
                list1.append(temp)
                #draw_instance = Draw(self.container_instance, list1, p1, p2, q1, q2, None)
                # draw_instance.plot()
                list1.pop()

                if flag:
                    polygon.move_from_to2(e1, e2)
                    pol2 = Polygon(polygon.coordinates)
                    (t1, t2) = self.get_edge_lines(polygon.coordinates)[index]
                    point1 = None
                    point2 = None
                    point3 = None

                    if (int(t1[0]), int(t1[1])) == (int(e2[0]), int(e2[1])):
                        edge = LineString([q1, q2])
                        target_point = Point(t2)
                        closest_point = edge.interpolate(edge.project(target_point))
                        point1 = closest_point.x, closest_point.y
                        point2 = t1
                        point3 = t2
                    elif (int(t2[0]), int(t2[1])) == (int(e2[0]), int(e2[1])):
                        edge = LineString([q1, q2])
                        target_point = Point(t1)
                        closest_point = edge.interpolate(edge.project(target_point))
                        point1 = closest_point.x, closest_point.y
                        point2 = t2
                        point3 = t1
                    list1.append(polygon)
                    #draw_instance = Draw(self.container_instance, list1, point1, point2, point3, (80, 80), None)
                    # draw_instance.plot()
                    list1.pop()
                    list_of_new_region = None
                    if edges_close:
                        list_of_new_region = self.for_almost_close_edges(point1, point2, point3, pol1, pol2)
                    elif edges_intersect:
                        list_of_new_region = self.for_edges_that_intersect(pol1, pol2)
                    item = copy.copy(polygon)
                    item.set_coordinates(list_of_new_region)
                    list1.append(item)
                    print(dex)
                    if dex >= 50:
                        draw_instance = Draw(self.container_instance, list1, point1, point2, point3, (80, 80), None)
                        draw_instance.plot()
                        list1.pop()
                        simplified_polygon = (Polygon(item.coordinates))
                        simplified_coords = simplified_polygon.exterior.coords
                        list_of = list(simplified_coords)
                        item.set_coordinates(list_of)
                        list1.append(item)
                        draw_instance = Draw(self.container_instance, list1, point1, point2, point3, (80, 80), None)
                        draw_instance.plot()

                    thelist.append(polygon)
                    convex_region_coords = list_of_new_region
                    sorted_items.remove(polygon)
        draw_instance = Draw(self.container_instance, thelist, (1, 1), (1, 1), (1, 1), (1, 1), None,None,None)
        draw_instance.plot()
        print(len(thelist))



    def plot(self):
        i = 0
        list = []
        for index, item in enumerate(self.item_instances):
            #self.place_polygon_closest_to_boundary3(item)
            list.append(item)
            i = i+1
            if i == 1:
                break
        #draw_instance = Draw(self.container_instance, list, (1, 1), (1, 1), (1, 1), (1, 1), None)
        #draw_instance.plot()
    def plot2(self):
        i = 0
        for index, item in enumerate(self.item_instances):
             self.place_polygon_closest_to_boundary2(item)
             i = i + 1
             if i == 1:
                 break



