import random
import copy
from Draw import Draw
import math
from shapely.geometry import Point, Polygon,MultiPolygon
from shapely import LineString, hausdorff_distance
from shapely.ops import unary_union

import time
import numpy as np

class Algo3:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def get_edge_lines(self,coord):
        edges = []
        num_points = len(coord)

        for i in range(num_points):
            point1 = coord[i]
            point2 = coord[(i + 1) % num_points]  # Wrap around to the first point

            line = (point1, point2)
            edges.append(line)

        return edges

    def calculate_slope(self,x1, y1, x2, y2):
        if x1 == x2:
            # The slope is float('-inf') for vertical lines
            return float('-inf')
        elif y1 == y2:
            # The slope is float('inf') for horizontal lines
            return float('inf')
        else:
            return (y2 - y1) / (x2 - x1)

    def edge_length(self,x1, y1, x2, y2):
        # Calculate the length of the edge using the Euclidean distance formula
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return length

    def line_a_cross(self,x, y, slope):
        if slope == float('inf'):
            # horizontal = y
            # x = c
            return float('-inf'), x
        elif slope == float('-inf'):
            # vertical
            # y = c
            return float('inf'), y
        else:
            s = -1 / slope
            # y = mx+n
            # y-y1 = m(x-x1)
            # y = mx-mx1+y1
            # n = -mx1+y1
            n = y - s * x
            return s, n

    def line_equation(self,x, y, slope):
        if slope == float('inf'):
            # The slope is for horizontal lines
            # y = c
            return float('inf'), y
        elif slope == float('-inf'):
            # The slope is for vertical lines
            # x = c
            return float('-inf'), x
        else:
            # y = mx+n
            # y-y1 = m(x-x1)
            # y = mx-mx1+y1
            # n = -mx1+y1
            n = y - slope * x
            return slope, n

    def do_lines_intersect(self,m1, n1, m2, n2):
        # Check if slopes are equal (parallel lines)
        if m1 == float('inf'):
            # horizontal,n1 = y
            return n2, n1
        elif m1 == float('-inf'):
            # vertical,n1 = x
            return n1, n2

        # Calculate the x-coordinate of the intersection point
        x_intersection = (n2 - n1) / (m1 - m2)

        # Calculate the y-coordinate using either line equation (both should yield the same result)
        y_intersection = m1 * x_intersection + n1
        # Check if x_intersection and y_intersection are not None
        if x_intersection is not None and y_intersection is not None:
            return x_intersection, y_intersection  # Intersection exists
        else:
            return float('inf'), float('inf')  # No intersection

    def calculate_midpoint(self,x1, y1, x2, y2):
        midpoint_x = (x1 + x2) / 2
        midpoint_y = (y1 + y2) / 2
        return midpoint_x, midpoint_y

    def is_point_between(self,x, y, x1, y1, x2, y2):
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

    def edge_cover(self,p1, p2, q1, q2):
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
            print("check", Point(midx, midy).distance(LineString([p1, p2])))
            x, y = self.point_on_edge_closest_to_point(p1, p2, (midx, midy))
            print("mid point", midx, midy)
            print("x,y", x, y)
            return self.is_point_between(x, y, x1, y1, x2, y2)

        elif ed1 < ed2:
            midx, midy = self.calculate_midpoint(x1, y1, x2, y2)
            x, y = self.point_on_edge_closest_to_point(q1, q2, (midx, midy))
            print("check", Point(midx, midy).distance(LineString([q1, q2])))

            return self.is_point_between(x, y, f1, f2, t1, t2)

        else:
            midx, midy = self.calculate_midpoint(x1, y1, x2, y2)
            x, y = self.point_on_edge_closest_to_point(q1, q2, (midx, midy))
            return self.is_point_between(x, y, f1, f2, t1, t2)

    def edges_alined(self,p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        # check if point A of edge1 is in edge2
        test1 = self.is_point_between(x1, y1, x3, y3, x4, y4)
        # check if point B of edge1 is in edge2
        test2 = self.is_point_between(x2, y2, x3, y3, x4, y4)
        # check if point A of edge2 is in edge1
        test3 = self.is_point_between(x3, y3, x1, y1, x2, y2)
        # check if point B of edge2 is in edge1
        test4 = self.is_point_between(x4, y4, x1, y1, x2, y2)
        print(test1, p1, p3, p4)
        print(test2, p2, p3, p4)
        print(test3, p3, p1, p2)
        print(test4, p4, p1, p2)

        return (test1 and test2) or (test3 and test4)

    def alined(self,p1, p2, q1, q2):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = q1
        x4, y4 = q2
        slope1 = self.calculate_slope(x1, y1, x2, y2)
        slope2 = self.calculate_slope(x3, y3, x4, y4)
        if slope1 == slope2:
            # which line is shorter:
            edge1 = self.edge_length(x1, y1, x2, y2)
            edge2 = self.edge_length(x3, y3, x4, y4)

            if edge1 > edge2:
                # calculate the smaller one in this case edge2
                mx, my = self.calculate_midpoint(x3, y3, x4, y4)
                print("middle point:", "x", mx, "y", my)
                print("between the points:", "x", x3, "y", y3, "and", "x", x4, "y", y4)
                # calculate the line equation that is perpendicular to edge1 in the middle point of it
                m, n = self.line_a_cross(mx, my, slope2)
                if m == float('inf'):
                    print("line equation of the middle point a cross:", "y=", n)
                if m == float('-inf'):
                    print("line equation of the middle point a cross:", "x=", n)

                # find line equation of edge 1

                m2, n2 = self.line_equation(x1, y2, slope1)
                # does the line cross edge 1
                a, b = self.do_lines_intersect(m, n, m2, n2)
                # print("x1:",x1,"y2:", y2,"slope:",slope1,"m:",m,"n:",n,"m2:",m2,"n2:", n2,"a:",a,"b:",b)

                return self.is_point_between(a, b, x1, y1, x2, y2)
            elif edge1 < edge2:
                # calculate the smaller one in this case edge2
                mx, my = self.calculate_midpoint(x1, y1, x2, y2)
                # calculate the line equation that is perpendicular to edge1 in the middle point of it
                m, n = self.line_a_cross(mx, my, slope1)
                # find line equation of edge 2
                m2, n2 = self.line_equation(x3, y3, slope2)
                # does the line cross edge 2
                a, b = self.do_lines_intersect(m, n, m2, n2)
                return self.is_point_between(a, b, x3, y3, x4, y4)
            else:
                # does not matter which one
                self.calculate_midpoint(x1, y1, x2, y2)
                mx, my = self.calculate_midpoint(x1, y1, x2, y2)
                # calculate the line equation that is perpendicular to edge1 in the middle point of it
                m, n = self.line_a_cross(mx, my, slope1)
                # find line equation of edge 2
                m2, n2 = self.line_equation(x3, y3, slope2)
                # does the line cross edge 2
                a, b = self.do_lines_intersect(m, n, m2, n2)
                return self.is_point_between(a, b, x3, y3, x4, y4)
        else:
            return False

    def find_point_on_edge_for_90_degree_angle(self,point_a, edge_start, edge_end):
        # Unpack the coordinates
        x_a, y_a = point_a
        x1, y1 = edge_start
        x2, y2 = edge_end

        # Calculate the direction vector from point A to edge start
        D = (x1 - x_a, y1 - y_a)

        # Calculate the direction vector of the edge
        E = (x2 - x1, y2 - y1)

        # Normalize the vectors
        magnitude_D = math.sqrt(D[0] ** 2 + D[1] ** 2)
        D_normalized = (D[0] / magnitude_D, D[1] / magnitude_D)

        magnitude_E = math.sqrt(E[0] ** 2 + E[1] ** 2)
        E_normalized = (E[0] / magnitude_E, E[1] / magnitude_E)

        # Calculate the dot product
        dot_product = D_normalized[0] * E_normalized[0] + D_normalized[1] * E_normalized[1]

        # Find the point of intersection
        intersection_x = x_a + dot_product * magnitude_D * E_normalized[0]
        intersection_y = y_a + dot_product * magnitude_D * E_normalized[1]

        return (intersection_x, intersection_y)

    # Calculate the Euclidean distance between two points
    def calculate_distance_of_points(self,x1, y1, x2, y2):
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        return distance

    def clean_coords(self,mergedPolys):
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

    def calculate_slope_and_angle(self,point1_edge1, point2_edge1, point1_edge2, point2_edge2):
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

    def calculate_angle(self,point, centroid):
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

    def remove_duplicate_coordinates(self,coordinates):
        unique_coordinates = []
        for point in coordinates:
            if point not in unique_coordinates:
                unique_coordinates.append(point)
        return unique_coordinates

    def filter_middle_points(self,points):
        if len(points) < 3:
            return points

        result = [points[0]]
        for i in range(1, len(points) - 1):
            previous_point = points[i - 1]
            current_point = points[i]
            next_point = points[i + 1]

            if (previous_point[0] + next_point[0]) / 2 == current_point[0] and (
                    previous_point[1] + next_point[1]) / 2 == \
                    current_point[1]:
                continue  # Skip the point if it's in the middle
            else:
                result.append(current_point)

        result.append(points[-1])
        return result

    def remove_point_from_list(self,points, point_to_remove):
        if point_to_remove in points:
            points.remove(point_to_remove)

    def insert_list_at_index(self,list1, list2, index):
        """
        Inserts list2 into list1 at the specified index.

        :param list1: The original list of points.
        :param list2: The list of points to be inserted.
        :param index: The index at which to insert list2.
        :return: The resulting list with list2 inserted at the specified index.
        """
        return list1[:index] + list2 + list1[index:]

    def check_points(self,p1, p2, p3, p4):
        point1 = Point(p1)
        point2 = Point(p2)
        point3 = Point(p3)
        point4 = Point(p4)
        distance1 = point1.distance(point3)
        distance2 = point1.distance(point4)
        distance3 = point2.distance(point3)
        distance4 = point2.distance(point4)
        thepo = None
        thesec = None
        ch = None
        if distance1 == 0:
            thepo = p3
            thesec = p4
            ch = point2.distance(point4)
        if distance3 == 0:
            thepo = p3
            thesec = p4
            ch = point1.distance(point4)
        if distance2 == 0:
            thepo = p4
            thesec = p3
            ch = point2.distance(point3)
        if distance4 == 0:
            thepo = p4
            thesec = p3
            ch = point1.distance(point3)

        return thesec

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

    def is_point_on_edge(self,point, endpoint1, endpoint2):
        x, y = point
        x1, y1 = endpoint1
        x2, y2 = endpoint2

        vector1 = (x - x1, y - y1)
        vector2 = (x2 - x1, y2 - y1)

        # Calculate the dot product of vector1 and vector2
        dot_product = vector1[0] * vector2[0] + vector1[1] * vector2[1]

        # Calculate the magnitudes of vector1 and vector2
        magnitude1 = (vector1[0] ** 2 + vector1[1] ** 2) ** 0.5
        magnitude2 = (vector2[0] ** 2 + vector2[1] ** 2) ** 0.5

        # Calculate the cosine of the angle between vector1 and vector2
        cosine = dot_product / (magnitude1 * magnitude2)

        # Check if the vectors are collinear (cosine is 1 or -1)
        return abs(cosine) == 1

    def distance_to_center(self,edge, center):
        cx, cy = center
        (p1, p2) = edge
        x1, y1 = p1
        x2, y2 = p2
        # Calculate the distance from the center to both endpoints and return the minimum
        dist1 = ((x1 - cx) ** 2 + (y1 - cy) ** 2) ** 0.5
        dist2 = ((x2 - cx) ** 2 + (y2 - cy) ** 2) ** 0.5
        return min(dist1, dist2)

    def are_edges_close(self,p1, p2, q1, q2, distance):
        x1, y1 = p1
        x2, y2 = p2
        xt1, ty1 = q1
        xt2, yt2 = q2

        edge1_size = self.edge_length(x1, y1, x2, y2)
        edge2_size = self.edge_length(xt1, ty1, xt2, yt2)

        threshold_ratio = 0.5  # 50% threshold

        # Determine the size of the larger edge
        larger_edge_size = max(edge1_size, edge2_size)

        # Calculate the threshold
        threshold = larger_edge_size * threshold_ratio
        print("threshold", threshold)
        # Compare distance to threshold
        if distance < threshold:
            return True  # Edges are close
        else:
            return False  # Edges are not close

    def requirements(self,p1, p2, t1, t2, epsilon):
        po1 = None
        po2 = None
        po3 = None
        po4 = None
        l1 = LineString([p1, p2])
        l2 = LineString([t1, t2])
        dist = hausdorff_distance(l1, l2)
        print("epsilon", epsilon, "dist", dist)
        angle = self.calculate_slope_and_angle(p1, p2, t1, t2)
        print("angle", angle)
        print("points", p1, p2, t1, t2)
        the_angle = None
        the_dist = None
        # draw_instance.plot()
        if dist < epsilon and (angle < 0.01 or angle > 179) and self.edge_cover(p1, p2, t1, t2) and self.are_edges_close(p1, p2,
                                                                                                               t1,
                                                                                                               t2,
                                                                                                               dist):
            print("is point between", self.edge_cover(p1, p2, t1, t2))
            print("edges are close", self.are_edges_close(p1, p2, t1, t2, dist))
            yes2 = True
            po1 = p1
            po2 = p2
            po3 = t1
            po4 = t2
            the_angle = angle
            the_dist = dist
            return True

        return False

    def custom_polygon_sort(self,polygon):
        value = polygon.value
        area = polygon.reaches_to_rec()
        num_coordinates = len(polygon.coordinates)

        # The expression for sorting: value / (area + number of coordinates)
        return value / (area + num_coordinates)

    def distance(self, point1, point2):
        # Calculate the Euclidean distance between two points
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

    def find_farthest_point(self,edge, center):
        # Find the farthest point in the edge from the center
        point1, point2 = edge
        d1 = self.distance(center, point1)
        d2 = self.distance(center, point2)

        if d1 > d2:
            return point1
        else:
            return point2

    def sort_edges_by_farthest_point(self,edges):
        convex_region = self.container_instance.coordinates
        if len(convex_region) < 3 or len(edges) == 0:
            return []

        center = self.container_instance.calculate_centroid()

        # Sort the edges based on the distance of the farthest point
        sorted_edges = sorted(edges, key=lambda edge: self.distance(center, self.find_farthest_point(edge, center)), reverse=True)

        return sorted_edges



    def plot(self):
        i = 0
        pol1 = None
        list_of_items = []
        temp = None
        list_check = []
        mainitem = None
        thelist = self.item_instances
        sorted_polygons = sorted(self.item_instances, key=self.custom_polygon_sort, reverse=True)
        sorted_items = sorted(sorted_polygons, key=lambda item: item.calculate_total_dimensions(), reverse=True)

        for index, item in enumerate(sorted_items):
            if i == 0:
                pol1 = Polygon(self.container_instance.coordinates)
                mainitem = self.container_instance.coordinates
                temp = item
                list_check.append(item)

            if i > 0:
                flag = False
                z = 0
                sorted1 = self.sort_edges_by_farthest_point(self.get_edge_lines(mainitem))
                for (p1, p2) in sorted1:
                    x1, y1 = p1
                    x2, y2 = p2
                    j = 0
                    for (q1, q2) in self.get_edge_lines(item.coordinates):
                        num_q1 = j
                        cord_of_current_item = item.move_to_point_value(j, x1, y1)

                        # shallow_copy = copy.copy(item)
                        # temp.set_coordinates(mainitem)

                        # list_check[0] = temp
                        # shallow_copy.move_to_point(j, x1, y1)

                        # list_check.append(shallow_copy)

                        # draw_instance = Draw(self.container_instance, list_check, (1, 1), (1, 1), (1, 1), (1, 1),None)
                        # draw_instance.plot()

                        # list_check.pop()

                        pol2 = Polygon(cord_of_current_item)
                        if pol2.within(pol1):
                            print("in", i)
                            (xey, yey) = self.get_edge_lines(self.container_instance.coordinates)[
                                (z - 1) % len(self.get_edge_lines(self.container_instance.coordinates))]
                            (t1, t2) = self.get_edge_lines(cord_of_current_item)[j]
                            (dex, dey) = self.get_edge_lines(cord_of_current_item)[
                                (j - 1) % len(cord_of_current_item)]

                            yes = False
                            po1 = None
                            po2 = None
                            po3 = None
                            po4 = None
                            flag3 = False
                            l1 = LineString([p1, p2])
                            l2 = LineString([t1, t2])
                            if l1.intersects(l2):
                                inters = l1.intersection(l2)
                                if not inters.is_empty:
                                    # Calculate the length of the intersection
                                    intersection_len = inters.length
                                    if intersection_len > 0:
                                        flag3 = True
                            epsilon = self.container_instance.calculate_distance_threshold()
                            dist = hausdorff_distance(l1, l2)
                            print("epsilon", epsilon, "dist", dist)
                            print("do they touch?", l1.distance(l2))
                            angle = self.calculate_slope_and_angle(p1, p2, t1, t2)
                            print("angle", angle)
                            print("points", p1, p2, t1, t2)
                            # draw_instance = Draw(self.container_instance, list_check, p1, p2, t1, t2, None)
                            # draw_instance.plot()
                            print("edges are close", self.are_edges_close(p1, p2, t1, t2, dist))
                            if flag3:
                                print("is point between", self.edge_cover(p1, p2, t1, t2))
                                print("edges are close", self.are_edges_close(p1, p2, t1, t2, dist))
                                yes = True
                                po1 = p1
                                po2 = p2
                                po3 = t1
                                po4 = t2
                            angle = self.calculate_slope_and_angle(p1, p2, dex, dey)
                            flag3 = False
                            l1 = LineString([p1, p2])
                            l2 = LineString([dex, dey])
                            if l1.intersects(l2):
                                inters = l1.intersection(l2)
                                if not inters.is_empty:
                                    # Calculate the length of the intersection
                                    intersection_len = inters.length
                                    if intersection_len > 0:
                                        flag3 = True
                            print("do they touch?", l1.distance(l2))

                            dist = hausdorff_distance(l1, l2)
                            print("epsilon", epsilon, "dist", dist)
                            print("angle", angle)
                            print("points", p1, p2, dex, dey)
                            # draw_instance = Draw(self.container_instance, list_check, p1, p2, dex, dey, None)
                            # draw_instance.plot()
                            print("edges are close", self.are_edges_close(p1, p2, dex, dey, dist))
                            if flag3:
                                print("is point between", self.edge_cover(p1, p2, dex, dey))
                                print("edges are close", self.are_edges_close(p1, p2, dex, dey, dist))
                                yes = True
                                po1 = p1
                                po2 = p2
                                po3 = dex
                                po4 = dey
                            flag3 = False
                            l1 = LineString([xey, yey])
                            l2 = LineString([t1, t2])
                            if l1.intersects(l2):
                                inters = l1.intersection(l2)
                                if not inters.is_empty:
                                    # Calculate the length of the intersection
                                    intersection_len = inters.length
                                    if intersection_len > 0:
                                        flag3 = True
                            print("do they touch?", l1.distance(l2))

                            dist = hausdorff_distance(l1, l2)
                            print("epsilon", epsilon, "dist", dist)
                            angle = self.calculate_slope_and_angle(xey, yey, t1, t2)
                            print("angle", angle)
                            print("points", xey, yey, t1, t2)
                            # draw_instance = Draw(self.container_instance, list_check, xey, yey, t1, t2, None)
                            # draw_instance.plot()
                            print("edges are close", self.are_edges_close(xey, yey, t1, t2, dist))
                            if flag3:
                                print("is point between", self.edge_cover(xey, yey, t1, t2))
                                print("edges are close", self.are_edges_close(xey, yey, t1, t2, dist))
                                yes = True
                                po1 = xey
                                po2 = yey
                                po3 = t1
                                po4 = t2
                            flag3 = False
                            l1 = LineString([xey, yey])
                            l2 = LineString([dex, dey])
                            if l1.intersects(l2):
                                inters = l1.intersection(l2)
                                if not inters.is_empty:
                                    # Calculate the length of the intersection
                                    intersection_len = inters.length
                                    if intersection_len > 0:
                                        flag3 = True
                            print("do they touch?", l1.distance(l2))

                            dist = hausdorff_distance(l1, l2)
                            print("epsilon", epsilon, "dist", dist)
                            angle = self.calculate_slope_and_angle(xey, yey, dex, dey)
                            print("angle", angle)
                            print("points", xey, yey, dex, dey)
                            print("edges are close", self.are_edges_close(xey, yey, dex, dey, dist))
                            if flag3:
                                print("is point between", self.edge_cover(xey, yey, dex, dey))
                                print("edges are close", self.are_edges_close(xey, yey, dex, dey, dist))
                                yes = True
                                po1 = xey
                                po2 = yey
                                po3 = dex
                                po4 = dey
                            yes2 = False
                            if not yes:
                                po1 = None
                                po2 = None
                                po3 = None
                                po4 = None
                                l1 = LineString([p1, p2])
                                l2 = LineString([t1, t2])
                                epsilon = self.container_instance.calculate_distance_threshold()
                                dist = hausdorff_distance(l1, l2)
                                print("epsilon", epsilon, "dist", dist)
                                angle = self.calculate_slope_and_angle(p1, p2, t1, t2)
                                print("angle", angle)
                                print("points", p1, p2, t1, t2)
                                which_index = None
                                the_angle = None
                                the_dist = None
                                # draw_instance = Draw(self.container_instance, list_check, p1, p2, t1, t2, None)
                                # draw_instance.plot()
                                print("edges are close", self.are_edges_close(p1, p2, t1, t2, dist))
                                if dist < epsilon and (angle < 10 or angle > 170) and self.edge_cover(p1, p2, t1,
                                                                                                      t2) and self.are_edges_close(
                                    p1,
                                    p2,
                                    t1,
                                    t2,
                                    dist):
                                    print("is point between", self.edge_cover(p1, p2, t1, t2))
                                    print("edges are close", self.are_edges_close(p1, p2, t1, t2, dist))
                                    yes2 = True
                                    po1 = p1
                                    po2 = p2
                                    po3 = t1
                                    po4 = t2
                                    the_angle = angle
                                    the_dist = dist
                                    which_index = j
                                angle = self.calculate_slope_and_angle(p1, p2, dex, dey)
                                l1 = LineString([p1, p2])
                                l2 = LineString([dex, dey])
                                dist = hausdorff_distance(l1, l2)
                                print("epsilon", epsilon, "dist", dist)
                                print("angle", angle)
                                print("points", p1, p2, dex, dey)
                                # draw_instance = Draw(self.container_instance, list_check, p1, p2, dex, dey, None)
                                # draw_instance.plot()
                                print("edges are close", self.are_edges_close(p1, p2, dex, dey, dist))
                                if dist < epsilon and (angle < 10 or angle > 170) and self.edge_cover(p1, p2, dex,
                                                                                                      dey) and self.are_edges_close(
                                    p1,
                                    p2,
                                    dex,
                                    dey,
                                    dist):
                                    print("is point between", self.edge_cover(p1, p2, dex, dey))
                                    print("edges are close", self.are_edges_close(p1, p2, dex, dey, dist))
                                    yes2 = True
                                    po1 = p1
                                    po2 = p2
                                    po3 = dex
                                    po4 = dey
                                    the_angle = angle
                                    the_dist = dist
                                    which_index = j
                                l1 = LineString([xey, yey])
                                l2 = LineString([t1, t2])
                                dist = hausdorff_distance(l1, l2)
                                print("epsilon", epsilon, "dist", dist)
                                angle = self.calculate_slope_and_angle(xey, yey, t1, t2)
                                print("angle", angle)
                                print("points", xey, yey, t1, t2)
                                # draw_instance = Draw(self.container_instance, list_check, xey, yey, t1, t2, None)
                                # draw_instance.plot()
                                print("edges are close", self.are_edges_close(xey, yey, t1, t2, dist))
                                if dist < epsilon and (angle < 10 or angle > 170) and self.edge_cover(xey, yey, t1,
                                                                                                      t2) and self.are_edges_close(
                                    xey,
                                    yey,
                                    t1,
                                    t2,
                                    dist):
                                    print("is point between", self.edge_cover(xey, yey, t1, t2))
                                    print("edges are close", self.are_edges_close(xey, yey, t1, t2, dist))
                                    yes2 = True
                                    po1 = xey
                                    po2 = yey
                                    po3 = t1
                                    po4 = t2
                                    the_angle = angle
                                    the_dist = dist
                                    which_index = (j - 1) % len(cord_of_current_item)
                                l1 = LineString([xey, yey])
                                l2 = LineString([dex, dey])
                                dist = hausdorff_distance(l1, l2)
                                print("epsilon", epsilon, "dist", dist)
                                angle = self.calculate_slope_and_angle(xey, yey, dex, dey)
                                print("angle", angle)
                                print("points", xey, yey, dex, dey)
                                print("edges are close", self.are_edges_close(xey, yey, dex, dey, dist))
                                if dist < epsilon and (angle < 10 or angle > 170) and self.edge_cover(xey, yey, dex,
                                                                                                      dey) and self.are_edges_close(
                                    xey,
                                    yey,
                                    dex,
                                    dey,
                                    dist):
                                    print("is point between", self.edge_cover(xey, yey, dex, dey))
                                    print("edges are close", self.are_edges_close(xey, yey, dex, dey, dist))
                                    yes2 = True
                                    po1 = xey
                                    po2 = yey
                                    po3 = dex
                                    po4 = dey
                                    the_angle = angle
                                    the_dist = dist
                                    which_index = (j - 1) % len(cord_of_current_item)
                            print("-----------------------------------------")
                            if not pol1.crosses(pol2):
                                print("+++++++++++++++++++++++++++++++++++++++")
                                if yes:
                                    thelist.remove(item)
                                    item.move_to_point(j, x1, y1)
                                    pol2 = Polygon(item.coordinates)
                                    list_of_items.append(item)
                                    mergedPolys = pol1.difference(pol2)
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

                                    # exterior_coords_list = self.remove_duplicate_coordinates(exterior_coords_list)

                                    pol1 = Polygon(exterior_coords_list)
                                    mainitem = exterior_coords_list
                                    temp.set_coordinates(mainitem)
                                    list_check[0] = temp
                                    draw_instance = Draw(self.container_instance, list_of_items, po1, po2,
                                                         po3, po4, None)
                                    # draw_instance.plot()

                                    draw_instance = Draw(self.container_instance, list_check, po1, po2,
                                                         po3, po4, None)
                                    # draw_instance.plot()

                                    # print("inside:", "angle", the_angle, "dist:", the_dist)
                                    flag = True
                                    break
                                if yes2:
                                    thelist.remove(item)
                                    item.move_to_point(j, x1, y1)
                                    # list_check.append(shallow_copy)
                                    pol2 = Polygon(item.coordinates)
                                    list_of_items.append(item)
                                    counter = self.order_coordinates_counterclockwise([po1, po2, po3, po4])
                                    pol4 = Polygon(counter)

                                    mergedPolys1 = pol1.difference(pol4)

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
                                    # exterior_coords_list1 = self.remove_duplicate_coordinates(exterior_coords_list1)

                                    mergedPolys = Polygon(exterior_coords_list1).difference(pol2)

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

                                    # exterior_coords_list = self.remove_duplicate_coordinates(exterior_coords_list)

                                    pol1 = Polygon(exterior_coords_list)
                                    mainitem = exterior_coords_list
                                    temp.set_coordinates(mainitem)
                                    list_check[0] = temp
                                    draw_instance = Draw(self.container_instance, list_of_items, po1, po2,
                                                         po3, po4, None)
                                    #draw_instance.plot()
                                    draw_instance = Draw(self.container_instance, list_check, po1, po2, po3, po4,
                                                         None)
                                    # print("inside:", "angle", the_angle, "dist:", the_dist)
                                    #draw_instance.plot()
                                    # list_check.pop()

                                    flag = True
                                    break

                        j = j + 1
                    if flag:
                        break
                    z = z + 1
            if i == 50:
                break
            i = i + 1





        draw_instance = Draw(self.container_instance, list_of_items, (1, 1), (1, 1), (1, 1), (1, 1), None)
        draw_instance.plot()



