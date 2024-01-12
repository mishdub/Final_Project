from Draw import Draw
import math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint

from shapely.ops import unary_union
import time
import warnings
import copy
import sympy as sp
import numpy as np

from decimal import Decimal, getcontext


class Algo24:

    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances
        warnings.showwarning = self.warning_handler
        self.error_occurred = False  # Initialize error_occurred as False
        # Define a warning handler to print warnings

    def calculate_angle(self, point, centroid):
        return (math.atan2(point[1] - centroid[1], point[0] - centroid[0]) + 2 * math.pi) % (2 * math.pi)

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

    def order_coordinates_counterclockwise(self, coordinates):
        """
        if self.is_counterclockwise(coordinates):
            print("orderd")
            return coordinates
        """
        # Calculate the centroid
        centroid_x = sum(x[0] for x in coordinates) / len(coordinates)
        centroid_y = sum(x[1] for x in coordinates) / len(coordinates)
        centroid = (centroid_x, centroid_y)

        # Sort the coordinates based on angles
        sorted_coordinates = sorted(coordinates, key=lambda point: self.calculate_angle(point, centroid))

        return sorted_coordinates

    def move_poly(self, polygon, angle, convex_region):
        # go over the front points of the polygon and crete lines from them in the direction and
        # check for intresection point in the all convex region
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

    def move_poly_LineString(self, polygon, angle, line_st):
        # go over the front points of the polygon and crete lines from them in the direction and
        # check for intresection point in a specific region (line string)
        list_of_lines = []
        list_of_lines.append(line_st)
        line_st = LineString(line_st)
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        min_distance = float('inf')
        from_p = None
        to_p = None
        list_of_points = []
        for point in polygon.coordinates:
            x, y = point
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line = LineString([(x, y), (x1, y1)])
            if not line.crosses(Polygon(polygon.coordinates)):
                list_of_lines.append([(x, y), (x1, y1)])
                in_po = self.find_intersection_point_linestring(line_st, [(x, y), (x1, y1)], (x, y))
                if in_po is not None:
                    list_of_points.append(in_po)
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
        return from_p, to_p, list_of_points, list_of_lines, min_distance

    def move_poly_LineString2(self, polygon, angle, line_st):
        # go over all the points of the line string and create lines from them and check intresections in the polygon
        list_of_lines = []
        angle = (angle + 180) % 360
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        min_distance = float('inf')
        from_p = None
        to_p = None
        list_of_points = []
        for point_line in line_st.coords:
            xl, yl = point_line
            x1, y1 = self.calculate_endpoint_from_direction(xl, yl, vx, vy, dime)
            p = self.find_intersection_point2(polygon.coordinates, [(xl, yl), (x1, y1)], (xl, yl))
            if p is not None:
                inx, iny = p
                p1 = Point(xl, yl)
                p2 = Point(inx, iny)
                list_of_points.append((xl, yl))
                list_of_points.append((inx, iny))
                distance = p1.distance(p2)
                if distance < min_distance:
                    min_distance = distance
                    from_p = (inx, iny)
                    to_p = (xl, yl)
        return from_p, to_p, list_of_points, list_of_lines, min_distance

    def move_poly_MultiLineString(self, polygon, angle, MultiLineString):
        list_of_lines = []
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        min_distance = float('inf')
        from_p = None
        to_p = None
        list_of_points = []
        for point in polygon.coordinates:
            x, y = point
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line = LineString([(x, y), (x1, y1)])
            if not line.crosses(Polygon(polygon.coordinates)):
                list_of_lines.append([(x, y), (x1, y1)])
                for line_st in MultiLineString:
                    in_po = self.find_intersection_point_linestring(line_st, [(x, y), (x1, y1)], (x, y))
                    if in_po is not None:
                        list_of_points.append(in_po)
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

        return from_p, to_p, list_of_points, list_of_lines, min_distance

    def move_poly_MultiLineString2(self, polygon, angle, MultiLineString):
        list_of_lines = []
        angle = (angle + 180) % 360
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        min_distance = float('inf')
        from_p = None
        to_p = None
        list_of_points = []
        for line_st in MultiLineString:
            # list_of_lines.append(line_st.coords)
            for point_line in line_st.coords:
                xl, yl = point_line
                x1, y1 = self.calculate_endpoint_from_direction(xl, yl, vx, vy, dime)
                p = self.find_intersection_point2(polygon.coordinates, [(xl, yl), (x1, y1)], (xl, yl))
                if p is not None:
                    inx, iny = p
                    p1 = Point(xl, yl)
                    p2 = Point(inx, iny)
                    list_of_points.append((xl, yl))
                    # list_of_points.append((inx, iny))
                    distance = p1.distance(p2)
                    if distance < min_distance:
                        min_distance = distance
                        from_p = (inx, iny)
                        to_p = (xl, yl)

        return from_p, to_p, list_of_points, list_of_lines, min_distance

    def place_poly(self, original_polygon, extended_poly, convex_region, angle, right_line, left_line):
        convex_exterior = Polygon(convex_region)
        convex_exterior = convex_exterior.exterior

        f_p = None
        t_p = None
        dis1 = float('inf')
        dis2 = float('inf')
        dis3 = float('inf')
        dis4 = float('inf')
        dis5 = float('inf')
        dis6 = float('inf')
        min_dis_in_multi = float('inf')
        min_dis_in_line = float('inf')
        min_dis_in_multi_and_line = float('inf')

        list_of_lines = []
        list_of_points = []
        list_of_l = []
        if extended_poly.intersects(convex_exterior):
            intersection = extended_poly.intersection(convex_exterior)
            print(intersection.geom_type)

            if intersection.is_empty:
                print("Polygons overlap, but no intersection.")
            else:

                if intersection.geom_type == "LineString":

                    f_p1, t_p1, list_of_points1, list_of_l1, dis1 = self.move_poly_LineString(original_polygon,
                                                                                              angle,
                                                                                              list(intersection.coords))
                    f_p2, t_p2, list_of_points2, list_of_l2, dis2 = self.move_poly_LineString2(original_polygon,
                                                                                               angle,
                                                                                               intersection)
                    if dis1 < dis2:
                        f_p = f_p1
                        t_p = t_p1
                        list_of_l = list_of_l1
                        list_of_points = list_of_points1
                        min_dis_in_line = dis1

                    else:
                        f_p = f_p2
                        t_p = t_p2
                        list_of_l = list_of_l2
                        list_of_points = list_of_points2
                        min_dis_in_line = dis2




                elif intersection.geom_type == "MultiLineString":
                    for line in intersection.geoms:
                        list_of_lines.append(list(line.coords))

                    f_p1, t_p1, list_of_points1, list_of_l1, dis3 = self.move_poly_MultiLineString2(original_polygon,
                                                                                                    angle,
                                                                                                    intersection.geoms)

                    f_p2, t_p2, list_of_points2, list_of_l2, dis4 = self.move_poly_MultiLineString(original_polygon,
                                                                                                   angle,
                                                                                                   intersection.geoms)

                    if dis3 < dis4:
                        f_p = f_p1
                        t_p = t_p1
                        list_of_l = list_of_l1
                        list_of_points = list_of_points1
                        min_dis_in_multi = dis3

                    else:
                        f_p = f_p2
                        t_p = t_p2
                        list_of_l = list_of_l2
                        list_of_points = list_of_points2
                        min_dis_in_multi = dis4

        else:
            print("not intresectino?")

        if min_dis_in_multi < min_dis_in_line:
            min_dis_in_multi_and_line = min_dis_in_multi
        else:
            min_dis_in_multi_and_line = min_dis_in_line

        spoint1 = Point((list(right_line.coords))[0])
        spoint2 = Point((list(left_line.coords))[0])
        right_line = list(right_line.coords)
        pre_point1 = right_line[0]

        left_line = list(left_line.coords)
        pre_point2 = left_line[0]
        point1 = self.find_intersection_point(convex_exterior.coords, right_line, pre_point1)
        point2 = self.find_intersection_point(convex_exterior.coords, left_line, pre_point2)
        point1 = Point(point1)
        point2 = Point(point2)

        dis5 = spoint1.distance(point1)
        dis6 = spoint2.distance(point2)
        if dis5 < dis6:
            if dis5 < min_dis_in_multi_and_line:
                f_p = (spoint1.x, spoint1.y)
                t_p = (point1.x, point1.y)

        else:
            if dis6 < min_dis_in_multi_and_line:
                f_p = (spoint2.x, spoint2.y)
                t_p = (point2.x, point2.y)

        return f_p, t_p, list_of_lines + list_of_l, list_of_points

    def classify_points_left_right1(self, line_angle, line_start, points):
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

    def classify_points_left_right3(self, line_angle, points):
        # Calculate the slope of the separating line
        line_slope = math.tan(math.radians(line_angle))

        left_side_points = []
        right_side_points = []

        for point in points:
            # Calculate the expected y-coordinate on the separating line
            expected_y = line_slope * point[0]

            # Determine if the point is above or below the separating line
            if point[1] < expected_y:
                left_side_points.append(point)
            else:
                right_side_points.append(point)

        return left_side_points, right_side_points

    def classify_points_left_right2(self, line_angle, line_start, points):
        left_side_points = []
        right_side_points = []

        leftmost_point = None
        rightmost_point = None

        for point in points:
            # Calculate the angle between the line and the vector from line_start to the point.
            angle_rad = math.atan2(point[1] - line_start[1], point[0] - line_start[0])
            angle_deg = math.degrees(angle_rad)

            # Determine if the point is on the left or right side.
            angle_difference = (angle_deg - line_angle + 360) % 360  # Ensure angle_difference is positive
            if angle_difference < 180:
                left_side_points.append(point)
                # Update leftmost_point if the current point is more left.
                if leftmost_point is None or point[0] < leftmost_point[0]:
                    leftmost_point = point
            else:
                right_side_points.append(point)
                # Update rightmost_point if the current point is more right.
                if rightmost_point is None or point[0] > rightmost_point[0]:
                    rightmost_point = point

        return left_side_points, right_side_points, leftmost_point, rightmost_point

    def find_farthest_point_from_line(self, line_coords, points, polygon, vx, vy, dime):
        # Create a LineString object from the line coordinates.
        line = LineString(line_coords)

        max_distance = -1
        farthest_point = None

        pol = Polygon(polygon)

        i = 0
        for point_coords in points:
            # Create a Point object from the point coordinates.
            point = Point(point_coords)

            # Calculate the distance from the point to the line.
            distance = point.distance(line)
            x, y = point_coords
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line2 = LineString([(x, y), (x1, y1)])
            if distance > max_distance:
                if not line2.crosses(pol):
                    max_distance = distance
                    farthest_point = point_coords
        return farthest_point

    def find_farthest_point_from_line_temp(self, line_coords, points):
        # Create a LineString object from the line coordinates.
        line = LineString(line_coords)

        max_distance = -1
        farthest_point = None
        for point_coords in points:
            # Create a Point object from the point coordinates.
            point = Point(point_coords)
            # Calculate the distance from the point to the line.
            distance = point.distance(line)
            if distance > max_distance:
                max_distance = distance
                farthest_point = point_coords

        return farthest_point

    def find_farthest_point_from_line_special(self, line_coords, points, polygon, vx, vy, dime, center):
        # Create a LineString object from the line coordinates.
        line = LineString(line_coords)

        max_distance = -1
        farthest_point = None

        pol = Polygon(polygon)
        c_point = Point(center)

        i = 0
        for point_coords in points:
            # Create a Point object from the point coordinates.
            point = Point(point_coords)

            # Calculate the distance from the point to the line.
            distance = point.distance(line)
            x, y = point_coords
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line2 = LineString([(x, y), (x1, y1)])
            if distance > max_distance:
                if not line2.crosses(pol):
                    max_distance = distance
                    farthest_point = point_coords

        return farthest_point

    def find_farthest_point_from_line_special2(self, line_coords, points, polygon, vx, vy, dime, center):
        # Create a LineString object from the line coordinates.
        line = LineString(line_coords)

        max_distance = -1
        min_dis = float('inf')
        farthest_point = None
        sec_farthest_point = None

        pol = Polygon(polygon)
        c_point = Point(center)

        i = 0
        for point_coords in points:
            # Create a Point object from the point coordinates.
            point = Point(point_coords)

            # Calculate the distance from the point to the line.
            distance = point.distance(line)
            x, y = point_coords
            x1, y1 = self.calculate_endpoint_from_direction(x, y, vx, vy, dime)
            line2 = LineString([(x, y), (x1, y1)])
            dis = point.distance(c_point)
            if distance > max_distance and dis < min_dis:
                if not line2.crosses(pol):
                    max_distance = distance
                    min_dis = dis
                    farthest_point = point_coords
        return farthest_point

    def calculate_line_angle(self, line):
        # Extract the coordinates of the start and end points of the line
        start_x, start_y = line.coords[0]
        end_x, end_y = line.coords[-1]

        # Calculate the angle in radians
        angle_radians = math.atan2(end_y - start_y, end_x - start_x)

        # Convert the angle from radians to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees

    def placement(self, angle, middle_polygon, convex_polygon):
        dime = self.container_instance.calculate_total_dimensions()
        center = self.calculate_centroid(middle_polygon)
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
        pol = Polygon(convex_polygon.coordinates)
        pol = pol.buffer(0.1)

        if not (filled_polygon.intersects(pol)):
            flag = True
        return flag, (px1, py1), p1, (px2, py2), p2, (cx, cy), (x1, y1), filled_polygon, right_line, left_line

    def placement_rec(self, angle, middle_polygon, convex_polygon):
        dime = self.container_instance.calculate_total_dimensions()
        center = self.calculate_centroid(middle_polygon)
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
        pol = Polygon(convex_polygon.coordinates)
        pol = pol.buffer(0.1)
        pol = self.polygon_to_rectangle(pol.exterior.coords)
        pol = Polygon(pol)

        if not (filled_polygon.intersects(pol)):
            flag = True
        return flag, (px1, py1), p1, (px2, py2), p2, (cx, cy), (x1, y1), filled_polygon, right_line, left_line

    def extend_pol(self, angle, convex_region, polygon):
        dime = self.container_instance.calculate_total_dimensions()
        center = self.calculate_centroid(polygon.coordinates)
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, polygon.coordinates)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x1, y1)]

        px1, py1 = self.find_farthest_point_from_line(line1, right, polygon.coordinates, vx, vy, dime)
        px2, py2 = self.find_farthest_point_from_line(line1, left, polygon.coordinates, vx, vy, dime)

        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        int_point1 = self.find_intersection_point(convex_region, [(px1, py1), p1],
                                                  (px1, py1))

        if int_point1 is None:
            int_point1 = px1, py1
            px, py = int_point1
        else:
            px, py = int_point1

        int_point2 = self.find_intersection_point(convex_region, [(px2, py2), p2],
                                                  (px2, py2))

        if int_point2 is None:
            int_point2 = px2, py2
            qx, qy = int_point2
        else:
            qx, qy = int_point2

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

    def check_ep(self, angle, p, convex_center):
        new_pol = Polygon(p.coordinates)
        new_pol = new_pol.buffer(0.1)
        copied = copy.deepcopy(p)
        list_of = list(new_pol.exterior.coords)
        copied.set_coordinates(list_of)
        dime = self.container_instance.calculate_total_dimensions()
        center = self.calculate_centroid(copied.coordinates)
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, copied.coordinates)

        ops_angle = (angle + 180) % 360

        vx2, vy2 = (
            math.cos(math.radians(ops_angle)), math.sin(math.radians(ops_angle)))

        x4, y4 = self.calculate_endpoint_from_direction(cx, cy, vx2, vy2, dime)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x4, y4)]
        po1 = self.find_farthest_point_from_line_special(line1, right, copied.coordinates, vx2, vy2, dime,
                                                         convex_center)
        qo1 = self.find_farthest_point_from_line_special2(line1, right, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        po2 = self.find_farthest_point_from_line_special(line1, left, copied.coordinates, vx2, vy2, dime, convex_center)
        qo2 = self.find_farthest_point_from_line_special2(line1, left, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        (px1, py1) = po1
        (qx1, qy1) = qo1
        (px2, py2) = po2
        (qx2, qy2) = qo2
        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        the_point = None
        right_angle = self.calculate_angle_in_degrees((px1, py1), p1)
        left_angle = self.calculate_angle_in_degrees((px2, py2), p2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point = (px1, py1)
        else:
            the_point = (px2, py2)

        q1 = self.calculate_endpoint_from_direction(qx1, qy1, vx, vy, dime)
        q2 = self.calculate_endpoint_from_direction(qx2, qy2, vx, vy, dime)

        the_point = None
        right_angle = self.calculate_angle_in_degrees((qx1, qy1), q1)
        left_angle = self.calculate_angle_in_degrees((qx2, qy2), q2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point = (qx1, qy1)
        else:
            the_point = (qx2, qy2)

        return (px2, py2), (qx1, qy1), left

    def check_ep_rec(self, angle, p, convex_center):
        new_pol = Polygon(p.coordinates)
        new_pol = new_pol.buffer(0.1)
        copied = copy.deepcopy(p)
        list_of = list(new_pol.exterior.coords)
        list_of = self.polygon_to_rectangle(list_of)
        copied.set_coordinates(list_of)
        dime = self.container_instance.calculate_total_dimensions()
        center = self.calculate_centroid(copied.coordinates)
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, copied.coordinates)

        ops_angle = (angle + 180) % 360

        vx2, vy2 = (
            math.cos(math.radians(ops_angle)), math.sin(math.radians(ops_angle)))

        x4, y4 = self.calculate_endpoint_from_direction(cx, cy, vx2, vy2, dime)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x4, y4)]
        po1 = self.find_farthest_point_from_line_special(line1, right, copied.coordinates, vx2, vy2, dime,
                                                         convex_center)
        qo1 = self.find_farthest_point_from_line_special2(line1, right, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        po2 = self.find_farthest_point_from_line_special(line1, left, copied.coordinates, vx2, vy2, dime, convex_center)
        qo2 = self.find_farthest_point_from_line_special2(line1, left, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        (px1, py1) = po1
        (px2, py2) = po2

        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        the_point1 = None
        right_angle = self.calculate_angle_in_degrees((px1, py1), p1)
        left_angle = self.calculate_angle_in_degrees((px2, py2), p2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point1 = (px1, py1)
        else:
            the_point1 = (px2, py2)

        (qx1, qy1) = qo1
        (qx2, qy2) = qo2

        q1 = self.calculate_endpoint_from_direction(qx1, qy1, vx, vy, dime)
        q2 = self.calculate_endpoint_from_direction(qx2, qy2, vx, vy, dime)

        the_point2 = None
        right_angle = self.calculate_angle_in_degrees((qx1, qy1), q1)
        left_angle = self.calculate_angle_in_degrees((qx2, qy2), q2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point2 = (qx1, qy1)
        else:
            the_point2 = (qx2, qy2)

        return (px2, py2), (qx1, qy1), left

    def check_ep2(self, angle, middle_polygon, center, polygon):
        dime = self.container_instance.calculate_total_dimensions()
        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, middle_polygon)
        center = self.calculate_centroid(polygon.coordinates)
        cx, cy = center

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x1, y1)]
        po1 = self.find_farthest_point_from_line(line1, right, polygon, vx, vy, dime)

        return po1

    def extend_pol_for_first_time(self, angle, polygon, center):
        dime = self.container_instance.calculate_total_dimensions()
        cx, cy = center

        vx, vy = (math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, polygon.coordinates)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x1, y1)]

        px1, py1 = self.find_farthest_point_from_line(line1, right, polygon.coordinates, vx, vy, dime)
        px2, py2 = self.find_farthest_point_from_line(line1, left, polygon.coordinates, vx, vy, dime)

        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        right_line = LineString([(px1, py1), p1])
        left_line = LineString([(px2, py2), p2])

        filled_polygon = Polygon(list(left_line.coords) + list(right_line.coords)[::-1])

        return filled_polygon, right_line, left_line

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
            print("beacuse empty")
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

    def find_intersection_point_polygon(self, polygon_coordinates, polygon2, po):
        # Create a polygon from the given coordinates

        polygon = Polygon(polygon_coordinates)

        exterior_ring = polygon.exterior

        # Create a LineString from the given line coordinates
        polygon2 = Polygon(polygon2)

        # Find the intersection between the line and the polygon
        intersection = polygon2.intersection(exterior_ring)

        # Check the type of the result to handle different cases
        if intersection.is_empty:
            print("beacuse empty")
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

    def find_intersection_point_special(self, polygon_coordinates, line_coords, po):
        # Create a polygon from the given coordinates

        polygon = Polygon(polygon_coordinates)
        polygon = polygon.buffer(0.1)

        exterior_ring = polygon.exterior

        # Create a LineString from the given line coordinates
        line = LineString(line_coords)

        # Find the intersection between the line and the polygon
        intersection = line.intersection(exterior_ring)

        # Check the type of the result to handle different cases
        if intersection.is_empty:
            print("beacuse empty")
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

    def find_intersection_point_linestring(self, line_string, line_coords, po):
        # Create a polygon from the given coordinates

        # Create a LineString from the given line coordinates
        line = LineString(line_coords)

        # Find the intersection between the line and the polygon
        intersection = line.intersection(line_string)

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

    def calculate_endpoint_from_rounded_angle(self, x1, y1, desired_angle, length, precision=6):
        # Calculate the end point
        angle = math.atan2(y1, x1)
        current_angle = math.degrees(angle)  # Convert to degrees

        # Find the difference between the desired angle and the current angle
        angle_difference = desired_angle - current_angle

        # Round the angle difference to the specified precision
        rounded_angle_difference = round(angle_difference, precision)

        # Calculate the adjusted endpoint using the rounded angle difference
        adjusted_angle = math.radians(current_angle + rounded_angle_difference)
        dx = math.cos(adjusted_angle)
        dy = math.sin(adjusted_angle)

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

        return min(width, height)

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
        # print("before",mergedPolys)

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

    def calculate_next_angle(self, current_angle, inc):
        threshold = self.container_instance.calculate_distance_threshold()
        dx, dy = (
            math.cos(math.radians(current_angle)), math.sin(math.radians(current_angle)))

        # pol_area = current_polygon.calculate_total_dimensions()
        # next_pol_area = next_polygon.calculate_total_dimensions()
        convex_area = self.container_instance.calculate_total_dimensions()
        # Calculate the distance adjustment based on both width and height
        distance = inc

        radius = math.sqrt(dx ** 2 + dy ** 2)
        new_angle = current_angle + math.atan2(dy, dx) + (distance / radius)

        # Ensure the angle is within [0, 360] degrees
        return new_angle

    def warning_handler(self, message, category, filename, lineno, file=None, line=None):
        self.error_occurred = True
        print(f"Warning: {category.__name__}: {message}")

    def calculate_angle_in_degrees(self, point, centroid):
        angle_radians = math.atan2(centroid[1] - point[1], centroid[0] - point[0])
        angle_degrees = math.degrees(angle_radians)
        return angle_degrees

    def check_if_line_cross(self, convex_polygon, polygon):
        new_list = []
        pol = Polygon(polygon.coordinates)
        center_con = self.calculate_centroid(convex_polygon.coordinates)

        for point in polygon.coordinates:
            line = LineString([center_con, point])
            if not line.crosses(pol):
                if point not in new_list:
                    new_list.append(point)
        return new_list

    def calculate_centroid(self, coords):
        # Create a convex polygon from the given coordinates
        convex_polygon = Polygon(coords)

        # Calculate the centroid of the convex polygon
        centroid = convex_polygon.centroid

        return centroid.x, centroid.y

    def polygon_to_rectangle(self, coords):
        # Check if the input list of coordinates is not empty
        if not coords:
            return None

        # Create a Shapely Polygon object
        polygon = Polygon(coords)

        # Find the minimum rotated rectangle
        min_rect = MultiPoint(polygon.exterior.coords).minimum_rotated_rectangle

        # Extract the coordinates of the rectangle vertices
        rect_coords = list(min_rect.exterior.coords)

        return rect_coords

    def find_edges_with_vertex(self, polygon, target_vertex):
        edges = polygon.get_edge_lines()

        """
            Find points of edges that have the specified target_vertex.

            Parameters:
            - edges: A list of edges, where each edge is represented by a tuple of two vertices.
            - target_vertex: The vertex to search for in the edges.

            Returns:
            - List of points of edges that contain the target_vertex (excluding the target_vertex).
            """
        matching_edges = [edge for edge in edges if target_vertex in edge]
        print(matching_edges)

        matching_points = [point for edge in matching_edges for point in edge if point != target_vertex]
        return list(set(matching_points))

    def check_ep3(self, angle, p, convex_center, new_center):
        new_pol = Polygon(p.coordinates)
        new_pol = new_pol.buffer(0.1)
        copied = copy.deepcopy(p)
        list_of = list(new_pol.exterior.coords)
        list_of = self.polygon_to_rectangle(list_of)
        copied.set_coordinates(list_of)
        dime = self.container_instance.calculate_total_dimensions()
        # center = self.calculate_centroid(copied.coordinates)
        center = new_center
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, copied.coordinates)

        ops_angle = (angle + 180) % 360

        vx2, vy2 = (
            math.cos(math.radians(ops_angle)), math.sin(math.radians(ops_angle)))

        x4, y4 = self.calculate_endpoint_from_direction(cx, cy, vx2, vy2, dime)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x4, y4)]
        po1 = self.find_farthest_point_from_line_special(line1, right, copied.coordinates, vx2, vy2, dime,
                                                         convex_center)
        qo1 = self.find_farthest_point_from_line_special2(line1, right, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        po2 = self.find_farthest_point_from_line_special(line1, left, copied.coordinates, vx2, vy2, dime, convex_center)
        qo2 = self.find_farthest_point_from_line_special2(line1, left, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        (px1, py1) = po1
        (px2, py2) = po2
        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        the_point = None
        right_angle = self.calculate_angle_in_degrees((px1, py1), p1)
        left_angle = self.calculate_angle_in_degrees((px2, py2), p2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point = (px1, py1)
        else:
            the_point = (px2, py2)

        (qx1, qy1) = qo1
        (qx2, qy2) = qo2

        q1 = self.calculate_endpoint_from_direction(qx1, qy1, vx, vy, dime)
        q2 = self.calculate_endpoint_from_direction(qx2, qy2, vx, vy, dime)

        the_point2 = None
        right_angle = self.calculate_angle_in_degrees((qx1, qy1), q1)
        left_angle = self.calculate_angle_in_degrees((qx2, qy2), q2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point2 = (qx1, qy1)
        else:
            the_point2 = (qx2, qy2)

        return (px2, py2), (qx1, qy1), left

    def check_ep3_temp(self, angle, p, convex_center, new_center):
        new_pol = Polygon(p.coordinates)
        new_pol = new_pol.buffer(0.1)
        copied = copy.deepcopy(p)
        list_of = list(new_pol.exterior.coords)
        list_of = self.polygon_to_rectangle(list_of)
        copied.set_coordinates(list_of)
        dime = self.container_instance.calculate_total_dimensions()
        # center = self.calculate_centroid(copied.coordinates)
        center = new_center
        cx, cy = center

        vx, vy = (
            math.cos(math.radians(angle)), math.sin(math.radians(angle)))
        left, right = self.classify_points_left_right1(angle, center, copied.coordinates)
        print(center)
        print(left)
        print(right)

        ops_angle = (angle + 180) % 360

        vx2, vy2 = (
            math.cos(math.radians(ops_angle)), math.sin(math.radians(ops_angle)))

        x4, y4 = self.calculate_endpoint_from_direction(cx, cy, vx2, vy2, dime)

        x1, y1 = self.calculate_endpoint_from_direction(cx, cy, vx, vy, dime)
        line1 = [(cx, cy), (x4, y4)]
        po1 = self.find_farthest_point_from_line_special(line1, right, copied.coordinates, vx2, vy2, dime,
                                                         convex_center)
        qo1 = self.find_farthest_point_from_line_special2(line1, right, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)
        print(po1, qo1)

        po2 = self.find_farthest_point_from_line_special(line1, left, copied.coordinates, vx2, vy2, dime, convex_center)
        qo2 = self.find_farthest_point_from_line_special2(line1, left, copied.coordinates, vx2, vy2, dime,
                                                          convex_center)

        (px1, py1) = po1
        (px2, py2) = po2
        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, dime)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, dime)

        the_point = None
        right_angle = self.calculate_angle_in_degrees((px1, py1), p1)
        left_angle = self.calculate_angle_in_degrees((px2, py2), p2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point = (px1, py1)
        else:
            the_point = (px2, py2)

        (qx1, qy1) = qo1
        (qx2, qy2) = qo2

        q1 = self.calculate_endpoint_from_direction(qx1, qy1, vx, vy, dime)
        q2 = self.calculate_endpoint_from_direction(qx2, qy2, vx, vy, dime)

        the_point2 = None
        right_angle = self.calculate_angle_in_degrees((qx1, qy1), q1)
        left_angle = self.calculate_angle_in_degrees((qx2, qy2), q2)
        right_angle = (right_angle % 360)
        left_angle = (left_angle % 360)

        if right_angle > left_angle:
            the_point2 = (qx1, qy1)
        else:
            the_point2 = (qx2, qy2)

        return (px2, py2), (px1, py1), left

    def farthest_point_from_polygon(self, polygon1, polygon2):
        # Convert input parameters to Shapely Polygon objects
        poly1 = Polygon(polygon1)
        poly2 = Polygon(polygon2)

        # Get the centroid of polygon1
        centroid1 = poly1.centroid

        # Find the point in polygon2 that is farthest from the centroid of polygon1
        farthest_point = max(poly2.exterior.coords, key=lambda point: centroid1.distance(Point(point)))

        return farthest_point

    def create_antiparallel_line(self, line):
        # Extract the coordinates of the line
        (x1, y1), (x2, y2) = line

        # Calculate the direction vector
        direction_vector = np.array([x2 - x1, y2 - y1])

        # Normalize the direction vector to preserve the length
        normalized_direction = direction_vector / np.linalg.norm(direction_vector)

        # Rotate the normalized direction vector by -45 degrees
        rotation_angle = np.radians(-90)
        rotated_direction = np.array([
            normalized_direction[0] * np.cos(rotation_angle) - normalized_direction[1] * np.sin(rotation_angle),
            normalized_direction[0] * np.sin(rotation_angle) + normalized_direction[1] * np.cos(rotation_angle)
        ])

        # Specify the desired length for the antiparallel line
        length = np.linalg.norm(direction_vector)

        # Calculate the coordinates for the antiparallel line
        x3, y3 = x2 + rotated_direction[0] * length, y2 + rotated_direction[1] * length

        return [(x2, y2), (x3, y3)]

    def project_point_to_linestring_edge_new(self, linestring, interior_point):
        """
        Project a point onto the nearest edge of a given LineString.

        Parameters:
        linestring (LineString): LineString geometry.
        interior_point (tuple): Coordinates of the point inside the LineString.

        Returns:
        tuple: Coordinates of the projected point on the LineString edge.
        """
        # Create a Shapely LineString from the provided LineString
        line = LineString(linestring)

        # Create a Shapely Point for the interior point
        point = Point(interior_point)

        # Project the interior point onto the LineString
        projected_point = line.interpolate(line.project(point))

        return projected_point.x, projected_point.y

    def project_point_to_convex_edge(self,convex_region, interior_point):
        """
        Project a point inside a convex region onto the nearest edge of the convex hull.

        Parameters:
        convex_region (list of tuples): Vertices of the convex region in counter-clockwise order.
        interior_point (tuple): Coordinates of the point inside the convex region.

        Returns:
        tuple: Coordinates of the projected point on the convex hull edge.
        """
        # Create a Shapely Polygon from the convex region
        polygon = Polygon(convex_region)

        # Create a Shapely Point for the interior point
        point = Point(interior_point)

        # Project the interior point onto the boundary of the convex hull
        projected_point = polygon.exterior.interpolate(polygon.exterior.project(point))

        return projected_point.x, projected_point.y

    def find_edge(self, convex_region, point):
        for i in range(len(convex_region)):
            edge_point1 = convex_region[i]
            edge_point2 = convex_region[(i + 1) % len(convex_region)]
            p = Point(point)
            line = LineString([edge_point1, edge_point2])
            if self.is_point_on_edge(point,edge_point1, edge_point2):
                return [edge_point1, edge_point2]

        return None

    def is_point_on_edge(self,point, edge_point1, edge_point2):
        # Check if the point is collinear with the edge
        cross_product = (edge_point2[1] - edge_point1[1]) * (point[0] - edge_point1[0]) - \
                        (edge_point2[0] - edge_point1[0]) * (point[1] - edge_point1[1])

        # Allow for a small tolerance due to floating-point precision
        tolerance = 1e-10

        return abs(cross_product) < tolerance

    def extend_line(self,coords, length):
        # Ensure the line has at least two points
        if len(coords) < 2:
            raise ValueError("Line must have at least two points")

        # Get the last two coordinates of the line
        x1, y1 = coords[-2]
        x2, y2 = coords[-1]

        # Calculate the direction of the line
        dx = x2 - x1
        dy = y2 - y1

        # Normalize the direction vector
        magnitude = (dx ** 2 + dy ** 2) ** 0.5
        normalized_dx = dx / magnitude
        normalized_dy = dy / magnitude

        # Calculate the new endpoint based on the normalized direction and length
        new_x = x2 + normalized_dx * length
        new_y = y2 + normalized_dy * length

        return new_x, new_y

    def temp_func(self, polygon, p_of_convex, p_of_middle, convex_region):
        ed_li = self.find_edges_with_vertex(polygon, p_of_convex)

        angle1 = self.calculate_angle_in_degrees(ed_li[0], p_of_convex)
        angle2 = self.calculate_angle_in_degrees(ed_li[1], p_of_convex)
        angle1_180 = (angle1 + 180) % 360

        angle2_180 = (angle2 + 180) % 360
        dime = Point(p_of_convex).distance(Point(p_of_middle))
        vx11, vy11 = (
            math.cos(math.radians(angle1)),
            math.sin(math.radians(angle1)))
        vx22, vy22 = (
            math.cos(math.radians(angle2)),
            math.sin(math.radians(angle2)))
        vx33, vy33 = (
            math.cos(math.radians(angle1_180)),
            math.sin(math.radians(angle1_180)))
        vx44, vy44 = (
            math.cos(math.radians(angle2_180)),
            math.sin(math.radians(angle2_180)))

        h1, h2 = ed_li[0]
        b1, b2 = ed_li[1]

        h3, h4 = self.calculate_endpoint_from_direction(h1, h2, vx11, vy11, dime)
        b3, b4 = self.calculate_endpoint_from_direction(b1, b2, vx22, vy22, dime)
        h5, h6 = self.calculate_endpoint_from_direction(h1, h2, vx33, vy33, dime)
        b7, b8 = self.calculate_endpoint_from_direction(b1, b2, vx44, vy44, dime)
        list_of_p = [(h3, h4), (b3, b4), (h5, h6), (b7, b8)]

        min_distance = float('inf')
        closest_point = None
        # if 0 or 2 ed_li[0]
        # if 1 or 3 ed_li[1]
        num = None
        count = 0
        for point in list_of_p:
            dist = Point(point).distance(Point(p_of_middle))
            if dist < min_distance:
                min_distance = dist
                closest_point = point
                if count % 2 == 0:
                    num = ed_li[0]
                else:
                    num = ed_li[1]
            count = count + 1

        dis1 = Point(closest_point).distance(Point(p_of_convex))
        dis2 = Point(closest_point).distance(Point(num))
        if dis1 > dis2:
            p_of_st = num
        else:
            p_of_st = p_of_convex


        dime2 = self.container_instance.calculate_total_dimensions()
        point = self.extend_line([closest_point, p_of_convex], dime)
        line = LineString([closest_point, point])
        convex_region_original = convex_region
        convex_region = Polygon(convex_region)
        convex_region = convex_region.exterior
        empty = []
        newl = []
        newl.append([closest_point, point])


        intresection_p = convex_region.intersection(line)
        poi = []

        new_line = self.find_edge(convex_region_original, (intresection_p.x,intresection_p.y))



        pol = Polygon(convex_region_original)
        if Point(p_of_st).within(pol):
            proj_p = self.project_point_to_linestring_edge_new(new_line, p_of_st)
        else:
            proj_p = self.project_point_to_convex_edge(convex_region_original, p_of_st)


        return proj_p,ed_li[0],ed_li[1],p_of_convex, [(h3, h4), (b3, b4), (h5, h6), (b7, b8)]

    def intersection_of_lines4(self, vertical_line, horizontal_line, prev_point_of_pol, the_point, convex_region):
        dime = self.container_instance.calculate_total_dimensions()
        # p4 = self.find_perpendicular_point_on_convex_region(prev_point_of_pol, convex_region)
        angle_ch = self.calculate_angle_in_degrees(prev_point_of_pol, the_point)
        angle_ch1 = (angle_ch + 180) % 360
        angle_ch2 = angle_ch % 360
        vx, vy = (
            math.cos(math.radians(angle_ch1)), math.sin(math.radians(angle_ch1)))
        vx2, vy2 = (
            math.cos(math.radians(angle_ch2)), math.sin(math.radians(angle_ch2)))
        g1, g2 = prev_point_of_pol
        point1 = self.calculate_endpoint_from_direction(g1, g2, vx, vy, dime)
        point2 = self.calculate_endpoint_from_direction(g1, g2, vx2, vy2, dime)

        # Define the coordinates of the endpoints of the two lines
        hor = horizontal_line
        vert = vertical_line
        vertical_line = LineString(vertical_line)
        horizontal_line = LineString(horizontal_line)
        main_line1 = LineString([(g1, g2), point1])
        main_line2 = LineString([(g1, g2), point2])

        # Find the intersection point
        intersection1 = vertical_line.intersection(main_line1)
        intersection2 = horizontal_line.intersection(main_line1)
        intersection3 = vertical_line.intersection(main_line2)
        intersection4 = horizontal_line.intersection(main_line2)
        pol = Polygon(convex_region)
        len1 = 0
        intersection_result1 = main_line1.intersection(pol)
        intersection_type1 = type(intersection_result1)
        if intersection_type1 == LineString:
            len1 = intersection_result1.length
        len2 = 0
        intersection_result2 = main_line2.intersection(pol)
        intersection_type2 = type(intersection_result2)
        if intersection_type2 == LineString:
            len2 = intersection_result2.length

        if len1 > len2:
            print("check1")
            if not intersection1.is_empty and not intersection2.is_empty:
                p1 = Point((g1, g2))
                p2 = Point((intersection1.x, intersection1.y))
                p3 = Point((intersection2.x, intersection2.y))
                dis1 = p1.distance(p2)
                dis2 = p1.distance(p3)
                if dis1 < dis2:
                    return intersection1.x, intersection1.y
                else:
                    return intersection2.x, intersection2.y
            elif not intersection1.is_empty:
                return intersection1.x, intersection1.y
            elif not intersection2.is_empty:
                return intersection2.x, intersection2.y
            else:
                print("type of ", type(main_line1.intersection(pol)))
                intersection_result = main_line1.intersection(pol)
                intersection_type = type(intersection_result)
                if intersection_type == LineString:
                    print(intersection_result.length)

                print("here")
                return (g1, g2), point1
        elif len2 > len1:
            print("check2")
            if not intersection3.is_empty and not intersection4.is_empty:
                p1 = Point((g1, g2))
                p2 = Point((intersection3.x, intersection3.y))
                p3 = Point((intersection4.x, intersection4.y))
                dis1 = p1.distance(p2)
                dis2 = p1.distance(p3)
                if dis1 < dis2:
                    return intersection3.x, intersection3.y
                else:
                    return intersection4.x, intersection4.y
            elif not intersection3.is_empty:
                return intersection3.x, intersection3.y
            elif not intersection4.is_empty:
                return intersection4.x, intersection4.y
            else:
                return (g1, g2), point2
        else:
            print("here")
            return (g1, g2), point1

    def create_lines(self, vertices):
        # Find the center of the convex region
        center_x, center_y = self.calculate_centroid(vertices)

        # Find the minimum and maximum coordinates
        min_x, min_y = min(x for x, y in vertices), min(y for x, y in vertices)
        max_x, max_y = max(x for x, y in vertices), max(y for x, y in vertices)



        # Create LineString objects for the vertical and horizontal lines
        vertical_line = LineString([(center_x, min_y), (center_x, max_y)])
        horizontal_line = LineString([(min_x, center_y), (max_x, center_y)])

        return [(center_x, min_y), (center_x, max_y)], [(min_x, center_y), (max_x, center_y)]



    def plot(self):
        angle = 0
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        middle_point = self.calculate_centroid(self.container_instance.coordinates)
        convex_region = self.container_instance.coordinates
        convex_region_original = self.container_instance.coordinates
        convex_region_less_detailed = self.container_instance.coordinates

        another_list = []
        temp_po = []
        temp_list = []
        value = 0
        start_time = time.time()
        previous_polygon = None
        for dex, polygon in enumerate(sorted_items):
            if dex == 700:
                break
            print(dex)

            x, y = middle_point
            polygon.move_item(x, y)
            copied = copy.deepcopy(polygon)
            f_p = None
            t_p = None
            pol2 = Polygon(polygon.coordinates)
            pol1 = Polygon(convex_region)
            if pol2.within(pol1):
                if dex == 0:
                    extended_polygon, right_line, left_line = self.extend_pol_for_first_time(angle, polygon,
                                                                                             middle_point)
                    list_of_co = list(extended_polygon.exterior.coords)
                    polygon.ex_dime = self.calculate_width_and_height(list_of_co)
                    f_p, t_p, list_of_lines, list_of_points = self.place_poly(polygon, extended_polygon, convex_region,
                                                                              angle, right_line, left_line)
                    polygon.move_from_to2(f_p, t_p)
                    the_point, sec_point, left_list = self.check_ep(angle, polygon, middle_point)
                    polygon.left_point = the_point
                    polygon.right_point = sec_point

                    # polygon.sec_left_point = sec_point
                    # left_list = self.check_ep2(angle, polygon)
                    polygon.left_list = left_list
                    polygon.curr_angle = angle

                    li = self.extend_pol(angle, convex_region, polygon)

                    list_of_new_region = self.for_edges_that_intersect(Polygon(convex_region),
                                                                       Polygon(polygon.coordinates))
                    list_of_new_region2 = self.for_edges_that_intersect(Polygon(convex_region),
                                                                        Polygon(li))
                    convex_region = list_of_new_region
                    convex_region_less_detailed = list_of_new_region2
                    middle_point = self.calculate_centroid(convex_region)
                    another_list.append(polygon)
                    value = value + polygon.value
                    previous_polygon = polygon

                if dex >= 1:
                    list_of_lines = []
                    list_of_points = []
                    # Get the polygon before the one at index dex
                    # previous_polygon = sorted_items[dex - 1]
                    i = 0
                    flag_temp = False
                    sec_flag = False
                    while not flag_temp:
                        extended_polygon = None
                        right_line = None
                        left_line = None
                        p_of_convex = None
                        new_p_of_convex = None
                        stop_flag = False

                        p_of_middle = None
                        this_point = None
                        other_point = None

                        points = self.check_if_line_cross(previous_polygon, polygon)

                        for j_index in range(2):
                            poi = []
                            for point in points:
                                line = []
                                a = None
                                angle = self.calculate_angle_in_degrees(point, previous_polygon.left_point)
                                while True:
                                    temp_a = a
                                    a, b, c = self.check_ep(angle, previous_polygon, point)
                                    if a == temp_a and a is not None:
                                        a, b, c = self.check_ep3(angle, previous_polygon, point, a)
                                    angle = self.calculate_angle_in_degrees(point, a)
                                    l = None
                                    if True:
                                        dime = self.container_instance.calculate_total_dimensions()
                                        xx, yy = point
                                        this_angle = (angle + 0.01 % 360)
                                        vx, vy = (
                                            math.cos(math.radians(this_angle)),
                                            math.sin(math.radians(this_angle)))
                                        xxx, yyy = self.calculate_endpoint_from_direction(xx, yy, vx, vy, dime)
                                        l = LineString([point, (xxx, yyy)])


                                    p = Polygon(previous_polygon.coordinates)
                                    p = p.buffer(0.1)
                                    if not l.crosses(p):
                                        to_point_temp = a
                                        # angle = angle_before
                                        break

                                angle = (angle + 0.01 % 360)

                                flag, d1, d2, d3, d4, d5, d6, extended_poly, right_li, left_li = self.placement(
                                    angle,
                                    polygon.coordinates,
                                    previous_polygon)

                                print(flag)

                                line.append([point, a])
                                poi.append(point)
                                poi.append(a)

                                copied2 = copy.deepcopy(polygon)
                                copied2.set_coordinates(convex_region)

                                copied.set_coordinates(extended_poly.exterior.coords)
                                another_list.append(copied)
                                another_list.append(polygon)
                                aru = []
                                aru.append(copied)
                                aru.append(copied2)
                                aru.append(polygon)
                                # aru.append(previous_polygon)

                                another_list.pop()
                                another_list.pop()


                                if flag:
                                    sec_flag = True
                                    extended_polygon = extended_poly
                                    right_line = right_li
                                    left_line = left_li
                                    if j_index == 0:
                                        ppp = Polygon(previous_polygon.coordinates)
                                        ppp = ppp.buffer(0.1)
                                        copiedp = copy.deepcopy(polygon)
                                        copiedp.set_coordinates(list(ppp.exterior.coords))
                                        proj_p,e,d,f, li_p = self.temp_func(copiedp, a, point, convex_region_original)
                                        vertical_line, horizontal_line = self.create_lines(convex_region)
                                        p = self.intersection_of_lines4(vertical_line, horizontal_line,
                                                                        a,
                                                                        proj_p,
                                                                        convex_region_original)
                                        if dex >= 1:
                                            po = []
                                            line3 = []
                                            #po.append(e)
                                            po.append(d)
                                            #po.append(f)








                                            draw_instance = Draw(self.container_instance, aru, (1, 1), (1, 1), (1, 1),
                                                                 (1, 1),
                                                                 po,
                                                                 None,
                                                                 None, line3)
                                            draw_instance.plot()

                                        rec_temp = Polygon(polygon.move_from_to2_value(point, p))
                                        if Polygon(rec_temp).within(pol1):
                                            polygon.move_from_to2(point, p)
                                            points = self.check_if_line_cross(previous_polygon, polygon)
                                        else:
                                            stop_flag = True


                                    break

                            if stop_flag:
                                break


                        # angle = min
                        if sec_flag:
                            f_p, t_p, list_of_lines, list_of_points = self.place_poly(polygon, extended_polygon,
                                                                                      convex_region, angle, right_line,
                                                                                      left_line)

                            polygon.move_from_to2(f_p, t_p)
                            another_list.append(polygon)
                            the_point, sec_point, left_list = self.check_ep(angle, polygon, middle_point)
                            polygon.left_point = the_point
                            polygon.right_point = sec_point

                            # polygon.sec_left_point = sec_point
                            # left_list = self.check_ep2(angle, polygon)
                            polygon.left_list = left_list
                            polygon.curr_angle = angle
                            polygon.leftline = this_point
                            polygon.rightline = other_point

                            if dex >= 89:
                                draw_instance = Draw(self.container_instance, another_list, (1, 1), (1, 1), (1, 1),
                                                     (1, 1),
                                                     None,
                                                     None,
                                                     None, None)
                                draw_instance.plot()

                            convex_region = convex_region_less_detailed
                            list_of_new_region = self.for_edges_that_intersect(Polygon(convex_region),
                                                                               Polygon(polygon.coordinates))

                            li = self.extend_pol(angle, convex_region, polygon)

                            list_of_new_region2 = self.for_edges_that_intersect(Polygon(convex_region),
                                                                                Polygon(li))
                            convex_region = list_of_new_region
                            convex_region_less_detailed = list_of_new_region2
                            middle_point = self.calculate_centroid(convex_region)
                            previous_polygon = polygon

                            break

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)
        print("num of polygons", len(another_list), "out of", len(self.item_instances), "time", elapsed_time, "value",
              value)

        draw_instance = Draw(self.container_instance, another_list, (1, 1), (1, 1), (1, 1), (1, 1), None,
                             None,
                             None, None)
        draw_instance.plot()





