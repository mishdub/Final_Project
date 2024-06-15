from Draw import Draw
import math
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, MultiPoint

from shapely.ops import unary_union
import time
import warnings
import copy
from PolygonTangentFinder import PolygonTangentFinder
from shapely.ops import triangulate



class Algo50:

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
            print("not intresectino")
            return None, None, None, None

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

        if point2 is None and point1 is not None:
            point1 = Point(point1)
            dis5 = spoint1.distance(point1)
            if dis5 < min_dis_in_multi_and_line:
                f_p = (spoint1.x, spoint1.y)
                t_p = (point1.x, point1.y)
        elif point1 is None and point2 is not None:
            point2 = Point(point2)
            dis6 = spoint2.distance(point2)
            if dis6 < min_dis_in_multi_and_line:
                f_p = (spoint2.x, spoint2.y)
                t_p = (point2.x, point2.y)
        elif point1 is not None and point2 is not None:
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

    def placement3(self, angle, middle_polygon, convex_region,a,polygon_curr,original_convex_Region):
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


        new_dist = Point((px1, py1)).distance(Point(a))
        new_dist2 = Point((px2, py2)).distance(Point(a))
        new_dist3 = Point((px1, py1)).distance((Polygon(original_convex_Region)).exterior)
        new_dist4 = Point((px2, py2)).distance((Polygon(original_convex_Region)).exterior)

        if new_dist > new_dist3:
            new_dist = new_dist3

        if new_dist2 > new_dist4:
            new_dist2 = new_dist4

        p1 = self.calculate_endpoint_from_direction(px1, py1, vx, vy, new_dist)
        p2 = self.calculate_endpoint_from_direction(px2, py2, vx, vy, new_dist2)

        right_line = LineString([(px1, py1), p1])
        left_line = LineString([(px2, py2), p2])

        filled_polygon = Polygon(list(left_line.coords) + list(right_line.coords)[::-1])

        flag = False
        pol = Polygon(convex_region).exterior

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

        if not Polygon(exterior_coords_list).is_valid:
            exterior_coords_list = self.polygon_to_rectangle(exterior_coords_list)

        return exterior_coords_list

    def check_ep(self, angle, p, convex_center):
        new_pol = p.extend_polygon(1.1)
        copied = copy.deepcopy(p)
        copied.set_coordinates(new_pol)
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

    def check_ep_without_rec(self, angle, p, convex_center, size_extend):
        new_pol = p.extend_polygon(size_extend)

        copied = copy.deepcopy(p)
        copied.set_coordinates(new_pol)
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

        return (px2, py2), (px1, py1), left

    def check_ep_new(self, angle, p, convex_center, size_extend):
        new_pol = p.extend_polygon(size_extend)

        copied = copy.deepcopy(p)
        copied.set_coordinates(new_pol)
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

    def check_ep_rec_new(self, angle, p, convex_center, size_extend):
        new_pol = p.extend_polygon(size_extend)
        copied = copy.deepcopy(p)
        list_of = new_pol
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

    def polygon_to_rectangle2(self, coords):
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
    def polygon_to_rectangle(self, coords):
        # Check if the input list of coordinates is not empty
        if not coords:
            return None

        # Find the minimum and maximum coordinates of the polygon
        min_x = min(coord[0] for coord in coords)
        min_y = min(coord[1] for coord in coords)
        max_x = max(coord[0] for coord in coords)
        max_y = max(coord[1] for coord in coords)

        # Create rectangle coordinates using min and max values
        rect_coords = [(min_x, min_y), (min_x, max_y), (max_x, max_y), (max_x, min_y)]

        return rect_coords

    def check_ep3_without_rec(self, angle, p, convex_center, new_center, size_extend):
        big_p = p.extend_polygon(size_extend)
        copied = copy.deepcopy(p)
        copied.set_coordinates(big_p)
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
        print("is there",right)
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

    def check_ep3_new(self, angle, p, convex_center, new_center, size_extend):
        new_pol = p.extend_polygon(size_extend)
        copied = copy.deepcopy(p)
        list_of = self.polygon_to_rectangle(new_pol)
        copied.set_coordinates(list_of)
        dime = self.container_instance.calculate_total_dimensions()
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

    def intersection_of_lines4(self, vertical_line, horizontal_line, angle_par, point, convex_region):
        dime = self.container_instance.calculate_total_dimensions()

        angle_ch2 = (angle_par + 180) % 360
        vx2, vy2 = (
            math.cos(math.radians(angle_ch2)), math.sin(math.radians(angle_ch2)))
        g1, g2 = point
        point2 = self.calculate_endpoint_from_direction(g1, g2, vx2, vy2, dime)

        vertical_line = LineString(vertical_line)
        horizontal_line = LineString(horizontal_line)
        main_line2 = LineString([(g1, g2), point2])
        print("check this point",(g1, g2), point2)

        intersection3 = vertical_line.intersection(main_line2)
        intersection4 = horizontal_line.intersection(main_line2)

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
            return False


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

    def for_edges_that_intersect2(self, pol1, pol2):
        big_p = pol2.extend_polygon(1)

        buffered_result = Polygon(big_p)

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

        else:
            # If it's a single Polygon, get its exterior coordinates directly
            return list(mergedPolys.exterior.coords)

    def placement2(self, angle, middle_polygon, convex_polygon):
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
        big_p = convex_polygon.extend_polygon(1)
        if not (filled_polygon.intersects(Polygon(big_p))):
            flag = True
        return flag, (px1, py1), p1, (px2, py2), p2, (cx, cy), (x1, y1), filled_polygon, right_line, left_line

    def find_grid_cell(self, point):
        # Calculate grid cell coordinates
        cell_x = int(point[0])
        cell_y = int(point[1])

        # Calculate grid cell corners
        top_left = (cell_x, cell_y)
        top_right = (cell_x + 1, cell_y)
        bottom_left = (cell_x, cell_y + 1)
        bottom_right = (cell_x + 1, cell_y + 1)

        return [top_left, top_right, bottom_left, bottom_right]


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


    def plot(self):
        original_cr = self.shrink_polygon(10, self.container_instance.coordinates)
        angle = 0
        current_angle = 0  # Starting angle
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=False)
        middle_point = self.calculate_centroid(original_cr)
        convex_region = original_cr
        convex_region_var2 = original_cr

        convex_region_original = original_cr
        convex_region_less_detailed = original_cr


        another_list = []
        value = 0
        start_time = time.time()
        previous_polygon = None

        for dex, polygon in enumerate(sorted_items):
            if dex == 10000:
                break
            print(dex)

            x, y = middle_point
            polygon.move_item(x, y)

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
                    polygon.left_list = left_list
                    polygon.curr_angle = angle

                    li = self.extend_pol(angle, convex_region, polygon)

                    list_of_new_region = self.for_edges_that_intersect2(Polygon(convex_region),
                                                                        polygon)
                    cop_li = copy.deepcopy(polygon)
                    cop_li.set_coordinates(li)
                    list_of_new_region2 = self.for_edges_that_intersect2(Polygon(convex_region),
                                                                         cop_li)

                    convex_region = list_of_new_region
                    convex_region_var2 = list_of_new_region

                    convex_region_less_detailed = list_of_new_region2
                    middle_point = self.calculate_centroid(convex_region)
                    another_list.append(polygon)
                    value = value + polygon.value
                    previous_polygon = polygon
                    past_angle = angle
                big_p_check = previous_polygon.extend_polygon(1.1)
                big_p_check = self.polygon_to_rectangle(big_p_check)


                pal_c = (Polygon(polygon.coordinates)).intersects((Polygon(big_p_check)))
                if dex >= 1 and not pal_c:
                    flag_temp = False
                    sec_flag = False
                    while not flag_temp:
                        l_line = None
                        extended_polygon = None
                        right_line = None
                        left_line = None
                        stop_flag = False
                        this_point = None
                        other_point = None
                        to_point_temp = None
                        move_back = False

                        points = self.check_if_line_cross(previous_polygon, polygon)
                        rec_cor = self.polygon_to_rectangle(previous_polygon.coordinates)
                        copied_rec = copy.deepcopy(polygon)
                        copied_rec.set_coordinates(rec_cor)

                        for j_index in range(3):
                            if j_index == 0:
                                points = self.check_if_line_cross(copied_rec, polygon)
                            elif j_index >= 1:
                                points = self.check_if_line_cross(previous_polygon, polygon)

                            for point in points:
                                a = None
                                angle = self.calculate_angle_in_degrees(point, previous_polygon.left_point)

                                if j_index == 0:
                                    rec_const = 2
                                    while True:
                                        temp_a = a
                                        a, b, c = self.check_ep_rec_new(angle, previous_polygon, point, rec_const)
                                        if a == temp_a and a is not None:
                                            a, b, c = self.check_ep3_new(angle, previous_polygon, point, a, rec_const)
                                        angle = self.calculate_angle_in_degrees(point, a)
                                        if True:
                                            dime = self.container_instance.calculate_total_dimensions()
                                            xx, yy = point
                                            this_angle = (angle + 0.01 % 360)
                                            vx, vy = (
                                                math.cos(math.radians(this_angle)),
                                                math.sin(math.radians(this_angle)))
                                            xxx, yyy = self.calculate_endpoint_from_direction(xx, yy, vx, vy, dime)
                                            l = LineString([point, (xxx, yyy)])
                                        big_p = previous_polygon.extend_polygon(rec_const)
                                        big_p = self.polygon_to_rectangle(big_p)

                                        if not l.crosses((Polygon(big_p))):
                                            the_a = a
                                            to_point_temp = a
                                            break

                                if j_index >= 1:
                                    while True:
                                        temp_a = a
                                        a, b, c = self.check_ep_new(angle, previous_polygon, point, 1.1)
                                        if a == temp_a and a is not None:
                                            a, b, c = self.check_ep3_without_rec(angle, previous_polygon, point, a, 1.1)
                                        angle = self.calculate_angle_in_degrees(point, a)
                                        if True:
                                            dime = self.container_instance.calculate_total_dimensions()
                                            xx, yy = point
                                            this_angle = (angle + 0.01 % 360)
                                            vx, vy = (
                                                math.cos(math.radians(this_angle)),
                                                math.sin(math.radians(this_angle)))
                                            xxx, yyy = self.calculate_endpoint_from_direction(xx, yy, vx, vy, dime)
                                            l = LineString([point, (xxx, yyy)])
                                        big_p = previous_polygon.extend_polygon(1.1)


                                        if not l.crosses((Polygon(big_p))):
                                            to_point_temp = a
                                            l_line = [point, (xxx, yyy)]
                                            # angle = angle_before
                                            break

                                angle = (angle + 0.01 % 360)

                                flag, d1, d2, d3, d4, d5, d6, extended_poly, right_li, left_li = self.placement2(
                                    angle,
                                    polygon.coordinates,
                                    previous_polygon)

                                print(flag)

                                if flag:
                                    sec_flag = True
                                    extended_polygon = extended_poly
                                    right_line = right_li
                                    left_line = left_li

                                    if j_index == 0:
                                        rec_temp = Polygon(polygon.move_from_to2_value(point, a))

                                        if Polygon(rec_temp).within(pol1):
                                            polygon.move_from_to2(point, a)
                                            points = self.check_if_line_cross(previous_polygon, polygon)
                                            print("its inside")


                                        elif Polygon(rec_temp).within(Polygon(convex_region_original)) and not Polygon(
                                                rec_temp).within(pol1):
                                            print("its not inside 2")
                                            before = copy.deepcopy(polygon)

                                            polygon.move_from_to2(point, a)

                                            points = self.check_if_line_cross(previous_polygon, polygon)

                                            for point in points:
                                                b = None
                                                angle = self.calculate_angle_in_degrees(point,
                                                                                        previous_polygon.left_point)
                                                while True:
                                                    temp_b = b
                                                    a, b, c = self.check_ep_without_rec(angle, previous_polygon, point, 1.1)
                                                    if b == temp_b and b is not None:
                                                        a, b, c = self.check_ep3_without_rec(angle, previous_polygon,
                                                                                             point, b, 1.1)
                                                    angle = self.calculate_angle_in_degrees(point, b)
                                                    if True:
                                                        dime = self.container_instance.calculate_total_dimensions()
                                                        xx, yy = point
                                                        this_angle = (angle - 0.01 % 360)
                                                        vx, vy = (
                                                            math.cos(math.radians(this_angle)),
                                                            math.sin(math.radians(this_angle)))
                                                        xxx, yyy = self.calculate_endpoint_from_direction(xx, yy, vx,
                                                                                                          vy,
                                                                                                          dime)
                                                        l = LineString([point, (xxx, yyy)])

                                                    big_p = previous_polygon.extend_polygon(1.1)

                                                    if not l.crosses((Polygon(big_p))):
                                                        break

                                                angle = (angle - 0.01 % 360)



                                                flag, d1, d2, d3, d4, d5, d6, extended_poly, right_li, left_li = self.placement2(
                                                    angle,
                                                    polygon.coordinates,
                                                    previous_polygon)
                                                print(flag)


                                                if flag:
                                                    di = Point(middle_point).distance(Point(a))
                                                    before.set_coordinates(
                                                        polygon.move_item_by_dis_and_angle_value(di, angle))

                                                    di = Point(middle_point).distance(
                                                        Polygon(polygon.coordinates))
                                                    polygon.move_item_by_dis_and_angle(di, angle)

                                                    new_an = (angle + 180) % 360
                                                    di2 = (Polygon(polygon.coordinates)).distance(
                                                        (Polygon(previous_polygon.coordinates)))
                                                    polygon.move_item_by_dis_and_angle(di2, new_an)
                                                    points = self.check_if_line_cross(previous_polygon, polygon)
                                                    move_back = True
                                                    print("move back got here")

                                                    break

                                        else:
                                            print("its not inside 3")

                                            before = copy.deepcopy(polygon)
                                            polygon.move_from_to2(point, a)

                                            points = self.check_if_line_cross(previous_polygon, polygon)

                                            for point in points:
                                                b = None
                                                angle = self.calculate_angle_in_degrees(point,
                                                                                        previous_polygon.left_point)
                                                while True:
                                                    temp_b = b
                                                    a, b, c = self.check_ep_without_rec(angle, previous_polygon, point, 1.1)
                                                    if b == temp_b and b is not None:
                                                        a, b, c = self.check_ep3_without_rec(angle, previous_polygon,
                                                                                             point, b, 1.1)
                                                    angle = self.calculate_angle_in_degrees(point, b)
                                                    if True:
                                                        dime = self.container_instance.calculate_total_dimensions()
                                                        xx, yy = point
                                                        this_angle = (angle - 0.01 % 360)
                                                        vx, vy = (
                                                            math.cos(math.radians(this_angle)),
                                                            math.sin(math.radians(this_angle)))
                                                        xxx, yyy = self.calculate_endpoint_from_direction(xx, yy, vx,
                                                                                                          vy,
                                                                                                          dime)
                                                        l = LineString([point, (xxx, yyy)])

                                                    big_p = previous_polygon.extend_polygon(1.1)

                                                    if not l.crosses((Polygon(big_p))):
                                                        l =  [point, (xxx, yyy)]
                                                        po = point
                                                        b_po = b
                                                        break

                                                angle = (angle - 0.01 % 360)

                                                flag, d1, d2, d3, d4, d5, d6, extended_poly, right_li, left_li = self.placement2(
                                                    angle,
                                                    polygon.coordinates,
                                                    previous_polygon)

                                                print(flag)

                                                if flag:
                                                    di = Point(middle_point).distance(Point(a))
                                                    before.set_coordinates(
                                                        polygon.move_item_by_dis_and_angle_value(di, angle))
                                                    di = Point(middle_point).distance(
                                                        Polygon(polygon.coordinates))
                                                    polygon.move_item_by_dis_and_angle(di, angle)
                                                    new_an = (angle + 180) % 360
                                                    di2 = (Polygon(polygon.coordinates)).distance(
                                                        (Polygon(previous_polygon.coordinates)))
                                                    polygon.move_item_by_dis_and_angle(di2, new_an)
                                                    points = self.check_if_line_cross(previous_polygon, polygon)

                                                    if not Polygon(polygon.coordinates).within(pol1):
                                                        print("not inside the convex ")
                                                        move_back = True
                                                    break


                                    if j_index == 1 and move_back:
                                        vertical_line, horizontal_line = self.create_lines(convex_region)
                                        point_target = self.intersection_of_lines4(vertical_line, horizontal_line,
                                                                                   angle, a, convex_region)

                                        if isinstance(point_target, tuple):
                                            line = LineString([point_target, a])
                                            pal = polygon.move_from_to2_value(point, point_target)
                                            if Polygon(pal).within(pol1) and (not line.crosses(pol1)) and (Polygon(polygon.coordinates)).within(pol1):
                                                polygon.move_from_to2(point, point_target)
                                            else:
                                                pal_check = polygon.move_from_to2_value(point, middle_point)
                                                if Polygon(pal_check).within(pol1):
                                                    polygon.move_from_to2(point, middle_point)
                                                else:
                                                    polygon.move_item(x, y)
                                        else:
                                            pal_check = polygon.move_from_to2_value(point, middle_point)
                                            if Polygon(pal_check).within(pol1):
                                                polygon.move_from_to2(point, middle_point)
                                            else:
                                                polygon.move_item(x, y)

                                    break
                            if stop_flag:
                                break

                        if sec_flag:
                            f_p, t_p, list_of_lines, list_of_points = self.place_poly(polygon, extended_polygon,
                                                                                      convex_region, angle, right_line,
                                                                                      left_line)
                            polygon.move_from_to2(f_p, t_p)

                            another_list.append(polygon)
                            the_point, sec_point, left_list = self.check_ep(angle, polygon, middle_point)
                            polygon.left_point = the_point
                            polygon.right_point = sec_point

                            polygon.left_list = left_list
                            polygon.curr_angle = angle
                            polygon.leftline = this_point
                            polygon.rightline = other_point

                            if dex >= 100:
                                draw_instance = Draw(self.container_instance, another_list, (1, 1), (1, 1), (1, 1),
                                                     (1, 1),
                                                     None,
                                                     None,
                                                     None, None)
                                draw_instance.plot()

                            list_of_new_region = self.for_edges_that_intersect2(Polygon(convex_region),
                                                                       polygon)

                            convex_region_var2 = self.for_edges_that_intersect2(Polygon(convex_region_var2),
                                                                       polygon)

                            li = self.extend_pol(angle, convex_region, polygon)

                            cop_li = copy.deepcopy(polygon)

                            cop_li.set_coordinates(li)

                            list_of_new_region2 = self.for_edges_that_intersect2(Polygon(convex_region_less_detailed), cop_li)
                            check_ang = self.calculate_angle_in_degrees(middle_point, t_p)
                            check_ang = check_ang % 360

                            if current_angle < check_ang:
                                current_angle = check_ang
                                convex_region = list_of_new_region

                            else:
                                convex_region = list_of_new_region2
                                current_angle = check_ang

                            convex_region_less_detailed = list_of_new_region2
                            #middle_point = self.calculate_centroid(convex_region)
                            previous_polygon = polygon

                            break

            else:

                triangles = triangulate((Polygon(convex_region)))

                # Find the largest triangle by area
                triangles = [tri for tri in triangles if tri.within(pol1)]

                largest_triangle = max(triangles, key=lambda t: t.area)

                if (Polygon(polygon.coordinates)).area <= largest_triangle.area:
                    temp_pol = copy.deepcopy(polygon)
                    temp_pol2 = copy.deepcopy(polygon)
                    temp_pol3 = copy.deepcopy(polygon)

                    centroid_tri = largest_triangle.centroid
                    pol_val = polygon.move_item_value(centroid_tri.x, centroid_tri.y)

                    temp_pol.set_coordinates(list(largest_triangle.exterior.coords))
                    temp_pol2.set_coordinates(pol_val)
                    temp_pol3.set_coordinates(convex_region)
                    new_li = []
                    new_li.append(temp_pol)
                    new_li.append(temp_pol2)
                    new_li.append(temp_pol3)

                    draw_instance = Draw(self.container_instance, new_li, (1, 1), (1, 1), (1, 1), (1, 1),
                                         None,
                                         None,
                                         None, None)
                    # draw_instance.plot()
                    new_li2 = []
                    new_li2.append(temp_pol3)

                    draw_instance = Draw(self.container_instance, new_li2, (1, 1), (1, 1), (1, 1), (1, 1),
                                         None,
                                         None,
                                         None, None)
                    # draw_instance.plot()
                    if Polygon(pol_val).within(pol1):
                        polygon.move_item(centroid_tri.x, centroid_tri.y)
                        centroid_cr = self.calculate_centroid(convex_region)
                        centroid_p = self.calculate_centroid(polygon.coordinates)
                        angle_tri = self.calculate_angle_in_degrees(centroid_cr, centroid_p)
                        flag, d1, d2, d3, d4, d5, d6, extended_poly_center, right_li_center, left_li_center = self.placement2(
                            angle_tri,
                            polygon.coordinates,
                            previous_polygon)

                        f_p, t_p, list_of_lines, list_of_points = self.place_poly(polygon,
                                                                                  extended_poly_center,
                                                                                  convex_region,
                                                                                  angle_tri,
                                                                                  right_li_center,
                                                                                  left_li_center)
                        polygon.move_from_to2(f_p, t_p)
                        another_list.append(polygon)
                        list_of_new_region = self.for_edges_that_intersect2(Polygon(convex_region),
                                                                            polygon)

                        convex_region_var2 = self.for_edges_that_intersect2(Polygon(convex_region_var2),
                                                                            polygon)

                        li = self.extend_pol(angle, convex_region, polygon)

                        cop_li = copy.deepcopy(polygon)

                        cop_li.set_coordinates(li)

                        list_of_new_region2 = self.for_edges_that_intersect2(Polygon(convex_region_less_detailed),
                                                                             cop_li)
                        check_ang = self.calculate_angle_in_degrees(middle_point, t_p)
                        check_ang = check_ang % 360

                        if current_angle < check_ang:
                            current_angle = check_ang
                            convex_region = list_of_new_region

                        else:
                            convex_region = list_of_new_region2
                            current_angle = check_ang

                        convex_region_less_detailed = list_of_new_region2
                        # middle_point = self.calculate_centroid(convex_region)




        end_time = time.time()
        elapsed_time = end_time - start_time
        print(elapsed_time)
        print("num of polygons", len(another_list), "out of", len(self.item_instances), "time", elapsed_time, "value",
              value)

        draw_instance = Draw(self.container_instance, another_list, (1, 1), (1, 1), (1, 1), (1, 1), None,
                             None,
                             None, None)
        draw_instance.plot()










