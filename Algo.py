import random
import copy
from Draw import Draw
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString

class Algo:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances
        self.generated_points = []

    def generate_random_point(self, max_attempts):
        for _ in range(max_attempts):
            # Generate random x and y coordinates within the range of the container
            x = random.uniform(min(self.container_instance.x_coords), max(self.container_instance.x_coords))
            y = random.uniform(min(self.container_instance.y_coords), max(self.container_instance.y_coords))

            # Check if the generated point is not in the list of generated points
            if (x, y) not in self.generated_points:
                self.generated_points.append((x, y))
                return x, y

        # If all attempts are exhausted, you can handle this case accordingly
        return None

    def plot(self, item, items):
        # Ensure there are valid coordinates in the container
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")
        max_attempts = 10  # Maximum number of attempts to generate a feasible point
        for _ in range(max_attempts):
            # Generate random x and y coordinates within the range of the container
            x = random.uniform(min(self.container_instance.x_coords), max(self.container_instance.x_coords))
            y = random.uniform(min(self.container_instance.y_coords), max(self.container_instance.y_coords))

            deep_copy_item = copy.deepcopy(item)
            deep_copy_item.move_item(x, y)

            # Check if the item is inside the container and does not overlap with other items
            # Check if the deep copied item is inside the container
            flag1 = True
            for (x_p, y_p) in deep_copy_item.coordinates:
                flag1 = self.container_instance.point_inside_container((x_p, y_p))
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
                    # if flag1 and not found_flag4:
                    item.move_item(x, y)
                    return x, y, True
            elif not items:
                if flag1:
                    item.move_item(x, y)
                    return x, y, True

        return 0, 0, False

    def plot2(self, item, items):
        # Ensure there are valid coordinates in the container
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")
        max_attempts = 10 # Maximum number of attempts to generate a feasible point
        for _ in range(max_attempts):
            # Generate random x and y coordinates within the range of the container
            #x = random.uniform(min(self.container_instance.x_coords), max(self.container_instance.x_coords))
            #y = random.uniform(min(self.container_instance.y_coords), max(self.container_instance.y_coords))
            m_a = 10
            point = self.generate_random_point(m_a)
            if point is not None:
                x, y = point
                deep_copy_item = copy.deepcopy(item)
                deep_copy_item.move_item(x, y)

                # Check if the item is inside the container and does not overlap with other items
                # Check if the deep copied item is inside the container
                flag1 = True
                for (x_p, y_p) in deep_copy_item.coordinates:
                    flag1 = self.container_instance.point_inside_container((x_p, y_p))
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
                        # if flag1 and not found_flag4:
                        item.move_item(x, y)
                        return x, y, True
                elif not items:
                    if flag1:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False


    def Ascending_order_by_item_size(self):
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions())
        list = []
        value = 0
        i = 0
        for index, item in enumerate(sorted_items):
            x, y, flag = self.plot(item, list)
            if flag is not False:
                list.append(item)
                value = value + item.value
            elif flag is False:
                continue
            i = i + 1
            print(i)
            if i == 200:
                break

        print("items num in total:",len(sorted_items), "items num picked:",len(list), "value:", value)
        draw_instance = Draw(self.container_instance, list)
        draw_instance.plot()

    def Descending_order_by_item_size(self):
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=True)
        list = []
        value = 0
        draw_instance = Draw(self.container_instance, list)
        for index, item in enumerate(sorted_items):
            x, y, flag = self.plot(item, list)
            if flag is not False:
                list.append(item)
                value = value + item.value
            elif flag is False:
                break

        print(value)

    def Ascending_order_by_item_value(self):
        sorted_items = sorted(self.item_instances, key=lambda item: item.value)
        list = []
        value = 0
        for index, item in enumerate(sorted_items):
            x, y, flag = self.plot(item, list)
            if flag is not False:
                list.append(item)
                value = value + item.value
            elif flag is False:
                break

        print(value)
        draw_instance = Draw(self.container_instance, list)
        draw_instance.plot()

    def Descending__order_by_item_value(self):
        sorted_items = sorted(self.item_instances, key=lambda item: item.value, reverse=True)
        list = []
        value = 0
        for index, item in enumerate(sorted_items):
            x, y, flag = self.plot(item, list)
            if flag is not False:
                list.append(item)
                value = value + item.value
            elif flag is False:
                break

        print(value)
        draw_instance = Draw(self.container_instance, list)
        draw_instance.plot()
