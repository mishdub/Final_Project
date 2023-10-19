import random
import copy
from Draw import Draw
from shapely.geometry import Point, Polygon
from shapely.geometry import LineString
import time


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

    def plot3(self, item, items):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")

        max_attempts = 100
        container_polygon = Polygon(self.container_instance.coordinates)

        for _ in range(max_attempts):
            # Generate random x and y coordinates within the range of the container
            x = random.uniform(min(self.container_instance.x_coords), max(self.container_instance.x_coords))
            y = random.uniform(min(self.container_instance.y_coords), max(self.container_instance.y_coords))
            item_coords = item.move_item_value(x, y)
            item_polygon = Polygon(item_coords)

            if item_polygon.within(container_polygon):
                if not items:
                    item.move_item(x, y)
                    return x, y, True
                else:
                    # Check for collisions with other items
                    collision_flag = False
                    for other_item in items:
                        if item_polygon.intersects(Polygon(other_item.coordinates)):
                            collision_flag = True
                            break

                    if not collision_flag:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False

    def plot4(self, item, items):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")
        grid_cols = 100
        grid_rows = 100
        container_x_min, container_x_max = min(self.container_instance.x_coords), max(self.container_instance.x_coords)
        container_y_min, container_y_max = min(self.container_instance.y_coords), max(self.container_instance.y_coords)

        cell_width = (container_x_max - container_x_min) / grid_cols
        cell_height = (container_y_max - container_y_min) / grid_rows

        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate cell boundaries
                cell_x_min = container_x_min + col * cell_width
                cell_x_max = container_x_min + (col + 1) * cell_width
                cell_y_min = container_y_min + row * cell_height
                cell_y_max = container_y_min + (row + 1) * cell_height

                # Generate a random point within the cell
                x = random.uniform(cell_x_min, cell_x_max)
                y = random.uniform(cell_y_min, cell_y_max)

                item_coords = item.move_item_value(x, y)

                # Check if the item is inside the container and does not overlap with other items
                item_polygon = Polygon(item_coords)
                container_polygon = Polygon(self.container_instance.coordinates)

                if item_polygon.within(container_polygon):
                    collision_flag = False
                    for other_item in items:
                        if item_polygon.intersects(Polygon(other_item.coordinates)):
                            collision_flag = True
                            break

                    if not collision_flag:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False

    def plot5(self, item, items):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")

        container_polygon = Polygon(self.container_instance.coordinates)

        container_x_min, container_x_max = container_polygon.bounds[0], container_polygon.bounds[2]
        container_y_min, container_y_max = container_polygon.bounds[1], container_polygon.bounds[3]

        # Calculate an initial cell size based on the square root of the container's area
        container_area = container_polygon.area
        initial_cell_size = (container_area) ** 0.5

        # Calculate grid rows and columns based on the initial cell size
        grid_cols = int((container_x_max - container_x_min) / initial_cell_size)
        grid_rows = int((container_y_max - container_y_min) / initial_cell_size)

        for row in range(grid_rows):
            for col in range(grid_cols):
                cell_x_min = container_x_min + col * initial_cell_size
                cell_x_max = container_x_min + (col + 1) * initial_cell_size
                cell_y_min = container_y_min + row * initial_cell_size
                cell_y_max = container_y_min + (row + 1) * initial_cell_size

                x = random.uniform(cell_x_min, cell_x_max)
                y = random.uniform(cell_y_min, cell_y_max)

                item_coords = item.move_item_value(x, y)
                item_polygon = Polygon(item_coords)

                if item_polygon.within(container_polygon):
                    collision_flag = any(
                        item_polygon.intersects(Polygon(other_item.coordinates)) for other_item in items)
                    if not collision_flag:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False

    def plot6(self, item, items):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")

        container_x_min, container_x_max = min(self.container_instance.x_coords), max(self.container_instance.x_coords)
        container_y_min, container_y_max = min(self.container_instance.y_coords), max(self.container_instance.y_coords)
        container_width = container_x_max - container_x_min
        container_height = container_y_max - container_y_min
        average_item_size = item.calculate_total_dimensions()
        grid_cols = int(container_width / average_item_size)
        grid_rows = int(container_height / average_item_size)
        cell_width = (container_x_max - container_x_min) / grid_cols
        cell_height = (container_y_max - container_y_min) / grid_rows

        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate cell boundaries
                cell_x_min = container_x_min + col * cell_width
                cell_x_max = container_x_min + (col + 1) * cell_width
                cell_y_min = container_y_min + row * cell_height
                cell_y_max = container_y_min + (row + 1) * cell_height

                # Generate a random point within the cell
                x =cell_x_min
                y =cell_y_min

                item_coords = item.move_item_value(x, y)

                # Check if the item is inside the container and does not overlap with other items
                item_polygon = Polygon(item_coords)
                container_polygon = Polygon(self.container_instance.coordinates)

                if item_polygon.within(container_polygon):
                    collision_flag = False
                    for other_item in items:
                        if item_polygon.intersects(Polygon(other_item.coordinates)):
                            collision_flag = True
                            break

                    if not collision_flag:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False

    def plot7(self, item, items):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")
        grid_cols = 100
        grid_rows = 100
        container_x_min, container_x_max = min(self.container_instance.x_coords), max(self.container_instance.x_coords)
        container_y_min, container_y_max = min(self.container_instance.y_coords), max(self.container_instance.y_coords)
        if items:
            x_start, y_start = items[-1].find_point_with_max_x()
        else:
            x_start = container_x_min
            y_start = container_y_min


        cell_width = (container_x_max - x_start) / grid_cols
        cell_height = (container_y_max - y_start) / grid_rows

        # Compute the container polygon once
        container_polygon = Polygon(self.container_instance.coordinates)

        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate cell boundaries
                cell_x_min = x_start + col * cell_width
                cell_x_max = x_start + (col + 1) * cell_width
                cell_y_min = y_start + row * cell_height
                cell_y_max = y_start + (row + 1) * cell_height

                x = (cell_x_min + cell_x_max) / 2
                y = (cell_y_min + cell_y_max) / 2
                item_coords = item.move_item_value(x, y)

                # Check if the item is inside the container and does not overlap with other items
                item_polygon = Polygon(item_coords)
                container_polygon = Polygon(self.container_instance.coordinates)

                if item_polygon.within(container_polygon):
                    collision_flag = False
                    for other_item in items:
                        if item_polygon.intersects(Polygon(other_item.coordinates)):
                            collision_flag = True
                            break

                    if not collision_flag:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False

    def plot8(self, item, items, container_x_min,container_x_max, container_y_min,container_y_max):
        if not self.container_instance.coordinates:
            raise ValueError("Container is empty")
        grid_cols = 100
        grid_rows = 100

        cell_width = (container_x_max - container_x_min) / grid_cols
        cell_height = (container_y_max - container_y_min) / grid_rows

        for row in range(grid_rows):
            for col in range(grid_cols):
                # Calculate cell boundaries
                cell_x_min = container_x_min + col * cell_width
                cell_x_max = container_x_min + (col + 1) * cell_width
                cell_y_min = container_y_min + row * cell_height
                cell_y_max = container_y_min + (row + 1) * cell_height

                # Generate a random point within the cell
                x = random.uniform(cell_x_min, cell_x_max)
                y = random.uniform(cell_y_min, cell_y_max)


                item_coords = item.move_item_value(x, y)

                # Check if the item is inside the container and does not overlap with other items
                item_polygon = Polygon(item_coords)
                container_polygon = Polygon(self.container_instance.coordinates)

                if item_polygon.within(container_polygon):
                    collision_flag = False
                    for other_item in items:
                        if item_polygon.intersects(Polygon(other_item.coordinates)):
                            collision_flag = True
                            break

                    if not collision_flag:
                        item.move_item(x, y)
                        return x, y, True

        return 0, 0, False

    def Ascending_order_by_item_size(self):
        sorted_items = sorted(self.item_instances, key=lambda item: item.calculate_total_dimensions())
        list = []
        value = 0
        i = 0

        start_time = time.time()
        for index, item in enumerate(sorted_items):
            x, y, flag = self.plot3(item, list)
            if flag is not False:
                list.append(item)
                value = value + item.value
            elif flag is False:
                continue
            i = i + 1
            print(i)





        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time)
        print("Items in total:", len(sorted_items), "Items picked:", len(list), "value:", value)
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
