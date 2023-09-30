import json
from Item import Item
from Container import Container
from Draw import Draw
import random
from shapely.geometry import Polygon,MultiPolygon,LineString,Point,MultiPoint


from shapely.validation import explain_validity

from shapely import normalize, Polygon, coverage_union, overlaps
from shapely.ops import unary_union












def load_json(filename):
    with open(filename, 'r') as file:
        return json.loads(file.read())


def print_container_info(container_instance):
    print("\nContainer:")
    for i, (x, y) in enumerate(zip(container_instance.x_coords, container_instance.y_coords), start=1):
        print(f"  Point {i}: ({x}, {y})")
    slope_dict = container_instance.slope_list()
    for edge, slope in slope_dict.items():
        print(f"Edge: {edge}, Slope: {slope}")
    equations = container_instance.calculate_line_equations()
    for edge, equation in equations.items():
        print(f"{edge}: {equation}")
    middle_projections =container_instance.determine_all_middle_projection_directions()
    for i, direction in enumerate(middle_projections):
        print(f"Edge {i + 1} middle point projection: {direction}")

def print_item_info(item_instances):
    for item_index, item_instance in enumerate(item_instances):
        print("\nItem", item_index + 1, ":")
        print("  Quantity:", item_instance.quantity)
        print("  Value:", item_instance.value)
        print("  Coordinates (x, y):")
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")
        print("  Slopes:")
        slopes = item_instance.slope_list()
        for edge, slope in slopes.items():
            print(f"    Edge {edge}: Slope {slope}")
        equations = item_instance.calculate_line_equations()
        for edge, equation in equations.items():
            print(f"{edge}: {equation}")

def test(item_instances, container_instance):
    edge_start, edge_end, middle_point = random.choice(container_instance.calculate_middle_points())
    proj = container_instance.determine_projection_direction_from_middle(middle_point)
    middle_x, middle_y = middle_point

    for item_index, item_instance in enumerate(item_instances):
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")
        if proj == "left":
            x, y = item_instance.find_point_with_max_x()
            print(proj, x, y, middle_x, middle_y)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y, middle_x, middle_y)
        elif proj == "right":
            x, y = item_instance.find_point_with_min_x()
            print(proj, x, y, middle_x, middle_x)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y, middle_x, middle_y)
        elif proj == "up":
            x, y = item_instance.find_point_with_min_y()
            print(proj, x, y, middle_x, middle_y)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y, middle_x, middle_y)
        elif proj == "down":
            x, y = item_instance.find_point_with_max_y()
            print(proj, x, y,middle_x, middle_y)
            item_instance.move_item(x, y, middle_x, middle_y)
            print(proj, x, y,middle_x, middle_y)
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")
        break


def item_with_max_dimensions(item_list):
        """
        Find and return the item with the maximum dimensions (either width or height) from a list of items.

        Args:
        - item_list: List of Item objects.

        Returns:
        - The Item object with the maximum dimensions.
        """
        max_item = None
        max_dimension = 0

        for item in item_list:
            max_dimension_for_item = item.calculate_total_dimensions()

            if max_dimension_for_item > max_dimension:
                max_item = item
                max_dimension = max_dimension_for_item

        return max_item, max_dimension

def item_with_min_dimensions(item_list):
        """
        Find and return the item with the minimum dimensions (either width or height) from a list of items.

        Args:
        - item_list: List of Item objects.

        Returns:
        - The Item object with the minimum dimensions.
        """
        min_item = None
        min_dimension = float('inf')

        for item in item_list:
            min_dimension_for_item = item.calculate_total_dimensions()

            if min_dimension_for_item < min_dimension:
                min_item = item
                min_dimension = min_dimension_for_item

        return min_item, min_dimension

def find_center_point(point1, point2, point3, point4):
    """
    Find the center point of a rectangle given its four corner points.

    Args:
    - point1, point2, point3, point4: Tuple (x, y) representing the four corner points of the rectangle.

    Returns:
    - The center point of the rectangle as a tuple (x, y).
    """
    x_values = [point1[0], point2[0], point3[0], point4[0]]
    y_values = [point1[1], point2[1], point3[1], point4[1]]

    center_x = sum(x_values) / 4
    center_y = sum(y_values) / 4

    return (center_x, center_y)


def check(point):
    """
           Check if a point is inside the item.

           Args:
           - point: (x, y) tuple representing the point to check.

           Returns:
           - True if the point is inside the container, False otherwise.
           """
    x, y = point
    list1= [(1620.6526169403587, 4094.7676954873227), (1620.6526169403587, 3192.7676954873227), (3965.6526169403587, 3192.7676954873227), (3965.6526169403587, 4094.7676954873227)]
    list2= [(479.3810687845348, 2069.595611747717), (2790.381068784535, 2069.595611747717), (2790.381068784535, 3088.595611747717), (1445.3810687845348, 3088.595611747717), (1445.3810687845348, 3723.595611747717), (479.3810687845348, 3723.595611747717)]
    n = len(list2)
    inside = False

    for i in range(n):
        x1, y1 = list2[i]
        x2, y2 = list2[(i + 1) % n]

        if (y1 < y <= y2 or y2 < y <= y1) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside

    return inside

def main():
    filename = input("Enter the filename containing JSON data: ")
    data = load_json(filename)

    container_instance = Container(data["container"]["x"], data["container"]["y"])

    item_instances = []
    for item_data in data["items"]:
        quantity = item_data["quantity"]
        value = item_data["value"]
        x_coords = item_data["x"]
        y_coords = item_data["y"]
        for _ in range(quantity):
            item_instance = Item(quantity, value, x_coords, y_coords)
            item_instances.append(item_instance)

   # print_container_info(container_instance)
    #print_item_info(item_instances)

    #test(item_instances, container_instance)

   # list = [[(0, 0), (1000, 0), (1000, 1000), (0, 1000)]]
    #grid_coordinates = container_instance.create_grid_coordinates(1000, list)

    #rectangles = container_instance.group_grid_coordinates_into_rectangles(1000)
    """
    low to high sort of items by their dimention 
    in the order of the sort we generate points by that dimention and place the item in one of the rectangles
    make sure to check that their exist such rectangles
    we put that item in a list of points to avoid in the next iteration
    """
    """
    # Sort the items by their total dimensions in ascending order
    sorted_items = sorted(item_instances, key=lambda item: item.calculate_total_dimensions())

    # Print the sorted items
    
    list = []
    i = 0
    for index, item in enumerate(sorted_items):
        dimen = item.calculate_total_dimensions()
        grid_coordinates = container_instance.create_grid_coordinates1(dimen, list)
        print(grid_coordinates)
        rec = container_instance.create_rectangles(grid_coordinates, dimen, list)
        for a, b, c, d in rec:
            if len(rec) > 0:
                xc, yc = find_center_point(a, b, c, d)
                item.move_item(xc, yc)
                break
            break
        if not len(rec) > 0:
            break
        if item not in list:
           list.append(item)
        i = i+1
        if i > 3:
            break
            
        """
    """
    sorted_items = sorted(item_instances, key=lambda item: item.calculate_total_dimensions())
    list = []
    i = 0
    value = 0
    Algo_instance = Algo(container_instance, item_instances)

    for index, item in enumerate(sorted_items):
        x, y, flag = Algo_instance.plot(item, list)
        if flag is not False:
            list.append(item)
            value = value+item.value
        elif flag is False:
            break
        i = i+1
        print(i)
        if i == 10:
            break
    print("value", value)

    # Print the generated grid coordinates

    draw_instance = Draw(container_instance,list)
    draw_instance.plot()
    """
   # Algo_instance = Algo(container_instance, item_instances)

    #val = Algo_instance.Ascending_order_by_item_size()
    i = 0
    b = None
    firstitem = None
    firstitemcoords = None
    sectitem = None

    li = None
    listo = []
    sorted_items = sorted(item_instances, key=lambda item: item.calculate_total_dimensions(), reverse=True)

    for index, item in enumerate(sorted_items):
        item.box()
        if i == 0:
            x, y = container_instance.calculate_centroid()
            item.move_item(x, y)
            firstitem = item.coordinates
            firstitemcoords = item.coordinates
            listo.append(item)

        flag = False
        string = "None"
        if i >= 1:
            for x, y in firstitem:
                coords = item.move_top_left_value(x, y)
                poly1 = Polygon(firstitemcoords)
                poly2 = Polygon(coords)
                poly3 =  Polygon(container_instance.coordinates)

                if poly2.touches(poly2) and not poly1.crosses(poly2) and not poly1.contains(poly2):
                    item.move_top_left(x, y)
                    flag = True
                    string = "top_left"
                    print(i,string,(not overlaps(poly1, poly2)))
                    break
                coords = item.move_bottom_left_value(x, y)
                poly2 = Polygon(coords)
                if poly1.intersects(poly2) and not poly1.crosses(poly2) and not poly1.contains(poly2):
                    item.move_bottom_left(x, y)
                    flag = True
                    string = "bottom_left"
                    print(i,string,(not overlaps(poly1, poly2)))
                    break
                coords = item.move_bottom_right_value(x, y)
                poly2 = Polygon(coords)
                if poly1.intersects(poly2) and not poly1.crosses(poly2) and not poly1.contains(poly2):
                    item.move_bottom_right(x, y)
                    flag = True
                    string = "bottom_right"

                    print(poly1)
                    print(poly2)
                    print(i,string,(not overlaps(poly1, poly2)))
                    break
                coords = item.move_top_right_value(x, y)
                poly2 = Polygon(coords)
                if poly1.intersects(poly2) and not poly1.crosses(poly2) and not poly1.contains(poly2):
                    item.move_top_right(x, y)
                    flag = True
                    string = "top_right"
                    print(i,string,(not overlaps(poly1, poly2)))
                    break
        if flag:

            listo.append(item)
            pol1 = Polygon(firstitemcoords)
            pol2 = Polygon(item.coordinates)


            pol = [pol1, pol2]
            vertices1 = set(list(pol1.exterior.coords))
            vertices2 = set(list(pol2.exterior.coords))
            corner_coords1 = vertices1 - vertices2
            corner_coords2 = vertices2 - vertices1
            m = list(corner_coords1.union(corner_coords2))
            print(m)

            """
            mergedPolys = unary_union(pol)

            exterior_coords_list = []

            # Check if mergedPolys is a MultiPolygon
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

            corner_coords = [coord for coord in exterior_coords_list if exterior_coords_list.count(coord) == 1]
            """
            firstitemcoords = m

            array = []
            for p in m:
                min = 0
                for q in container_instance.get_edge_lines():
                    line = LineString(q)
                    point = Point(p)
                    dis = point.distance(line)
                    #print("i:",i,"point:",p,"line:",q,"dist:",dis)
                    if dis > min:
                        min = dis
                array.append((p, min))
            sorted_array = sorted(array, key=lambda x: x[1], reverse=True)
            print(sorted_array)
            sorted_points = [point for point, _ in sorted_array]
            firstitem = m
        i = i+1
        if i == 6:
            break



    """
    poly1 = Polygon(firstitem.coordinates)
    poly2 = Polygon(sectitem.coordinates)
    print(poly1.intersects(poly2))
    print(overlaps(poly1, poly2))
    print(poly1.intersection(poly2))

    pol = [poly1, poly2]
    mergedPolys = unary_union(pol)

    x_coords = []
    y_coords = []
    for x, y in mergedPolys.exterior.coords:
        print("x:", x,"y",y)
        x_coords.append(x)
        y_coords.append(y)

    ite = Item(0, 0, x_coords, y_coords)
    l = []
    l.append(ite)
    """


    draw_instance = Draw(container_instance, listo)

    draw_instance.plot()




if __name__ == "__main__":
    main()
