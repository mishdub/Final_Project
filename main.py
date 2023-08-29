import json
from Item import Item
from Container import Container
from Draw import Draw

def load_json(filename):
    with open(filename, 'r') as file:
        return json.loads(file.read())


def print_container_info(container_instance):
    print("\nContainer:")
    for i, (x, y) in enumerate(zip(container_instance.x_coords, container_instance.y_coords), start=1):
        print(f"  Point {i}: ({x}, {y})")


def print_item_info(item_instances):
    for item_index, item_instance in enumerate(item_instances):
        print("\nItem", item_index + 1, ":")
        print("  Quantity:", item_instance.quantity)
        print("  Value:", item_instance.value)
        print("  Coordinates (x, y):")
        for j, (x, y) in enumerate(zip(item_instance.x_coords, item_instance.y_coords), start=1):
            print(f"    Point {j}: ({x}, {y})")

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

    print_container_info(container_instance)
    print_item_info(item_instances)

    draw_instance = Draw(container_instance, item_instances)
    draw_instance.plot()


if __name__ == "__main__":
    main()
