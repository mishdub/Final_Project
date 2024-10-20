import json
from item import Item
from container import Container
from algorithm import Algorithm
from draw import Draw



def load_json(filename):
    with open(filename, 'r') as file:
        return json.loads(file.read())


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

    Algo_instance = Algorithm(container_instance, item_instances)
    result_list = Algo_instance.run_algorithm()

    draw_instance = Draw(container_instance, result_list)

    draw_instance.plot()  # This creates the plot

if __name__ == "__main__":
    main()
