from Item import Item


class Algo:
    def __init__(self, data):
        self.data = data

    def calculate_grid_spacing(self, max_item_width, max_item_height, num_items):
        container_width = max(self.data["container"]["x"])
        container_height = max(self.data["container"]["y"])

        max_item_dimension = max(max_item_width, max_item_height)

        # Determine the number of grid cells in each dimension
        num_cells_x = int(container_width / max_item_dimension)
        num_cells_y = int(container_height / max_item_dimension)

        # Calculate the grid spacing
        grid_spacing_x = container_width / num_cells_x
        grid_spacing_y = container_height / num_cells_y

        return grid_spacing_x, grid_spacing_y

    def create_item_instances_grid_based(self):
        item_instances = []
        for item_data in self.data["items"]:
            quantity = item_data["quantity"]
            value = item_data["value"]
            x_coords = item_data["x"]
            y_coords = item_data["y"]

            max_item_width = max(x_coords) - min(x_coords)
            max_item_height = max(y_coords) - min(y_coords)

            grid_spacing_x, grid_spacing_y = self.calculate_grid_spacing(max_item_width, max_item_height, quantity)

            # Create item instances using the grid-based approach
            for i in range(quantity):
                x = min(x_coords) + (i + 0.5) * grid_spacing_x
                y = min(y_coords) + (i + 0.5) * grid_spacing_y
                item_instance = Item(quantity, value, [x], [y])
                item_instances.append(item_instance)
        return item_instances
