import matplotlib.pyplot as plt

class Draw:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def plot(self):
        container_x = self.container_instance.x_coords
        container_y = self.container_instance.y_coords

        # Extract item coordinates
        item_x_coords = [item_instance.x_coords for item_instance in self.item_instances]
        item_y_coords = [item_instance.y_coords for item_instance in self.item_instances]

        # Plot container
        plt.plot(container_x + [container_x[0]], container_y + [container_y[0]], 'b-', label='Container')

        # Plot items
        for x_coords, y_coords in zip(item_x_coords, item_y_coords):
            plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], 'r-', label='Item')

        # Set plot labels and legend
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Visual Representation of Container and Items')
        plt.legend()

        # Show the plot
        plt.show()
