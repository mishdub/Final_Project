import matplotlib.pyplot as plt
import random

class Draw:
    def __init__(self, container_instance, item_instances, p1, p2, t1, t2,exterior_coords_list):
        self.container_instance = container_instance
        self.item_instances = item_instances
        self.p1 = p1
        self.p2 = p2
        self.t1 = t1
        self.t2 = t2
        self.exterior_coords_list = exterior_coords_list

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

        # Add grid
        plt.grid(True)

        # Plot custom point if provided

        if self.exterior_coords_list:
            for (x, y) in self.exterior_coords_list:
                plt.plot(x, y, 'ro', label="my point")
        elif not self.exterior_coords_list:
            x, y = self.p1
            plt.plot(x, y, 'go', label="my point")

            x, y = self.p2
            plt.plot(x, y, 'go', label="my point")

            x, y = self.t1
            plt.plot(x, y, 'ro', label="my point")

            x, y = self.t2
            plt.plot(x, y, 'ro', label="my point")






        # Show the plot
        plt.show()


