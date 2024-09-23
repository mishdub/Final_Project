import matplotlib.pyplot as plt
import numpy as np
import colorsys

class Draw:
    def __init__(self, container_instance, item_instances):
        self.container_instance = container_instance
        self.item_instances = item_instances

    def generate_random_color(self):
        """Generate a random vibrant color."""
        h = np.random.rand()
        s = 1  # Full saturation for vibrant color
        v = np.random.uniform(0.8, 1)  # High brightness
        color = colorsys.hsv_to_rgb(h, s, v)
        return color

    def plot(self):
        container_x = self.container_instance.x_coordinates
        container_y = self.container_instance.y_coordinates

        # Extract item coordinates
        item_x_coordinates = [item_instance.x_coordinates for item_instance in self.item_instances]
        item_y_coordinates = [item_instance.y_coordinates for item_instance in self.item_instances]

        # Set background color to white
        plt.figure(facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        # Plot container's interior filled with black
        plt.fill(container_x + [container_x[0]], container_y + [container_y[0]], color='black', label='Container')

        # Plot items with vibrant random colors inside the container
        for x_coordinates, y_coordinates in zip(item_x_coordinates, item_y_coordinates):
            color = self.generate_random_color()
            plt.fill(x_coordinates + [x_coordinates[0]], y_coordinates + [y_coordinates[0]], color=color, alpha=1)  # Fully opaque
            plt.plot(x_coordinates + [x_coordinates[0]], y_coordinates + [y_coordinates[0]], color=color, linestyle='-', linewidth=1)

        # Set plot labels and legend
        plt.axis('off')

        # Show the plot
        plt.show()
