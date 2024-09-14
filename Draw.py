import matplotlib.pyplot as plt
import numpy as np
import colorsys

class Draw:
    def __init__(self, container_instance, item_instances, p1, p2, t1, t2, exterior_coords_list, angle1, angle2, list_of_lines):
        self.container_instance = container_instance
        self.item_instances = item_instances
        self.p1 = p1
        self.p2 = p2
        self.t1 = t1
        self.t2 = t2
        self.exterior_coords_list = exterior_coords_list
        self.angle1 = angle1
        self.angle2 = angle2
        self.list_of_lines = list_of_lines

    def generate_random_color(self):
        """Generate a random vibrant color."""
        # Generate random hue, full saturation (1), and high brightness (0.8 to 1)
        h = np.random.rand()
        s = 1  # Full saturation for vibrant color
        v = np.random.uniform(0.8, 1)  # High brightness
        color = colorsys.hsv_to_rgb(h, s, v)
        return color

    def plot(self):
        container_x = self.container_instance.x_coordinates
        container_y = self.container_instance.y_coordinates

        # Extract item coordinates
        item_x_coords = [item_instance.x_coordinates for item_instance in self.item_instances]
        item_y_coords = [item_instance.y_coordinates for item_instance in self.item_instances]

        # Set background color to white
        plt.figure(facecolor='white')
        ax = plt.gca()
        ax.set_facecolor('white')

        # Plot container's interior filled with black
        plt.fill(container_x + [container_x[0]], container_y + [container_y[0]], color='black', label='Container')

        # Plot items with vibrant random colors inside the container
        for x_coords, y_coords in zip(item_x_coords, item_y_coords):
            # Generate a random vibrant color
            color = self.generate_random_color()
            plt.fill(x_coords + [x_coords[0]], y_coords + [y_coords[0]], color=color, alpha=1)  # Fully opaque
            plt.plot(x_coords + [x_coords[0]], y_coords + [y_coords[0]], color=color, linestyle='-', linewidth=1)

        # Set plot labels and legend
        #plt.legend()
        plt.axis('off')


        # Show the plot
        plt.show()
