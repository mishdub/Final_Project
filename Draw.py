import matplotlib.pyplot as plt
import numpy as np

import random

class Draw:
    def __init__(self, container_instance, item_instances, p1, p2, t1, t2,exterior_coords_list,angle1,angle2):
        self.container_instance = container_instance
        self.item_instances = item_instances
        self.p1 = p1
        self.p2 = p2
        self.t1 = t1
        self.t2 = t2
        self.exterior_coords_list = exterior_coords_list
        self.angle1 = angle1
        self.angle2 = angle2

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

        x1,y1 = self.p1
        x2,y2 = self.p2

        plt.plot([x1, x2], [y1, y2], 'g-', label='Line1')

        xk1, yk1 = self.t1
        xk2, yk2 = self.t2

        plt.plot([xk1, xk2], [yk1, yk2], 'b-', label='Line2')


        # Set plot labels and legend
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.title('Visual Representation of Container and Items')
        plt.legend()

        # Add grid
        plt.grid(True)
        # Calculate intersection points with plot boundaries
        if self.angle1 is not None:
            x1, y1 = self.p1
            m1 = np.tan(np.deg2rad(self.angle1))
            b1 = y1 - m1 * x1
            x_boundary = np.array([min(container_x), max(container_x)])
            y_boundary = m1 * x_boundary + b1
            plt.plot(x_boundary, y_boundary, 'g-', label=f'Angle 1: {self.angle1} degrees')

        if self.angle2 is not None:
            x2, y2 = self.t1
            m2 = np.tan(np.deg2rad(self.angle2))
            b2 = y2 - m2 * x2
            x_boundary = np.array([min(container_x), max(container_x)])
            y_boundary = m2 * x_boundary + b2
            plt.plot(x_boundary, y_boundary, 'm-', label=f'Angle 2: {self.angle2} degrees')
        # Plot custom point if provided
        """
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
            # Draw lines representing angles
        """


        # Show the plot
        plt.show()


