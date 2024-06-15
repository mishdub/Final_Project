import matplotlib.pyplot as plt
import numpy as np


class Algo37:
    def plot_circles_tangents_and_touch_points(self,center1, radius1, center2, radius2):
        # Define the circles
        circle1 = plt.Circle(center1, radius1, color='blue', fill=False)
        circle2 = plt.Circle(center2, radius2, color='green', fill=False)

        # Calculate differences
        d_x = center2[0] - center1[0]
        d_y = center2[1] - center1[1]
        dist = np.hypot(d_x, d_y)
        r1, r2 = radius1, radius2

        # Set up plot
        fig, ax = plt.subplots()
        ax.add_artist(circle1)
        ax.add_artist(circle2)

        # Determine limits
        lim = max(dist + r1 + r2, r1 * 2, r2 * 2, 20)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect('equal', 'box')

        if dist == 0:
            # Circles are concentric
            return

        # Internal tangents
        if dist > r1 + r2:
            base_angle = np.arctan2(d_y, d_x)
            angle = np.arccos((r1 - r2) / dist)
            for sign in [-1, 1]:
                # Tangent points on circle 1
                x1 = np.cos(base_angle + angle * sign) * r1 + center1[0]
                y1 = np.sin(base_angle + angle * sign) * r1 + center1[1]

                # Tangent points on circle 2
                x2 = np.cos(base_angle + angle * sign) * r2 + center2[0]
                y2 = np.sin(base_angle + angle * sign) * r2 + center2[1]

                dx = np.cos(base_angle + angle * sign + np.pi / 2)
                dy = np.sin(base_angle + angle * sign + np.pi / 2)

                plt.plot([x1 - dx * 1000, x1 + dx * 1000], [y1 - dy * 1000, y1 + dy * 1000], 'r--')

                # Mark tangent points on circles
                plt.plot(x1, y1, 'ko')  # Black dot on circle 1
                plt.plot(x2, y2, 'ko')  # Black dot on circle 2

        plt.grid(True)
        plt.show()
