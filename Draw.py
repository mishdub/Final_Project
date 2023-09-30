import matplotlib.pyplot as plt
import random

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

        # Add grid
        plt.grid(True)


       # grid_coordinates = self.container_instance.create_grid_coordinates(1000, None, None)

        it = None
        for index, item in enumerate(self.item_instances):
            it = item
            break
        li = [it]
        i = 0
        save = None
        for index, item in enumerate(self.item_instances):
            save = item
            i = i+1
            if i == 2:
                break

  
       # grid_coordinates = self.container_instance.generate_feasible_point(save,li)
       # print(grid_coordinates)
      #  r = self.container_instance.create_rectangles(grid_coordinates, 500, li)
      #  print(r)

       # rec, temp = self.container_instance.group_grid_coordinates_into_rectangles(2311, [])
       # print(rec)
       # print(temp)
       
       



        #Plot grid coordinates
        """
        for x, y in grid_coordinates:
           plt.text(x, y, f'({x}, {y})', ha='center', va='center')
        """

        


        """
        # Adding grid of points
        grid_step = 1000  # Adjust this to change the spacing between grid points
        for x in range(int(min(container_x)), int(max(container_x)) + 1, int(grid_step)):
            for y in range(int(min(container_y)), int(max(container_y)) + 1, int(grid_step)):
                plt.plot(x, y, 'ko')  # 'ko' represents black circles (points)
        """


        # Show the plot
        plt.show()

