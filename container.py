
class Container:
    def __init__(self, x_coordinates, y_coordinates):
        self.coordinates = list(zip(x_coordinates, y_coordinates))
        self.x_coordinates = x_coordinates
        self.y_coordinates = y_coordinates
        self.max_x = max(x_coordinates)
        self.max_y = max(y_coordinates)
        self.min_x = min(x_coordinates)
        self.min_y = min(y_coordinates)

    def get_largest_dimension(self):
        # Calculate the width of the object by subtracting the minimum x value from the maximum x value
        width = self.max_x - self.min_x

        # Calculate the height of the object by subtracting the minimum y value from the maximum y value
        height = self.max_y - self.min_y

        # Determine the largest dimension (either width or height) by using the max function
        total_dimensions = max(width, height)

        # Return the largest dimension
        return total_dimensions







