from functools import cmp_to_key


class PointTangentFinder:
    # Program to find tangents from a point to a polygon.
    def __init__(self):
        pass

    # Checks whether the line is crossing the polygon
    def orientation(self, a, b, c):
        res = (b[1] - a[1]) * (c[0] - b[0]) - (c[1] - b[1]) * (b[0] - a[0])
        if res == 0:
            return 0
        if res > 0:
            return 1
        return -1

    # Finds upper tangent from a point to a polygon
    def findUpperTangent(self, point, polygon):
        n = len(polygon)
        upper_tangent = None
        max_slope = float('-inf')

        for i in range(n):
            slope = (polygon[i][1] - point[1]) / (polygon[i][0] - point[0]) if polygon[i][0] != point[0] else float('inf')
            if slope > max_slope:
                if self.orientation(point, polygon[i], polygon[(i + 1) % n]) < 0 and self.orientation(point, polygon[i], polygon[i - 1]) < 0:
                    upper_tangent = polygon[i]
                    max_slope = slope

        return upper_tangent

    # Finds lower tangent from a point to a polygon
    def findLowerTangent(self, point, polygon):
        n = len(polygon)
        lower_tangent = None
        min_slope = float('inf')

        for i in range(n):
            slope = (polygon[i][1] - point[1]) / (polygon[i][0] - point[0]) if polygon[i][0] != point[0] else float('-inf')
            if slope < min_slope:
                if self.orientation(point, polygon[i], polygon[(i + 1) % n]) > 0 and self.orientation(point, polygon[i], polygon[i - 1]) > 0:
                    lower_tangent = polygon[i]
                    min_slope = slope

        return lower_tangent

    # Example usage in main block to ensure it's correct
if __name__ == '__main__':
    finder = PointTangentFinder()
    point = (800, 720)  # Example point represented as a tuple
    polygon = [(733, 719), (705, 734), (733, 716)]  # Example polygon represented as a list of tuples
    print("Upper Tangent:")
    upper_tangent = finder.findUpperTangent(point, polygon)
    print(upper_tangent)
    print("Lower Tangent:")
    lower_tangent = finder.findLowerTangent(point, polygon)
    print(lower_tangent)
