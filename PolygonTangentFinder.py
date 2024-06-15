from functools import cmp_to_key


class PolygonTangentFinder:
    # program to find upper tangent of two polygons.
    def __init__(self):
        # This stores the centre of polygon, used in compare function
        self.mid = [0, 0]

    # determines the quadrant of a point
    # (used in compare())
    def quad(self, p):
        if p[0] >= 0 and p[1] >= 0:
            return 1
        if p[0] <= 0 and p[1] >= 0:
            return 2
        if p[0] <= 0 and p[1] <= 0:
            return 3
        return 4

    # Checks whether the line is crossing the polygon
    def orientation(self, a, b, c):
        res = (b[1] - a[1]) * (c[0] - b[0]) - (c[1] - b[1]) * (b[0] - a[0])
        if res == 0:
            return 0
        if res > 0:
            return 1
        return -1

    # compare function for sorting
    def compare(self, p1, q1):
        p = [p1[0] - self.mid[0], p1[1] - self.mid[1]]
        q = [q1[0] - self.mid[0], q1[1] - self.mid[1]]
        one = self.quad(p)
        two = self.quad(q)

        if one != two:
            if one < two:
                return -1
            return 1
        if p[1] * q[0] < q[1] * p[0]:
            return -1
        return 1

    # Finds upper tangent of two polygons 'a' and 'b'
    # represented as two vectors.
    def findUpperTangent(self, a, b):
        n1, n2 = len(a), len(b)
        # Start with the topmost points based on y-coordinate
        ia = max(range(n1), key=lambda i: a[i][1])
        ib = max(range(n2), key=lambda i: b[i][1])

        done = False
        while not done:
            done = True
            # Rotate `ib` counterclockwise to find upper tangent on b
            while self.orientation(a[ia], b[ib], b[(ib + 1) % n2]) <= 0:
                ib = (ib + 1) % n2
                done = False

            # Rotate `ia` clockwise to find upper tangent on a (opposite direction because of `a`'s arrangement)
            while self.orientation(b[ib], a[ia], a[(ia - 1 + n1) % n1]) >= 0:
                ia = (ia - 1 + n1) % n1
                done = False

        return [(a[ia][0], a[ia][1]), (b[ib][0], b[ib][1])]
    # Finds lower tangent of two polygons 'a' and 'b'
    # represented as two vectors.
    def findLowerTangent(self, a, b):
        n1, n2 = len(a), len(b)

        # Start with the leftmost point of a and the rightmost point of b
        ia = min(range(n1), key=lambda i: a[i][0])
        ib = max(range(n2), key=lambda i: b[i][0])

        done = False
        while not done:
            done = True

            # Move point on a clockwise to find the real lower tangent
            while self.orientation(b[ib], a[ia], a[(ia + 1) % n1]) <= 0:
                ia = (ia + 1) % n1
                done = False

            # Move point on b counterclockwise to find the real lower tangent
            while self.orientation(a[ia], b[ib], b[(ib - 1 + n2) % n2]) >= 0:
                ib = (ib - 1 + n2) % n2
                done = False

        return [(a[ia][0], a[ia][1]), (b[ib][0], b[ib][1])]

    # Example usage in main block to ensure it's correct
if __name__ == '__main__':
    finder = PolygonTangentFinder()
    a = [[733, 719], [705, 734], [733, 716]]
    b = [[1411, 719], [1383, 734], [1411, 716]]
    print("Upper Tangent:")
    print(finder.findUpperTangent(a, b))
    print("Lower Tangent:")
    print(finder.findLowerTangent(a, b))





