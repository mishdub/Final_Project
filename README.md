# Polygon Packing Problem

This project focuses on the [Polygon Packing problem](https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2024/#problem-description) from the CG 2024 competition. By exploring different methods, we aim to provide useful insights and contribute to better solutions for this optimization challenge.

## Problem Description

The input to the problem consists of:

- **A container**: A convex polygon that serves as the bounding region where items must be placed.
- **A set of items**: Each item is also a polygon with a fixed shape, orientation, and an associated integer value.
  
## Key Constraints

- **No Overlap**: The polygons must be arranged without overlapping or intersecting.
- **No Rotation, Tilting, or Dilation**: The items must maintain their original orientation, shape, and scale during placement.

## Output

- **Maximum Total Value**: The maximum total value that can be obtained from the subset of polygons packed into the container.
- **Subset of Polygons**: The subset of polygons that achieve this maximum value, including their placement coordinates within the container.

