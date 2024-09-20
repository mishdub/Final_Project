# Polygon Packing Problem

This project focuses on the [Polygon Packing problem](https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2024/#problem-description) from the CG 2024 competition. By exploring different methods, we aim to provide useful insights and contribute to better solutions for this optimization challenge.

## Problem Description

The input to the problem consists of:

- **A container**: A convex polygon that serves as the bounding region where items must be placed.
- **A set of items**: Each item is also a polygon with a fixed shape, orientation, and an associated integer value.
  
<div style="display: flex; justify-content: space-around;">
    <img src="https://drive.google.com/uc?export=download&id=1mG4rHhBmrBfExq5-ysKq_oIXSYOjgXXi" alt="Photo 1" width="300" />
    <img src="https://drive.google.com/uc?export=download&id=14ZeNfKfievqEdrsuaC4eCngFRgcENGsW" alt="Photo 2" width="300" />
</div>

  
## Constraints

- **No Overlap**: The polygons must be arranged without overlapping or intersecting.
- **No Rotation, Tilting, or Dilation**: The items must maintain their original orientation, shape, and scale during placement.

## Output

- **Maximum Total Value**: The maximum total value that can be obtained from the subset of polygons packed into the container.
- **Subset of Polygons**: The subset of polygons that achieve this maximum value, including their placement coordinates within the container.


---
More information about the algorithms used is available in the following paper: [Project Book](https://drive.google.com/uc?export=download&id=1rgAn8u_n6t_144W3VBG8GQCIFDVnDkwX).

![Looped Sticker](https://drive.google.com/uc?export=download&id=1hIy4e2GCPKnkbMyykuQIpAvOdQGN0FrX)
