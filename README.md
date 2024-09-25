<div align="center">

# Polygon Packing Problem

</div>


This project focuses on the [Polygon Packing problem](https://cgshop.ibr.cs.tu-bs.de/competition/cg-shop-2024/#problem-description) from the CG 2024 competition. 
By exploring different methods, we aim to provide useful insights and contribute to potential solutions for this optimization challenge.

<div align="center">
    <img src="https://drive.google.com/uc?export=download&id=1hIy4e2GCPKnkbMyykuQIpAvOdQGN0FrX" alt="Looped Sticker" style="max-width: 100%; height: auto;" />
</div>

---

## Problem Description
## The input

- **A container**: A convex polygon that serves as the bounding region where items must be placed.
- **A set of items**: Each item is a polygon with a fixed shape, orientation, and an associated integer value.
  
<div align="center">
    <img src="https://drive.google.com/uc?export=download&id=14ZeNfKfievqEdrsuaC4eCngFRgcENGsW" alt="Photo 2" width="300" />
    <img src="https://drive.google.com/uc?export=download&id=1mG4rHhBmrBfExq5-ysKq_oIXSYOjgXXi" alt="Photo 1" width="300" />
</div>

  
## The Constraints

- **No Overlap**: The polygons must be arranged without overlapping or intersecting.
  
- **No Rotation, Tilting, or Dilation**: The items must maintain their original orientation, shape, and scale during placement.

## The Output

- **Maximum Total Value**: The maximum total value that can be obtained from the subset of polygons packed into the container.
- **Subset of Polygons**: The subset of polygons that achieve this maximum value, including their placement coordinates within the container.
<div align="center">
    <img src="https://drive.google.com/uc?export=download&id=1Md2Ugz5EXh01djkKjqbgDq0VdyMRr7PM">
</div>

## Algorithms

Our final solution employs three primary methods for the Packing algorithm: **Double Tangent**, **Item Placement**, and **Splitting**.

1. **Main Algorithm**: The convex region is split into subregions if its width and height are imbalanced. Items are then placed in a counterclockwise direction, aiming to enhance packing efficiency by minimizing gaps between items. The algorithm prioritizes high utility items, which have small areas and high value.

2. **Double Tangent**: This method calculates tangents between items to determine optimal placement, ensuring that the new item does not intersect the previous one.

3. **Item Placement**: The item is moved towards the boundary of the convex region based on the smallest distance between its points and the region's boundary.

4. **Splitting**: If the region is imbalanced in terms of width and height, it is recursively split into smaller regions with more equal dimensions.

More information about the algorithms used is available in the following paper: [Project Book](https://drive.google.com/file/d/1zsJCYtVNwr5ZldGUu-WDJCH95DzbJBNq/view?usp=sharing).

## Technical Implementation

Our solution is implemented in Python utilizing the following libraries and tools:

- **Geometric Libraries**: We used the **Shapely** library for handling polygonal data and performing intersection checks.
- **Visualization**: We used the **Matplotlib** library for visual output, showcasing the final arrangement of polygons within the container for better analysis.

## Installation Guide

### 1. Clone the Repository
To copy the remote repository to your local machine, use the following command:

```bash
git clone https://github.com/mishdub/Final_Project.git
```

Once cloned, navigate into the project directory:

```bash
cd Final_Project
```

### 2. Install Dependencies
Ensure Python is installed on your system. Then, install the required libraries for polygon handling and visualizations:

```bash
pip install Shapely matplotlib
```

### 3. Running the Algorithm
To execute the polygon packing algorithm, run the `main.py` file:

```bash
python main.py
```

When prompted, input the instance path for the algorithm to use. For example:

```bash
Challenge Instances/Atris/atris1672.cgshop2024_instance.json
```

---
  
**Prepared by:**

Mishell Dubovitski  
Alina Zakhozha

**Supervised by:**  
Dr. Gabriel Nivasch

