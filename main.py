import time
import random
import cv2 as cv
import numpy as np
from PIL import Image
from mpi4py import MPI

# Configurare path
path_img = "cal.png"


# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# print(size)

# Stores the result (points of convex hull)
hull = set()


# Function to calculate centroid of points
def calculate_centroid(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    centroid_x = sum(x) / len(points)
    centroid_y = sum(y) / len(points)
    return centroid_x, centroid_y

# Function to calculate angle between two points and the x-axis
def calculate_angle(point, centroid):
    return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

# Function to sort points based on their angle relative to the centroid
def sort_points_by_angle(points, centroid):
    return sorted(points, key=lambda p: calculate_angle(p, centroid))

# ----------------------------------------------------------

# Returns the side of point p with respect to line
# joining points p1 and p2.
def findSide(p1, p2, p):
    val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0

# returns a value proportional to the distance
# between the point p and the line joining the
# points p1 and p2
def lineDist(p1, p2, p):
    return abs((p[1] - p1[1]) * (p2[0] - p1[0]) -
               (p2[1] - p1[1]) * (p[0] - p1[0]))


# End points of line L are p1 and p2. side can have value
# 1 or -1 specifying each of the parts made by the line L
def quickHull(a, n, p1, p2, side):
    ind = -1
    max_dist = 0

    # finding the point with maximum distance
    # from L and also on the specified side of L.
    for i in range(n):
        temp = lineDist(p1, p2, a[i])

        if (findSide(p1, p2, a[i]) == side) and (temp > max_dist):
            ind = i
            max_dist = temp

    # If no point is found, add the end points
    # of L to the convex hull.
    if ind == -1:
        hull.add((p1[0], p1[1]))  # Add tuple instead of string
        hull.add((p2[0], p2[1]))  # Add tuple instead of string
        return

    # Recur for the two parts divided by a[ind]
    quickHull(a, n, a[ind], p1, -findSide(a[ind], p1, p2))
    quickHull(a, n, a[ind], p2, -findSide(a[ind], p2, p1))


def printHull(a, n):
    # a[i].second -> y-coordinate of the ith point
    if (n < 3):
        print("Convex hull not possible")
        return []

    # Finding the point with minimum and
    # maximum x-coordinate
    min_x = 0
    max_x = 0
    for i in range(1, n):
        if a[i][0] < a[min_x][0]:
            min_x = i
        if a[i][0] > a[max_x][0]:
            max_x = i

    quickHull(a, n, a[min_x], a[max_x], 1)
    quickHull(a, n, a[min_x], a[max_x], -1)

    # Gather results to root process
    all_hulls = comm.gather(hull, root=0)

    if rank == 0:
        # Combine hull points from all processes into a single array
        all_points = []
        for proc_hull in all_hulls:
            centroid = calculate_centroid(list(proc_hull))
            sortd = sort_points_by_angle(list(proc_hull), centroid)

            # og_image = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
            # output_image = cv.cvtColor(og_image, cv.COLOR_GRAY2BGR)
            #
            # cv.polylines(output_image, [np.array(list(proc_hull))], isClosed=True, color=(0, 255, 0), thickness=2)
            #
            # # Show the result
            # cv.imshow('Convex Hull', output_image)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            all_points.extend(sortd)

        return all_points


def secondPrintHull(a, n):
    # reset hull
    hull.clear()

    # a[i].second -> y-coordinate of the ith point
    if (n < 3):
        print("Convex hull not possible")
        return

    min_x = 0
    max_x = 0
    for i in range(1, n):
        if a[i][0] < a[min_x][0]:
            min_x = i
        if a[i][0] > a[max_x][0]:
            max_x = i

    quickHull(a, n, a[min_x], a[max_x], 1)
    quickHull(a, n, a[min_x], a[max_x], -1)

    return hull


# Divide data into equal-sized chunks for scattering
def chunk_data(data, size):
    chunk_size = len(data) // size
    surplus = len(data) % size
    chunks = [data[i:i + chunk_size] for i in range(0, (size-1) * chunk_size, chunk_size)]
    chunks.append(data[(size-1) * chunk_size:])  # Last process gets the surplus
    return chunks


# ---------------------------------------------------------------------------------------------------
# ---------------- DRIVER CODE ----------------------------------------------------------------------


start_time = time.time()

if rank == 0:
    # Load the black and white image
    img = Image.open(path_img).convert("L")
    width, height = img.size
    pixels = img.load()

    # Extract black pixel coordinates
    black_pixels = []
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0:  # Assuming black pixel value is 0
                black_pixels.append((x, y))
    n = len(black_pixels)
else:
    black_pixels = None  # Initialize as None for non-root processes
    n = 0

# Broadcast data to all processes
black_pixels = comm.bcast(black_pixels, root=0)

# Scatter data to different processes
black_pixels_chunks = chunk_data(black_pixels, size)
black_pixels_local = comm.scatter(black_pixels_chunks, root=0)

# Execute printHull function
all_points = printHull(black_pixels_local, len(black_pixels_local))

# Gather all points to the root process
all_points = comm.gather(all_points, root=0)


# Combine all points into a single array
if rank == 0:
    all_points = [x for x in all_points if x is not None]
    all_points_flat = [point for sublist in all_points for point in sublist]

    # print("\nAll points in Convex Hull:", all_points_flat)

    # Measure execution time
    end_time = time.time()
    print("\nExecution time: {:.4f} seconds".format(end_time - start_time))

    # centrul punctelor negre / formei
    centroid = calculate_centroid(all_points_flat)
    sorted_points = sort_points_by_angle(all_points_flat, centroid)

    # A doua executare a algoritmului
    final_result = secondPrintHull(sorted_points, len(sorted_points))
    centroid = calculate_centroid(final_result)
    final_sorted_points = sort_points_by_angle(final_result, centroid)

    # Afisare imagine rezultata ----------------------------------------------------------------------

    og_image = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
    output_image = cv.cvtColor(og_image, cv.COLOR_GRAY2BGR)
    cv.polylines(output_image, [np.array(final_sorted_points)], isClosed=True, color=(0, 255, 0), thickness=2)

    # Show the result
    cv.imshow('Convex Hull', output_image)
    cv.waitKey(0)
    cv.destroyAllWindows()
