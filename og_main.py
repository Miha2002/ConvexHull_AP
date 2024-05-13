import time
import cv2 as cv
import numpy as np
from PIL import Image

# Configure
path_img = "cal.png"

# Stores the result (points of convex hull)
hull = set()

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
        return

    # Finding the point with minimum and
    # maximum x-coordinate
    min_x = 0
    max_x = 0
    for i in range(1, n):
        if a[i][0] < a[min_x][0]:
            min_x = i
        if a[i][0] > a[max_x][0]:
            max_x = i

    # Recursively find convex hull points on
    # one side of line joining a[min_x] and
    # a[max_x]
    quickHull(a, n, a[min_x], a[max_x], 1)

    # Recursively find convex hull points on
    # other side of line joining a[min_x] and
    # a[max_x]
    quickHull(a, n, a[min_x], a[max_x], -1)

    return hull

# Driver code
start_time = time.time()

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

# Execute printHull function
all_points = printHull(black_pixels, n)

# Calculate centroid
def calculate_centroid(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    centroid_x = sum(x) / len(points)
    centroid_y = sum(y) / len(points)
    return centroid_x, centroid_y

# Calculate angle
def calculate_angle(point, centroid):
    return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])

# Measure execution time
end_time = time.time()
print("\nExecution time: {:.4f} seconds".format(end_time - start_time))

# Sort points by angle
centroid = calculate_centroid(all_points)
sorted_points = sorted(all_points, key=lambda p: calculate_angle(p, centroid))

og_image = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
output_image = cv.cvtColor(og_image, cv.COLOR_GRAY2BGR)
cv.polylines(output_image, [np.array(sorted_points)], isClosed=True, color=(0, 255, 0), thickness=2)

# Show the result
cv.imshow('Convex Hull', output_image)
cv.waitKey(0)
cv.destroyAllWindows()
