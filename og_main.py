import time
import cv2 as cv
import numpy as np
from PIL import Image

# Configurare
path_img = "art.png"
hull = set()


def calculate_centroid(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    centroid_x = sum(x) / len(points)
    centroid_y = sum(y) / len(points)
    return centroid_x, centroid_y


def calculate_angle(point, centroid):
    return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])


def findSide(p1, p2, p):
    val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])

    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


def lineDist(p1, p2, p):
    return abs((p[1] - p1[1]) * (p2[0] - p1[0]) -
               (p2[1] - p1[1]) * (p[0] - p1[0]))


def quickHull(a, n, p1, p2, side):
    ind = -1
    max_dist = 0

    for i in range(n):
        temp = lineDist(p1, p2, a[i])

        if (findSide(p1, p2, a[i]) == side) and (temp > max_dist):
            ind = i
            max_dist = temp

    if ind == -1:
        hull.add((p1[0], p1[1]))
        hull.add((p2[0], p2[1]))
        return

    quickHull(a, n, a[ind], p1, -findSide(a[ind], p1, p2))
    quickHull(a, n, a[ind], p2, -findSide(a[ind], p2, p1))


def printHull(a, n):
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


# Driver code
start_time = time.time()

img = Image.open(path_img).convert("L")
width, height = img.size
pixels = img.load()

black_pixels = []
for y in range(height):
    for x in range(width):
        if pixels[x, y] == 0:
            black_pixels.append((x, y))
n = len(black_pixels)
all_points = printHull(black_pixels, n)

end_time = time.time()
print("\nExecution time: {:.4f} seconds".format(end_time - start_time))

centroid = calculate_centroid(all_points)
sorted_points = sorted(all_points, key=lambda p: calculate_angle(p, centroid))
og_image = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
output_image = cv.cvtColor(og_image, cv.COLOR_GRAY2BGR)
cv.polylines(output_image, [np.array(sorted_points)], isClosed=True, color=(0, 255, 0), thickness=2)

# Afisare
cv.imshow('Convex Hull', output_image)
cv.waitKey(0)
cv.destroyAllWindows()
