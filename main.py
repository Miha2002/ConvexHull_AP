import time
import cv2 as cv
import numpy as np
from PIL import Image
from mpi4py import MPI

# Configurare path imagine
path_img = "art.png"

# Initializare MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
# print(size)

# Set pentru salvarea punctelor
hull = set()

# ------------------ Functii ajutatoare ----------------------------------------------
# Calculeaza centrul punctelor rezultate
def calculate_centroid(points):
    x = [point[0] for point in points]
    y = [point[1] for point in points]
    centroid_x = sum(x) / len(points)
    centroid_y = sum(y) / len(points)
    return centroid_x, centroid_y


# Calculeaza unghiul dintre axa OX si 2 puncte
def calculate_angle(point, centroid):
    return np.arctan2(point[1] - centroid[1], point[0] - centroid[0])


# Sorteaza punctele rezultate in functe de unghiul lor fata de centrul formei
def sort_points_by_angle(points, centroid):
    return sorted(points, key=lambda p: calculate_angle(p, centroid))


# Impartirea datelor pentru cele n procese
def chunk_data(data, size):
    chunk_size = len(data) // size
    # surplus = len(data) % size
    chunks = [data[i:i + chunk_size] for i in range(0, (size-1) * chunk_size, chunk_size)]
    chunks.append(data[(size-1) * chunk_size:])  # Ultimul proces primeste mai mult, in caz ca nu se imparte perfect
    return chunks


# ------------------ Functiile algoritmului QuickHull ----------------------------------

# Calc pozitia punctului p fata de dreapta p1,p2
def findSide(p1, p2, p):
    val = (p[1] - p1[1]) * (p2[0] - p1[0]) - (p2[1] - p1[1]) * (p[0] - p1[0])
    if val > 0:
        return 1
    if val < 0:
        return -1
    return 0


# Calc distanta de la pct p pana la dreapta p1,p2
def lineDist(p1, p2, p):
    return abs((p[1] - p1[1]) * (p2[0] - p1[0]) -
               (p2[1] - p1[1]) * (p[0] - p1[0]))


# Algoritmul QuickHull
def quickHull(a, n, p1, p2, side):
    ind = -1
    max_dist = 0

    # Se cauta punctul de dist maxima pt ind=-1 fata de dreapta p1,p2
    for i in range(n):
        temp = lineDist(p1, p2, a[i])

        if (findSide(p1, p2, a[i]) == side) and (temp > max_dist):
            ind = i
            max_dist = temp

    # Daca nu se gaseste niciun punct p, de adauga p1,p2 la rezultat
    if ind == -1:
        hull.add((p1[0], p1[1]))  # Add tuple instead of string
        hull.add((p2[0], p2[1]))  # Add tuple instead of string
        return

    # Partea de recursivitate
    quickHull(a, n, a[ind], p1, -findSide(a[ind], p1, p2))
    quickHull(a, n, a[ind], p2, -findSide(a[ind], p2, p1))


def printHull(a, n):
    # Nu putem avea o forma convexa din 2 puncte
    if (n < 3):
        print("Error: Not enough points. Convex hull not possible!")
        return []

    # Se cauta punctele care au coord max si min, punctele din care pornim algoritmul
    min_x = 0
    max_x = 0
    for i in range(1, n):
        if a[i][0] < a[min_x][0]:
            min_x = i
        if a[i][0] > a[max_x][0]:
            max_x = i

    quickHull(a, n, a[min_x], a[max_x], 1)
    quickHull(a, n, a[min_x], a[max_x], -1)

    # Se strang toate rezultatele de la procese in process 0
    all_hulls = comm.gather(hull, root=0)

    if rank == 0:
        # Combinam rezultatele
        all_points = []
        for proc_hull in all_hulls:
            # Afisare pe portiuni
            centroid = calculate_centroid(list(proc_hull))
            sortd = sort_points_by_angle(list(proc_hull), centroid)

            # og_image = cv.imread(path_img, cv.IMREAD_GRAYSCALE)
            # output_image = cv.cvtColor(og_image, cv.COLOR_GRAY2BGR)
            # cv.polylines(output_image, [np.array(list(sortd))], isClosed=True, color=(0, 255, 0), thickness=2)
            # cv.imshow('Convex Hull', output_image)
            # cv.waitKey(0)
            # cv.destroyAllWindows()

            all_points.extend(sortd)
        return all_points


# Apelam din nou QuickHull, dar folosim numai procesul 0 pentru a organiza rezultatele initiale
def secondPrintHull(a, n):
    # Golim hull pentru rezultatele noi
    hull.clear()

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


# ------------------ Driver code ----------------------------------------------------------------
start_time = time.time()

if rank == 0:
    # Deschidem imaginea si o facem alb-negru
    img = Image.open(path_img).convert("L")
    img = img.point(lambda x: 0 if x<128 else 255, '1') # Face imaginea alb/negru = 1/0
    # img.show()
    width, height = img.size
    pixels = img.load()

    # Extragem numai pixelii negri, pe care ii consideram punctele din algoritm
    black_pixels = []
    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0:  # negru = 0
                black_pixels.append((x, y))
    n = len(black_pixels)
else:
    black_pixels = None
    n = 0

# Impartirea datelor la toate procesele
black_pixels = comm.bcast(black_pixels, root=0)
black_pixels_chunks = chunk_data(black_pixels, size)
black_pixels_local = comm.scatter(black_pixels_chunks, root=0)

# Apelarea functiei QuickHull
all_points = printHull(black_pixels_local, len(black_pixels_local))

# Se strang rezultatele
all_points = comm.gather(all_points, root=0)

if rank == 0:
    all_points = [x for x in all_points if x is not None]
    all_points_flat = [point for sublist in all_points for point in sublist]
    # print("\nAll points in Convex Hull:", all_points_flat)

    # Calculam timpul de executie al algoritmului
    end_time = time.time()
    print("\nInitial execution time for {} processes: {:.4f} seconds".format(size, end_time - start_time))

    # Centrul punctelor negre / formei
    centroid = calculate_centroid(all_points_flat)
    sorted_points = sort_points_by_angle(all_points_flat, centroid)

    # A doua executare a algoritmului
    final_result = secondPrintHull(sorted_points, len(sorted_points))
    centroid = calculate_centroid(final_result)
    final_sorted_points = sort_points_by_angle(final_result, centroid)

    total_end_time = time.time()
    print("Total execution time for {} processes: {:.4f} seconds\n".format(size, total_end_time - start_time))

    # ------------------ Afisarea rezultatului ------------------------------------------------------------
    # output_image = cv.imread(path_img)
    # cv.polylines(output_image, [np.array(final_sorted_points)], isClosed=True, color=(0, 255, 0), thickness=2)
    #
    # cv.imshow('Convex Hull', output_image)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
