import numpy as np
import sys

# This assignment based on homework submitted for CS445 Machine Learning, Fall 2022

# Calculate L2 distances
# Input is a data point sliced to x/y coordinates and a single centroid
# Returns distance between data point and centroid

def L2(data, centroid):
    pretotal = (data - centroid) ** 2
    sumtotal = np.sum(pretotal)
    return np.sqrt(sumtotal)

# Identify closest centroid
# Input is data point including its cluster assignment element and the centroid array
# Output is the data slice with updated cluster assignment


def calc_closest(data, centroid, num_k):
    last_distance = sys.float_info.max
    for i in range(0, num_k):
        current_distance = L2(data[0:2], centroid[i]) ** 2
        if last_distance > current_distance:
            last_distance = current_distance
            data[2] = i
    return data

# Iterate through data array to update cluster assignments
# Inputs are data full data array and the centroid array
# Output is the data array with updated cluster assignments


def calc_distance(data, centroid, num_k):
    for i in range(0, len(data)):
        data[i] = calc_closest(data[i, :], centroid, num_k)
    return data

# Centroid initializer
# Input is data array
# Output is num_k random set of x/y coordinate slices


def randomize_centroids(data, num_k):
    random = np.random.randint(0, high=len(data), size=num_k)
    output = data[random, :-1]
    return output

# Compute centroids based on current data membership
# Input is data array
# Output is updated centroid array


def compute_centroids(data, num_k):
    centroids = np.zeros((num_k, 2))
    num_points = np.zeros((num_k, 1))
    for i in range(0, len(data)):
        centroids[int(data[i, 2])][0] = centroids[int(
            data[i, 2])][0] + data[i, 0]
        centroids[int(data[i, 2])][1] = centroids[int(
            data[i, 2])][1] + data[i, 1]
        num_points[int(data[i, 2])] += 1
    return centroids / num_points

# Calculate MSE for current data and centroids
# Input is data array with cluster assignments and centroids array
# Output is MSE


def WCSS(data, centroids):
    sum = 0
    for i in range(0, len(data)):
        sum = sum + (L2(data[i, 0:2], centroids[int(data[i, 2])]) ** 2)

    return sum

# Primary EM loop for one run of K-Means
# Input is starting data array
# Output is final MSE, resulting centroid array, and resulting data array with cluster assignments
# Uses double-lookback on centroids to avoid a centroid oscillation issue
# Terminates if the current centroid array match either of the previous two centroid arrays
# Includes a catch to return early if a centroid drops out


def kmeans_run(data, num_k):
    dims = np.ndim(data)
    centroid_start = randomize_centroids(data, num_k)
    centroid_last = np.zeros(centroid_start.shape)
    centroid_last_2 = np.zeros(centroid_start.shape)
    repeat = True
    while (repeat):
        centroid_last_2 = centroid_last
        centroid_last = centroid_start
        data = calc_distance(data, centroid_start, num_k)
        centroid_start = compute_centroids(data, num_k)
        if (np.isnan(centroid_start).any()):
            return 0, centroid_start, data

        if (np.array_equal(centroid_start, centroid_last, True) == True):
            repeat = False
        if (np.array_equal(centroid_start, centroid_last_2, True) == True):
            repeat = False

    run_WCSS = WCSS(data, centroid_start)

    return run_WCSS, centroid_start, data

# Filename helper to generate a file name based on current parameters


def generate_filename(type, num_k):
    return "kmeans-k" + str(num_k) + "-" + type + ".csv"

def run_2d(file, r, num_k):

    # Set up data
    cluster_data_preprop = np.loadtxt(file)

    cluster_data = np.zeros(
        (len(cluster_data_preprop), len(cluster_data_preprop[0]) + 1))
    cluster_data[:, :-1] = cluster_data_preprop


    # Initialize arrays to capture data for each run
    all_WCSS = np.zeros(r)
    all_centroids = np.zeros((r, num_k, 2))
    all_data = np.zeros((r, len(cluster_data), len(cluster_data[0])))
    i = 0

    # Primary loop for each run
    # Includes a check for a vanished centroid, restarting a run if one is detected
    while i < r:
        all_WCSS[i], all_centroids[i], all_data[i] = kmeans_run(cluster_data, num_k)
        if (np.isnan(all_centroids[i]).any()):
            print("Found NaN on run " + str(i) + ", restarting")
        else:
            print("k-Means run " + str(i))
            print("WCSS: " + str(all_WCSS[i]))
            print("Centroids: " + str(all_centroids[i]))
            i += 1

    # Find lowest MSE and save that data
    best_run = np.argmin(all_WCSS)
    np.savetxt(generate_filename("centroids", num_k),
            all_centroids[best_run], delimiter=",", fmt="%f")
    np.savetxt(generate_filename("dataset", num_k),
            all_data[best_run], delimiter=",", fmt="%f")
    print("Best WCSS was on run " + str(best_run))
    
def run_3d(r, num_k):

    # Set up data
    cluster_data_preprop = np.loadtxt("510_cluster_dataset.txt")

    cluster_data = np.zeros(
        (len(cluster_data_preprop), len(cluster_data_preprop[0]) + 1))
    cluster_data[:, :-1] = cluster_data_preprop


    # Initialize arrays to capture data for each run
    all_WCSS = np.zeros(r)
    all_centroids = np.zeros((r, num_k, 2))
    all_data = np.zeros((r, len(cluster_data), len(cluster_data[0])))
    i = 0

    # Primary loop for each run
    # Includes a check for a vanished centroid, restarting a run if one is detected
    while i < r:
        all_WCSS[i], all_centroids[i], all_data[i] = kmeans_run(cluster_data, num_k)
        if (np.isnan(all_centroids[i]).any()):
            print("Found NaN on run " + str(i) + ", restarting")
        else:
            print("k-Means run " + str(i))
            print("WCSS: " + str(all_WCSS[i]))
            print("Centroids: " + str(all_centroids[i]))
            i += 1

    # Find lowest MSE and save that data
    best_run = np.argmin(all_WCSS)
    np.savetxt(generate_filename("centroids", num_k),
            all_centroids[best_run], delimiter=",", fmt="%f")
    np.savetxt(generate_filename("dataset", num_k),
            all_data[best_run], delimiter=",", fmt="%f")
    print("Best WCSS was on run " + str(best_run))
    
    
run_2d("510_cluster_dataset.txt", 10, 5)
