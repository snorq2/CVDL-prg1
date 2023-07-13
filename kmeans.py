import numpy as np
import sys
from PIL import Image

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
    dims = len(data)-1
    for i in range(0, num_k):
        current_distance = L2(data[0:dims], centroid[i]) ** 2
        if last_distance > current_distance:
            last_distance = current_distance
            data[dims] = i
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
    random = np.random.randint(0, high=len(data)-1, size=num_k)
    output = data[random, :-1]
    return output

# Compute centroids based on current data membership
# Input is data array
# Output is updated centroid array

def compute_centroids(data, num_k):
    centroids = np.zeros((num_k, len(data[0])-1))
    num_points = np.zeros((num_k, 1))
    dims = len(data[0])
    for i in range(0, len(data)):
        for j in range(0, len(data[0]) - 1):
            centroids[int(data[i, dims-1])][j] = centroids[int(data[i, dims-1])][j] + data[i, j]
        num_points[int(data[i, dims-1])] += 1
    return centroids / num_points

# Calculate MSE for current data and centroids
# Input is data array with cluster assignments and centroids array
# Output is MSE

def WCSS(data, centroids):
    sum = 0
    dims = len(data[0])
    for i in range(0, len(data)):
        sum = sum + (L2(data[i, 0:(dims-1)], centroids[int(data[i, dims-1])]) ** 2)
    return sum

# Primary EM loop for one run of K-Means
# Input is starting data array
# Output is final MSE, resulting centroid array, and resulting data array with cluster assignments
# Uses double-lookback on centroids to avoid a centroid oscillation issue
# Terminates if the current centroid array match either of the previous two centroid arrays
# Includes a catch to return early if a centroid drops out

def kmeans_run(data, num_k):
    centroid_start = randomize_centroids(data, num_k)
    centroid_last = np.zeros(centroid_start.shape)
    centroid_last_2 = np.zeros(centroid_start.shape)
    repeat = True
    i = 0
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
        
        i += 1
        if i % 10 == 0:
            print("Iter " + str(i))

    run_WCSS = WCSS(data, centroid_start)

    return run_WCSS, centroid_start, data

# Filename helper to generate a file name based on current parameters

def generate_filename(name, type, num_k):
    return name +"_kmeans-k" + str(num_k) + "-" + type + ".csv"

# Executes an arbitrary number of runs of the kmeans algorithm, identifying and saving the one with the least WCSS
# Inputs are the array run kmeans on, the number of runs, the number of centroids, and the unique name of the run
# Output is to a series of CSV files, uniquely identified by the name and run variables.
def run(arr, r, num_k, name):

    # Set up data   
    cluster_data = np.zeros(
        (len(arr), len(arr[0]) + 1))
    cluster_data[:, :-1] = arr


    # Initialize arrays to capture data for each run
    all_WCSS = np.zeros(r)
    all_centroids = np.zeros((r, num_k, len(arr[0])))
    all_data = np.zeros((r, len(cluster_data), len(cluster_data[0])))
    i = 0

    # Primary loop for each run
    # Includes a check for a vanished centroid, restarting a run if one is detected
    while i < r:
        print("Starting k-Means run " + str(i))
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
    np.savetxt(generate_filename(name, "centroids", num_k),
            all_centroids[best_run], delimiter=",", fmt="%f")
    np.savetxt(generate_filename(name, "dataset", num_k),
            all_data[best_run], delimiter=",", fmt="%f")
    with open(generate_filename(name, "WCSS", num_k), 'w') as f:
        f.write(str(all_WCSS[best_run]))
        f.close()
    print("Best WCSS was on run " + str(best_run))

# Generate the colorspace of an image by flattening and deduplicating the array
# Input is an image as a Numpy array
# Output is another Numpy array with all the color channels arranged into an appropriately shaped array for kmeans
def img_colorspace_dedup(img):
    output = img.reshape(-1, img.shape[-1])
    output = np.unique(output, axis=0)
    return output

# Converts the colors into a string to be used in a dictionary lookup
# Input is a n-tuple, output is a string based on this tuple
def color_to_string(color):
    output = ""
    for i in range(0, len(color)):
        output = output + str(color[i]) + "-"
    return output

# Creates a lookup dictionary for converting colors into their nearest centroid
# Takes in a colormap with associated kmeans classification and an array of centroids indexed by the classification column in the colormap
# Returns a dictionary keyed by the colormap.
def generate_converter(colormap, centroids):
    dictmap = { color_to_string(colormap[0][0:3]): centroids[colormap[0][3]]}
    for i in range(1, len(colormap)):
        dictmap.update({color_to_string(colormap[i][0:3]): centroids[colormap[i][3]]})
    return dictmap

# Replaces colors in an image converted into a Numpy array with ones in a dictionary
# Input is the numpy array and the colormap dictionary
# Output is a numpy array with the colors remapped per the dictionary
def update_colors(img, colormap):
    for i in range(0, len(img)):
        for j in range(0, len(img[0])):
            key = color_to_string(img[i][j])
            img[i][j][0] = colormap[key][0]
            img[i][j][1] = colormap[key][1]
            img[i][j][2] = colormap[key][2]
    return img

# Sets up a run to generate the kmeans analysis of an image
# Input is the unique identifier for the images in the assignment, the number of centroids to generate, and the number of runs to test
# Output is the CSV's from the run function.
def generate_kmean_colormap(imname, k, runs):
    img = Image.open('Kmean_' + imname + '.jpg')
    img_arr = np.array(img)
    colorspace = img_colorspace_dedup(img_arr)
    run(colorspace, runs, k, 'imname')

# Applies the colormaps from the generate_kmean_colormap function to produce a new image
# Input is the unique identifier for the images in the assignment and the number of centroids
# Output is a new image file as processed through kmeans
def apply_kmean_colormap(imname, k):
    img = Image.open('Kmean_' + imname + '.jpg')
    img_arr = np.array(img)
    centroids = np.loadtxt(generate_filename(imname, 'centroids', k), delimiter=',')
    colormap = np.loadtxt(generate_filename(imname, 'dataset', k), delimiter=',').astype('uint8')
    centroids = (np.rint(centroids)).astype(int)
    dictionary = generate_converter(colormap, centroids)
    newimg = update_colors(img_arr, dictionary)
    pilim = Image.fromarray(newimg)
    pilim.save('Kmean_processed_k' + str(k) + '_' + imname + '.jpg')

# Batch function to execute the 2D clustering
def run_2d():
    data = np.loadtxt("510_cluster_dataset.txt")  
    run(data, 10, 2, 'dataset')
    run(data, 10, 3, 'dataset')
    run(data, 10, 4, 'dataset')
    
run_2d()

generate_kmean_colormap('img1', 5, 6)
generate_kmean_colormap('img1', 10, 6)
generate_kmean_colormap('img2', 5, 6)
generate_kmean_colormap('img2', 10, 6)

apply_kmean_colormap('img1', 5)
apply_kmean_colormap('img1', 10)
apply_kmean_colormap('img2', 5)
apply_kmean_colormap('img2', 10)