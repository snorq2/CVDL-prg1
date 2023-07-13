from PIL import Image
import numpy as np

# Gaussian filters
g_filter_3x3 = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
g_filter_5x5 = (1/273) * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])

# DoG filters
g_filter_gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
g_filter_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

# Execute convolution on a patch of image
# Input is an image patch in Numpy array format and the filter to apply.  Both must be same size and shape.
# Output is the convoluted value.
def gaussian_sample(sample: np.ndarray, filter: np.ndarray):
    if np.shape(sample) != np.shape(filter):
        raise AttributeError("Sample must have same shape as filter")
    
    output = sample * filter
    return np.sum(output)

# Execute a filter across an entire image array
# The function automatically pads the image appropriately
# Executes against a single color channel only.
# Input is an unpadded image in Numpy array form, and the filter array to apply to the image.  The filter array must be a square with an odd number of elements on each side.
# Output is a convoluted image in Numpy array form.
def gaussian_iterate(source: np.ndarray, filter: np.ndarray):
    padding = int((len(filter)-1) / 2)
    output = np.zeros(np.shape(source))
    prepped = np.pad(source, padding)
    for i in range(0, len(output)):
        for j in range(0, len(output[0])):
            output[i, j] = gaussian_sample(prepped[i:i+(padding*2)+1, j:j+(padding*2)+1], filter)
    return output

# Opens an image and applies the provided filter.
# Assumes input is either one or three channels, wtih 8 bit channels.
# Input is the image filename and the filter to apply.
# Output is the convoluted image as a Numpy array.
def apply_gaussian(source, filter, floor = 0):
    img = np.array(Image.open(source).convert("L"))
    new_img = np.zeros(np.shape(img), dtype='uint8')
    if np.ndim(img) == 3:
        for i in range(0, len(img[0, 0]-1)):
            new_img[:,:,i] = np.round(gaussian_iterate(img[:,:,i], filter))
    else:
        new_img = gaussian_iterate(img, filter)
    if new_img.min() < floor:
        new_img = np.where(new_img<floor, 0, new_img)
    new_img = np.rint(new_img).astype('uint8')
    return new_img

# Applies a Sobel filter to an image.
# Input is the source file.
# Output is the resulting image as a Numpy array.
def apply_sobel(source, floor=0):
    gx = apply_gaussian(source, g_filter_gx, floor)
    gy = apply_gaussian(source, g_filter_gy, floor)
    gx = gx * gx
    gy = gy * gy
    g = np.sqrt(gx + gy)
    g = np.rint(g).astype('uint8')
    return g

# Executes the entire assignment run against the chosen provided image.
# Input is the image number.
# Output is the saved image output from each filter.
def assignment_run(imnum):
    current_img = "filter" + str(imnum) + "_img.jpg"   
    # gaus_3x3 = Image.fromarray(apply_gaussian(current_img, g_filter_3x3))
    # gaus_5x5 = Image.fromarray(apply_gaussian(current_img, g_filter_5x5))
    gaus_gx = Image.fromarray(apply_gaussian(current_img, g_filter_gx, 50))
    gaus_gy = Image.fromarray(apply_gaussian(current_img, g_filter_gy, 50))
    # gaus_3x3.save('3x3_' + current_img)
    # gaus_5x5.save('5x5_' + current_img)
    gaus_gx.save('gx_' + current_img)
    gaus_gy.save('gy_' + current_img)
    sobel = Image.fromarray(apply_sobel(current_img, 50))
    sobel.save('sobel_' + current_img)
    
# assignment_run(1)
assignment_run(2)