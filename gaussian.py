from PIL import Image
import numpy as np

g_filter_3x3 = (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]])
g_filter_5x5 = (1/273) * np.array([[1, 4, 7, 4, 1], [4, 16, 26, 16, 4], [7, 26, 41, 26, 7], [4, 16, 26, 16, 4], [1, 4, 7, 4, 1]])

g_filter_gx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
g_filter_gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

img1 = Image.open('filter1_img.jpg')
img2 = Image.open('filter2_img.jpg')

# img1.show()
# img2.show()

filt_img1 = np.array(img1)
filt_img2 = np.array(img2)

print(filt_img1)


def gaussian_sample(sample: np.ndarray, filter: np.ndarray):
    if np.shape(sample) != np.shape(filter):
        raise AttributeError("Sample must have same shape as filter")
    
    output = sample * filter
    # output = round(np.sum(output))
    # return output
    
    return np.sum(output)

def gaussian_iterate(source: np.ndarray, filter: np.ndarray):
    padding = int((len(filter)-1) / 2)
    output = np.zeros(np.shape(source))
    prepped = np.pad(source, padding)
    for i in range(0, len(output)):
        for j in range(0, len(output[0])):
            output[i, j] = gaussian_sample(prepped[i:i+(padding*2)+1, j:j+(padding*2)+1], filter)
    return output
    
def apply_gaussian(source, filter):
    img = np.array(Image.open(source))
    new_img = np.zeros(np.shape(img), dtype='uint8')
    if np.ndim(img) == 3:
        for i in range(0, len(img[0, 0]-1)):
            new_img[:,:,i] = np.round(gaussian_iterate(img[:,:,i], filter))
    else:
        new_img = gaussian_iterate(img, filter)
            
    return Image.fromarray(new_img)

def apply_sobel(source):
    img = np.array(Image.open(source))
    gx = gaussian_iterate(img, g_filter_gx)
    gy = gaussian_iterate(img, g_filter_gy)
    gx = gx * gx
    gy = gy * gy
    g = np.sqrt(gx + gy)
    return Image.fromarray(g)
    
 
current_img = "filter2_img.jpg"   
gaus_1 = apply_gaussian(current_img, g_filter_3x3)
# gaus_1_5x5 = apply_gaussian(current_img, g_filter_5x5)
# gaus_1_gx = apply_gaussian(current_img, g_filter_gx)
# gaus_1_gy = apply_gaussian(current_img, g_filter_gy)
img2.show()
gaus_1.show()
# gaus_1_5x5.show()
# gaus_1_gx.show()
# gaus_1_gy.show()
# sobel_1 = apply_sobel('filter1_img.jpg')
# sobel_1.show()