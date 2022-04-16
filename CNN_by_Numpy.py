import numpy as np
import sys
from scipy import signal

def zero_padding(var, height, width):
    padArray = np.zeros((height,width))
    padArray[0:var.shape[0], 0:var.shape[1]] = var
    return padArray

def conv_(img, filter, K, a, b):

    result = np.zeros((K,K))
    for i in range(K):
        for j in range(K):
            curr_region = img[a+3*i+0 : a+3*i+3, b+3*j+0 : b+3*j+3]
            curr_filter = filter[3*i+0 :   3*i+3,   3*j+0 :   3*j+3]
            result[i,j] = np.sum(curr_region*curr_filter)
    return np.sum(result)

def conv(img, conv_filter):

    if len(img.shape) != len(conv_filter.shape) : # Check whether number of dimensions is the same
        print("Error: Number of dimensions in conv filter and image do not match.")  
        exit()
    if len(img.shape) > 2 or len(conv_filter.shape) > 2: # Check if number of image channels matches the filter depth.
        if img.shape[-1] != conv_filter.shape[-1]:
            print("Error: Number of channels in both image and filter must match.")
            sys.exit()
    if conv_filter.shape[0] != conv_filter.shape[1]: # Check if filter dimensions are equal.
        print('Error: Filter must be a square matrix. I.e. number of rows and columns must match.')
        sys.exit()

    # to define the feature map
    a=img.shape[0]-conv_filter.shape[0]+1
    b=img.shape[1]-conv_filter.shape[1]+1
    feature_maps = np.zeros((a, b))

    # to calculate the size to extend
    K = np.uint16(np.ceil(conv_filter.shape[0]/3.0))
    #zero padding to the filter
    conv_filter_with_zero_padding = zero_padding(conv_filter,3*K,3*K)
    #zero padding to the image
    img_with_zero_padding = zero_padding(img, a-1+3*K, b-1+3*K)

    for a_num in range(a):
        for b_num in range(b):
            feature_maps[a_num, b_num] = conv_(img_with_zero_padding, conv_filter_with_zero_padding, K, a_num, b_num)
            
    return feature_maps



#original filter
filter = np.random.randint(3, size=(5, 5))
#input image
img = np.random.randint(10, size=(7, 7))
#convolution 
result = conv(img, filter)
print(result)

#check
out = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        out[i,j] = np.sum(img[0+i:5+i,0+j:5+j]*filter ) 
print(out)


