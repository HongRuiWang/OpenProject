import numpy as np
from numpy import array, reshape

from sklearn.preprocessing import KernelCenterer


# conv_5x5 without split
def conv_without_split(image, filter, kernel_size, out_h, out_w):

    output_v = []
    for oh in range(out_h):
        for ow in range(out_w):
            acc = 0
            for ih in range(kernel_size):
                for iw in range(kernel_size):
                    Image = image[ih+oh][iw+ow]
                    Kernel = filter[ih][iw]
                    multiple = Image*Kernel
                    acc = acc + multiple
            output_v.append(acc)
    output = array(output_v)
    output = output.reshape((out_w, out_h))
    return output

# hardware execution
def conv_3x3(image, filter):
    kernel_size = 3
    acc = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            Image = image[i][j]
            Kernel = filter[i][j]
            multiple = Image*Kernel
            acc = acc + multiple
    return acc

#zero padding
def zero_padding(var, height, width):
    padArray = np.zeros((height,width))
    padArray[0:var.shape[0], 0:var.shape[1]] = var
    return padArray


# ===================================================================
# image & kernel size
image_w = 7  
image_h = 7  
kernel_w = 5 
kernel_h = 5 

# simulate input image and kernel matrix
input_image = np.random.randint(1, 10, size = (image_w, image_h)) 
filter_matrix = np.random.randint(-1, 1, size = (kernel_w, kernel_h)) 
print("input image : ")
print(input_image)
print("kernel matrix : ")
print(filter_matrix)

# to calculate the output size
output_w = (image_w - kernel_w) + 1
output_h = (image_h - kernel_h) + 1

# baseline
result_without_split = conv_without_split(input_image, filter_matrix, 5, output_h, output_w)

# to split 5x5 filter to 4 3x3 filter
## to calculate the size to extend
K = np.uint16(np.ceil(kernel_w/3.0))
## zero padding to the filter
filter_with_zero_padding = zero_padding(filter_matrix, 3*K, 3*K)
#print("filter_with_zero_padding")
#print(filter_with_zero_padding)
# zero padding to the image
image_with_zero_padding = zero_padding(input_image, output_h-1+3*K, output_w-1+3*K)
#print("image_with_zero_padding")
#print(image_with_zero_padding)
##to split 5x5 filter to 4 3x3 filter

output = []
for oh in range(output_h):
    for ow in range(output_w):
        acc=0
        for i in range(K):
            for j in range(K):
                #
                curr_region = image_with_zero_padding[oh+3*i+0 : oh+3*i+3, ow+3*j+0 : ow+3*j+3]
                curr_filter = filter_with_zero_padding[3*i+0 :   3*i+3,   3*j+0 :   3*j+3]
                # hardware execution
                result = conv_3x3(curr_region, curr_filter)
                acc = acc + result
                
        output.append(acc)
print(output)
output = array(output)
output = output.reshape((output_h, output_w))
print("result with split : ")
print(output)
print("result without split : ")
print(result_without_split)








