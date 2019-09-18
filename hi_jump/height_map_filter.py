# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 12:14:33 2019

@author: mfocchi
"""


#URL: https://www.codingame.com/playgrounds/2524/basic-image-manipulation/filtering
#defiravites https://www.crisluengo.net/archives/22
#derivatives https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=4767851
#derivatives https://www.cs.cornell.edu/courses/cs6670/2011sp/lectures/lec02_filter.pdf
#sobel filter https://www.hdm-stuttgart.de/~maucher/Python/ComputerVision/html/Filtering.html
# https://stackoverflow.com/questions/50179551/gaussian-derivative-kernel-for-edge-detection-algorithm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

import scipy 
from scipy import ndimage
from scipy import signal

#important
np.set_printoptions(precision = 3, linewidth = 200, suppress = True)
np.set_printoptions(threshold=np.inf)  #prints full matrix

#from PIL import Image, ImageDraw
#
## Load image:
#input_image = Image.open("input.png")
#input_pixels = input_image.load()
#input_image.show() 


    
def convolute( kernel, in_map, out_map):
    
    # Middle of the kernel
    offset = (int)(kernel.shape[0] / 2)
    print kernel.shape[1]

    
    #output_image = Image.new("RGB", input_image.size)
    #draw = ImageDraw.Draw(output_image)   
    # Compute convolution between intensity and kernels
    for x in range(offset, in_map.shape[0] - offset):
        for y in range(offset, in_map.shape[1] - offset):
            blur_pixel = 0
            for a in range(kernel.shape[0]):
                for b in range(kernel.shape[1]):
                    xn = x + a - offset
                    yn = y + b - offset
#                    print "xn yn ", xn, yn
                    pixel = in_map[xn, yn]
                    blur_pixel += pixel * kernel[a][b]    
            out_map[x,y] = blur_pixel
            
def convoluteHorizzontalkernel( kernel, in_map, out_map):
    
    # Middle of the kernel
    offset = (int) (len(kernel) / 2)
    
    for x in range(in_map.shape[0]):
        for y in range(offset, in_map.shape[1] - offset):
            blur_pixel = 0
            for b in range(len(kernel)):
                  
                    yn = y + b - offset 
#                    print "xn yn ", xn, yn
                    pixel = in_map[x, yn]
                    blur_pixel += pixel * kernel[b]    
            out_map[x,y] = blur_pixel
            
def convoluteVerticalkernel( kernel, in_map, out_map):
    
    # Middle of the kernel
    offset = (int) (len(kernel) / 2)
    
    for x in range(offset, in_map.shape[0] - offset):
        for y in range( in_map.shape[1]):
            blur_pixel = 0
            for a in range(len(kernel)):
                  
                    xn = x + a - offset 
#                    print "xn yn ", xn, yn
                    pixel = in_map[xn, y]
                    blur_pixel += pixel * kernel[a]    
            out_map[x,y] = blur_pixel            
        
def computeDerivative(raw_map, res, direction):
#    sobel filter
     
#    dx=  np.array([[-1.0, 0.0, 1.0],[-1.0, 0.0, 1.0],[-1.0, 0.0, 1.0]])
   
#    dy=1/8.0 * np.array([[-1.0, -2.0, -1.0],[0.0, 0.0, 0.0],[1.0, 2.0, 1.0]])
#
#     
#    derivative_map_x = scipy.signal.convolve2d(raw_map, dx, mode='same', boundary = 'symm', fillvalue=0)
#    derivative_map_y = scipy.signal.convolve2d(raw_map, dy, mode='same', boundary = 'symm', fillvalue=0)
##    derivative_map_y = convolute(dy, raw_map)
#    derivative = np.sqrt(derivative_map_x * derivative_map_x + derivative_map_y * derivative_map_y)
    
#    dx = ndimage.sobel(raw_map, 0)   
#    dy = ndimage.sobel(raw_map, 1)
#    derivative = np.hypot(dx, dy)    
    
    kernel =  np.array([-1.0, 0.0, 1.0])/(2.0*res) #is one line
    derivative_map  = np.zeros_like(raw_map)  
        
    if (direction == 'X')    :
        convoluteHorizzontalkernel(kernel, raw_map, derivative_map)
    if (direction == 'Y')    :
        convoluteVerticalkernel(kernel, raw_map, derivative_map)
    
    return derivative_map
    
def smoothHeightMap(kernel_size, raw_map):
    
    box_kernel = np.ones((kernel_size, kernel_size))/kernel_size**2


    # Gaussian kernel TODO
    gaussian_kernel = [[1 / 256.0, 4  / 256.0,  6 / 256.0,  4 / 256.0, 1 / 256.0],
                       [4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0],
                       [6 / 256.0, 24 / 256.0, 36 / 256.0, 24 / 256.0, 6 / 256.0],
                       [4 / 256.0, 16 / 256.0, 24 / 256.0, 16 / 256.0, 4 / 256.0],
                       [1 / 256.0, 4  / 256.0,  6 / 256.0,  4 / 256.0, 1 / 256.0]]
                       
    kernel = box_kernel                       
    height_map_blur = np.copy(raw_map)
    convolute(kernel, raw_map, height_map_blur)
    return height_map_blur

def plotHeightMap(map, llimit = 0, ulimit = 1):
    plt.figure()
    height_map_raw = ulimit*np.ones_like(map)
    height_map_raw -= map

    plt.imshow(height_map_raw, cmap="gray", vmin=llimit, vmax=ulimit)

    plt.show()
    
    

def createPalletMap(pallet_height, edge_position, height_map_resolution, height_map_size):
    #buld custom height map
    index_edge_position = (int)(edge_position /height_map_resolution)
    number_of_cells = (int)(height_map_size / height_map_resolution)
    height_map = np.zeros((number_of_cells,number_of_cells))
    height_map[:,index_edge_position:number_of_cells] = pallet_height
#    height_map[index_edge_position:number_of_cells,:] = 1.0
  
    return height_map

def create2PalletMap(pallet_height, edge_position, second_pallet_size, second_pallet_height, height_map_resolution, height_map_size):
    #buld custom height map
    index_edge_position = (int)(edge_position /height_map_resolution)
    number_of_cells = (int)(height_map_size / height_map_resolution)
    height_map = np.zeros((number_of_cells,number_of_cells))
    height_map[:,index_edge_position:number_of_cells] = pallet_height   
    index_second_pallet_size = (int)(second_pallet_size /height_map_resolution)

    height_map[0 : number_of_cells/2, index_edge_position : index_edge_position+index_second_pallet_size ] = pallet_height + second_pallet_height    
    return height_map



if __name__ == '__main__':
    
    pallet_height = 0.16
    height_map_resolution = 0.01
    height_map_size = 1.0
    edge_position = 0.4
    # Box Blur kernel
    kernel_size  = 3
    
    height_map = createPalletMap(pallet_height, edge_position, height_map_resolution, height_map_size)
#    print height_map
#    plotHeightMap(height_map)    
    height_map_blur = smoothHeightMap(kernel_size, height_map)
#    plotHeightMap(height_map_blur)
    print height_map_blur        

    height_map_der = computeDerivative(height_map_blur, height_map_resolution,'X')
#    print height_map_der
    plotHeightMap(height_map_der)
    
    
    two_pallet_height_map =create2PalletMap(pallet_height, edge_position, 0.1, 0.08, height_map_resolution, height_map_size)   
    plotHeightMap(two_pallet_height_map)   
    print two_pallet_height_map
    ## Create output image 
    #output_image.show()     
    #output_image.save("output.png")