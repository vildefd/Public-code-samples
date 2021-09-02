# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as ply
from PIL import Image


Img = Image.open("..\data_src\img_src\Tromsdalstinden.tif", 'r')
width, height = Img.size

img = np.array(Img).astype(np.uint8)

ply.imshow(img, cmap = 'Greys_r')
ply.title('Original')
ply.show()

bitplanes = [] #Array containing bitslices 1-8
bitmask = 0x00000001
numplanes = 8 #number of total bitplanes


for plane in range(0, numplanes):# 3-bit image, 0-255
  
  bitplanes.append(np.zeros((height, width), dtype = int))#Create new image 
  
  for y in range(0, height):
    for x in range(0, width):    

      bitplanes[plane][y, x] = (img[y, x] & bitmask).astype(np.uint8)
      
    #end for x
  #end for y
  bitmask = bitmask << 1 #left-shift the significant bit from least to more significant
  
  #Show image
  ply.imshow(bitplanes[plane], cmap='Greys_r')
  ply.title('Bitplane ' + str(plane))
  ply.show()
#end for plane


#With plane 6-7:
newimg = bitplanes[7] | bitplanes[6] | bitplanes[5]
#Show image
ply.imshow(newimg, cmap='Greys_r')
ply.colorbar()
ply.title('Planes 6-8')
ply.show()

#With plane 5-7:
newimg = bitplanes[4] | bitplanes[3] | bitplanes[2]
#Show image
ply.imshow(newimg, cmap='Greys_r')
ply.colorbar()
ply.title('Planes 3-5')
ply.show()


#With plane 3-1:
newimg = bitplanes[2] | bitplanes[1] | bitplanes[0]
#Show image
ply.imshow(newimg, cmap='Greys_r')
ply.colorbar()
ply.title('Planes 1-3')
ply.show()