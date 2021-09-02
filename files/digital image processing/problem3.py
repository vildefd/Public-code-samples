# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as ply
from PIL import Image
#import scipy.signal as spy
import functions as func

Img = Image.open('..\data_src\img_src\SAR_Troms.tiff')

width, height = Img.size

img = np.array(Img).astype(np.uint8)

ply.imshow(img, cmap = 'Greys_r')
ply.title('Original')
ply.colorbar()
ply.show()

# a)
h_val = np.zeros(256, dtype = int)

#Computing histogram
for y in range(0, height):
  for x in range(0, width):
    h_val[img[y, x]] += 1
  
#Plotting histogram
ply.bar(range(len(h_val)), h_val)
ply.title("Histogram")
ply.show()

# b )
# Extracting two 100x100 images
oya = img[625:725, 790:890]#nordspissen
ocean = img[790:890, 1200:1300]#hav sør for øyen

h_val2 = np.zeros(256, dtype = int)#oya histogram values
h_val3 = np.zeros(256, dtype = int)#ocean histogram values

#Computing histogram of oya and ocean
for y in range(0, 100):
  for x in range(0, 100):
    h_val2[oya[y, x]] += 1 
    h_val3[ocean[y, x]] += 1
  #end for x
#end for y

#Displaying histogram for oya  
ply.bar(range(len(h_val2)), h_val2)
ply.title('Histogram of Tromsoya')
ply.show()

ply.imshow(oya, cmap = 'Greys_r')
ply.title('100x100 cutout of Tromsoya')
ply.colorbar()
ply.show()

#Displaying histogram for ocean
ply.bar(range(len(h_val3)), h_val3, align = 'center')
ply.title('Histogram of ocean')
ply.show()

ply.imshow(ocean, cmap = 'Greys_r')
ply.title('100x100 cutout of ocean')
ply.colorbar()
ply.show()

print('Mean, oya:', np.mean(oya)                   )
print('Standard deviation, oya:',  np.std(oya)     )
print('Mean, ocean:', np.mean(ocean)               )
print('Standard deviation, ocean:', np.std(ocean)  )

# c)
# Apply smoothing filter
smoothmask = 1/49 * np.ones((7, 7), dtype=int)#Averaging

smoothedimg = func.myconv2(img, smoothmask).astype(np.uint8)#Convolving with mask

#smoothedimg = (smoothedimg**1.06).astype(np.uint8) #Gamma correction

ply.imshow(smoothedimg, cmap='Greys_r')
ply.title('Smoothed, 7x7 averaging mask')
ply.colorbar()
ply.show()

# d)
#Computing new histograms
h_val = np.zeros(256, dtype = int)

#Computing histogram
for y in range(0, height):
  for x in range(0, width):
    h_val[smoothedimg[y, x]] += 1
  
#Plotting histogram
ply.bar(range(len(h_val)), h_val)
ply.title("Histogram, smoothed")
ply.show()

#Looking at histogram of ocean and island
oya = smoothedimg[625:725, 790:890]#nordspissen
ocean = smoothedimg[790:890, 1200:1300]#hav sør for øyen

h_val2 = np.zeros(256, dtype = int)#oya histogram values
h_val3 = np.zeros(256, dtype = int)#ocean histogram values

#Computing histogram of oya and ocean
for y in range(0, 100):
  for x in range(0, 100):
    h_val2[oya[y, x]] += 1 
    h_val3[ocean[y, x]] += 1
  #end for x
#end for y

#Displaying histogram for oya  
ply.bar(range(len(h_val2)), h_val2)
ply.title('Histogram of Tromsoya, smoothed')
ply.show()

ply.imshow(oya, cmap = 'Greys_r')
ply.title('100x100 cutout of Tromsoya, smoothed')
ply.colorbar()
ply.show()

#Displaying histogram for ocean
ply.bar(range(len(h_val3)), h_val3)
ply.title('Histogram of ocean, smoothed')
ply.show()

ply.imshow(ocean, cmap = 'Greys_r')
ply.title('100x100 cutout of ocean, smoothed')
ply.colorbar()
ply.show()

print('Mean, oya, smoothed:', np.mean(oya)                   )
print('Standard deviation, oya, smoothed:',  np.std(oya)     )
print('Mean, ocean, smoothed:', np.mean(ocean)               )
print('Standard deviation, ocean, smoothed:', np.std(ocean)  )


