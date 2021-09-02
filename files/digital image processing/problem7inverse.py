# -*- coding: utf-8 -*-
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as ply
from PIL import Image
import functions as func

#Inverse filtering
I = Image.open('....\data_src\img_src\Campus.tif')
width, height = I.size

img = np.array(I, dtype=np.uint8)

ply.imshow(img, cmap='Greys_r' )
ply.title('Original')
ply.colorbar()
ply.show()

#Applying steps for filtering in frequency domain:
M, N = np.shape(img) #Sizes of original image

img = np.pad(img, ((0, M), (0, N)), mode='constant', constant_values=0)
P, Q = np.shape(img)#Sizes of padded image

FT = ft.fft2(img)
FT = ft.fftshift(FT)

ply.imshow(np.log(np.sqrt(FT.real**2 + FT.imag**2)), cmap='Greys_r')
ply.colorbar()
ply.title('Fourier transform')
ply.show()


H = np.ones((P, Q), dtype = float)
k = 0.0025 #atmospheric disturbance

for v in range(0, M):
  for u in range(0, N):
    H[M - v, N - u] = np.exp(-k*(u**2 + v**2)**(5/6))
    H[M - v, N + u] = np.exp(-k*(u**2 + v**2)**(5/6))
    H[M + v, N - u] = np.exp(-k*(u**2 + v**2)**(5/6))
    H[M + v, N + u] = np.exp(-k*(u**2 + v**2)**(5/6))
    #because of symmetry, the distances are the same radially from center
  #end for x
#end for y 

ply.imshow(H, cmap='Greys_r')
ply.colorbar()
ply.title('H function')
ply.show()

ply.plot( H[N, M:Q] )
ply.title('Profile of H')
ply.show()

#Creating Butterworth filter
#Computing all distances from center, D(u, v)
D = np.zeros((P, Q), dtype = float)
for v in range(0, M):
  for u in range(0, N):
    D[M + v, N + u] = np.sqrt(u**2 + v**2)/2
    D[M + v, N - u] = np.sqrt(u**2 + v**2)/2
    D[M - v, N + u] = np.sqrt(u**2 + v**2)/2
    D[M - v, N - u] = np.sqrt(u**2 + v**2)/2
    #because of symmetry, the distances are the same radially from center
  #end for x
#end for y
  
ply.imshow(D, cmap = 'Greys_r')
ply.colorbar()
ply.title('Distances')
ply.show()

n = 5
D0 = 40

#Butterworth low-pass filter function:
butter = 1 / np.sqrt( 1 + (D / (D0))**(2*n) )

ply.imshow(butter, cmap='Greys_r')
ply.colorbar()
ply.title('Butterworth')
ply.show()

#Applying Butterworth filter to remove eventual outer spikes
H = H*butter

ply.plot(H[M, N:Q])
ply.title('Profile of H, after Butterworth')
ply.show()

ply.imshow(H, cmap='Greys_r')
ply.colorbar()
ply.title('Butterworth applied to H')
ply.show()


F =  FT / H #Applying inverse filtering


ply.imshow(np.log(np.sqrt(F.real**2 + F.imag**2)), cmap='Greys_r')
ply.colorbar()
ply.title('Result, inverse, freq.dom.')
ply.show()

resultImg = np.real(ft.ifft2(F))
resultImg = resultImg[0:M, 0:N]#Remove padding, and scale values
resultImg = func.scale(resultImg, 256).astype(np.uint8)

ply.imshow(resultImg, cmap='Greys_r')
ply.colorbar()
ply.title('Result, inverse')
ply.show()

