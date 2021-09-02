# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 23:32:55 2018

@author: Vilde F. D.
"""
import numpy as np
import numpy.fft as ft
import matplotlib.pyplot as ply
from PIL import Image
import functions as func

I = Image.open('..\data_src\img_src\Campus.tif')
width, height = I.size

img = np.array(I, dtype=np.uint8)

ply.imshow(img, cmap='Greys_r' )
ply.title('Original')
ply.colorbar()
ply.show()

#Applying steps for filtering in frequency domain:
M, N = np.shape(img)

img = np.pad(img, ((0, M), (0, N)), mode='constant', constant_values=0)
P, Q = np.shape(img)

hist = np.zeros(256, dtype = int)

for y in range(0, height):
    for x in range(0, width):
      hist[img[ y , x ] ] += 1 
      
   #end for x
#end for y
ply.bar(range(len(hist)), hist)
ply.title('Histogram')
ply.show()

G = ft.fft2(img)
G = ft.fftshift(G)

ply.imshow((np.sqrt(G.real**2 + G.imag**2)), cmap='Greys_r')
ply.colorbar()
ply.title('Fourier transform')
ply.show()

ply.imshow(np.log(np.sqrt(G.real**2 + G.imag**2)), cmap='Greys_r')
ply.colorbar()
ply.title('Logarithm of Fourier transform')
ply.show()

#Computing all H(u, v) (the atmospheric noisemodel) from center, D(u, v)
H = np.ones((P, Q), dtype = float)
k = 0.0025 #atmospheric disturbance

for v in range(0, M):
  for u in range(0, N):
    #Atmospheric disturbance model
    H[M + v, N + u] = np.exp(-k*(u**2 + v**2)**(5/6))
    H[M + v, N - u] = np.exp(-k*(u**2 + v**2)**(5/6))
    H[M - v, N + u] = np.exp(-k*(u**2 + v**2)**(5/6))
    H[M - v, N - u] = np.exp(-k*(u**2 + v**2)**(5/6))
    #because of symmetry, the distances are the same radially from center
  #end for x
#end for y 
  
ply.imshow(H, cmap='Greys_r')
ply.colorbar()
ply.title('H-function')
ply.show()

ply.imshow(np.log(H), cmap='Greys_r')
ply.colorbar()
ply.title('Logarithm of H')
ply.show()

ply.plot( H[M, N:Q] )
ply.title('Profile of H')
ply.show()

#K = 0.009 # Signal to noise ratio
K = 0.00009
#K = 90 

#Wiener filter function:
#W = 1/H * ( np.abs(H.conjugate()* H)**2 / ( ( np.abs(H.conjugate() * H )**2 ) + K ) )
W = 1/H * ( (np.abs(H**2)) / ( ( np.abs(H**2) ) + K ) )#H(u, v) is real, so using this instead

ply.plot( W[M, N:Q] )#Profile of W
ply.title('Profile of Wiener filter')
ply.show()

ply.imshow(W, cmap='Greys_r')
ply.colorbar()
ply.title('Wiener function')
ply.show()

Fhat = W*G #Applying Wiener filter

ply.imshow(np.log(np.sqrt(Fhat.real**2 + Fhat.imag**2)), cmap='Greys_r')
ply.colorbar()
ply.title('Applied Wiener function, Fourier')
ply.show()

Fhat = ft.fftshift(Fhat)
resultImg = np.real(ft.ifft2(Fhat))
resultImg = func.scale(resultImg[0:M, 0:N], 256)#Remove padding, and scale values

ply.imshow(resultImg, cmap='Greys_r')
ply.colorbar()
ply.title('Result, Wiener')
ply.show()

 


