# -*- coding: utf-8 -*-
import numpy as np

def scale(f, L):
  if (L == 0 | (L-1)== 0):
    return(f)
    
  fscaled = (L-1)*(f - np.min(f))/np.max(f - np.min(f))#Scaling
  return(fscaled)
#end def scale


def rotate180(matrix):
	matrix = np.flipud(matrix) #reflect horizontally
	matrix = np.fliplr(matrix) #reflect vertically
	return(matrix)
#end rotate180

def myconv2(img, mask, with_padding = False):
  
  maskWidth, maskHeight = np.shape(mask)
  
  padW = maskWidth//2 #padding left and right
  padH = maskHeight//2 #padding above and below
	
  newImg = np.pad(img, ((padH, padH),(padW, padW)), mode='constant', constant_values=0) #pads img matrix with zeros
  newHeight, newWidth = np.shape(newImg) #Get width and height of the padded image
  
  mask = rotate180(mask)
	#Do the convolution operation by rolling mask over image
  for y in range(0, newHeight - maskHeight):
   for x in range(0, newWidth - maskWidth):     
    # "Boundary box" values:
    
	#Convolute mask :
    newImg[y, x] = np.sum( newImg[ y : y + maskHeight, x: x + maskWidth ] * mask)
    #end for x-range
  #end for y-range
	
	#Return result:
  if(with_padding == False):
	#Boundary box for old image
    startY = padH 
    endY = newHeight - padH 
    startX = padW
    endX = newWidth - padH
    return(newImg[startY : endY, startX : endX ] )#Return convoluted image without padding
    
  else:
    return(newImg) #Return convoluted image with padding 
  #end if-else
#end myconv2