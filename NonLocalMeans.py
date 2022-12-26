#Busra_Unlu_211711008_HW9

import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

#load image
image = cv.imread("noisyImage_Gaussian.jpg",0)
image2 = cv.imread("noisyImage_Gaussian_01.jpg",0)



def NonLocalMeans(image,h, templateWindowSize, searchWindowSize):

  we,he=image.shape
  padwidth = searchWindowSize//2
  padwidth2 = templateWindowSize//2

  # Create padded image with mirroring
  paddedImage = cv.copyMakeBorder(image,padwidth, padwidth, padwidth, padwidth, cv.BORDER_REFLECT)

  #output image
  output = paddedImage.copy()

  #Compare tiles in search window
  for i in range(padwidth, padwidth + he):
    for j in range(padwidth, padwidth + we):
  
      #comparison tiles
      compNbhd = paddedImage[j - padwidth2:j + padwidth2 + 1 , i-padwidth2:i+padwidth2 + 1]
      
      pixelColor = 0
      totalWeight = 0

      #Weight calculation
      for k in range(i - padwidth, i - padwidth + searchWindowSize - templateWindowSize, 1):
        for m in range(j - padwidth, j - padwidth + searchWindowSize - templateWindowSize, 1):   

          #find the small box       
          smallNbhd = paddedImage[m:m+templateWindowSize ,k:k+templateWindowSize ]
          distance = np.sqrt(np.sum(np.square(smallNbhd - compNbhd)))

          #weight is computed as a weighted softmax over the euclidean distances
          weight = np.exp(-distance/h)
          totalWeight += weight
          pixelColor += weight*paddedImage[m + padwidth2, k + padwidth2]

      pixelColor /= totalWeight
      output[j, i] = pixelColor

  return output[padwidth:padwidth+we,padwidth:padwidth+he]


#Gaussian filter
gauss1=cv.GaussianBlur(image, (5, 5), 0, borderType = cv.BORDER_REFLECT)
gauss2=cv.GaussianBlur(image2, (5, 5), 0, borderType = cv.BORDER_REFLECT)

#perform NLM filtering (h,small, big)
NLM_image = NonLocalMeans(image, 11,3,7)
NLM_image2 = NonLocalMeans(image2, 11,3,7)

#opencv NLM (h,small,big)(7,3,5)
y1=cv.fastNlMeansDenoising(image,None,11,3,7)
y2=cv.fastNlMeansDenoising(image2,None,11,3,7)



plt.figure(figsize=(18, 10))
plt.subplot(2, 3, 1)
plt.title("OpenCV NLM")
plt.imshow(y1, cmap='gray')
plt.subplot(2, 3, 2)
plt.title("Gaussian 1")
plt.imshow(gauss1, cmap='gray')
plt.subplot(2, 3, 3)
plt.title("MY NLM")
plt.imshow(NLM_image, cmap='gray')

plt.subplot(2, 3, 4)
plt.title("OpenCV NLM 2")
plt.imshow(y2, cmap='gray')
plt.subplot(2, 3, 5)
plt.title("Gaussian 2")
plt.imshow(gauss2, cmap='gray')
plt.subplot(2, 3, 6)
plt.title("MY NLM 2")
plt.imshow(NLM_image2, cmap='gray')


plt.show()
