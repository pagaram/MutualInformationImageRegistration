import numpy as np
import cv2
import ImageRegistration as IR

# reading test image
img = cv2.imread('panda.png', 0)

#shifting image in xy and y direction
shifted = IR.transformImage(img, 0, 2, 3)

#computing the shift and mutual information after registration
x_limit = [0, 5]
y_limit = [0, 5]
x, y, mutualO = IR.registerImages(img, shifted, x_limit, y_limit,pixel_spacing=0.5)
print(x)
print(y)
print(mutualO)

