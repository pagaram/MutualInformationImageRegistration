import numpy as np
import cv2
import MutualInformation as MI


#primary function to compute image translation and rotation
def registerImages(imgA, imgB, x_limit, y_limit, pixel_spacing=1, bins=256):
    x = 0
    y = 0
    mutual_max = 0

    x_values = np.arange(x_limit[0], x_limit[1], pixel_spacing)
    y_values = np.arange(y_limit[0], y_limit[1], pixel_spacing)

    # looping to see the maximum mutual information
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            imgATransformed = transformImage(imgA, x_values[i], y_values[j], 0)
            mutual_Im = imageMutualInformation(imgATransformed, imgB, bins)

            if mutual_Im > mutual_max:
                mutual_max = mutual_Im
                x = x_values[i]
                y = y_values[j]

    return x, y, mutual_max

#rotates and translates image
def transformImage(img, angle, xShift, yShift):

    sine = np.sin(angle)
    cosine = np.cos(angle)
    M = np.float32([[cosine, -1 * sine, xShift], [sine, cosine, yShift]])
    transformed = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))

    return transformed

# computes intentsity probability
def histogramProbability(img, bins):

    hist, _ = np.histogram(img.ravel(), bins=bins)
    histNorm = hist / float(np.sum(hist))

    return histNorm

# joint intensity probability of two images
def histogramJointProbability(imgA, imgB, bins):

    joint_hist, _, _ = np.histogram2d(imgA.ravel(), imgB.ravel(), bins=bins)
    joint_histNorm = joint_hist/float(np.sum(joint_hist))

    return joint_histNorm


#mutual information between two images
def imageMutualInformation(imgA, imgB, bins):

    probXY = histogramJointProbability(imgA, imgB, bins)
    probX = np.sum(probXY, axis=1)
    probY = np.sum(probXY, axis=0)
    mutualIm = MI.mutual_information(probX, probY, probXY)

    return mutualIm