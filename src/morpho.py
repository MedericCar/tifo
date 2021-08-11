from skimage.morphology import binary_opening, binary_closing, disk, skeletonize
import cv2
import matplotlib.pyplot as plt
import numpy as np


def extract_skeleton(veins_img, display=False):
    veins = veins_img.copy()
    
    veins = binary_opening(veins, disk(1))
    veins = binary_closing(veins, disk(2))
    veins = binary_opening(veins, disk(3))
    veins = binary_closing(veins, disk(3))
    
    skel = skeletonize(veins, method='lee').astype(np.uint8)

    if display:
        plt.figure(figsize=(30,30))

        plt.subplot(121)
        plt.title('Veins image')
        plt.imshow(veins, cmap='gray')

        plt.subplot(122)
        plt.title('Skeleton image')
        plt.imshow(skel, cmap='gray')

    return skel


def detect_corners(skel):

    corners = cv2.cornerHarris(skel, 2, 3, 0.04)
    corners_coords = np.argwhere(corners > 0.4*corners.max())
    
    return corners_coords