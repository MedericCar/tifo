from sklearn.neighbors import KernelDensity
from scipy.signal import argrelmax
import cv2
import matplotlib.pyplot as plt
import numpy as np


def adaptive_global_thresholding(img, display=False, plot_outfile=None):
    
    # Downsize image to speedup computations
    scale_percent = 0.1
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    # Estimate the density of the image histogram
    X = resized.flatten().reshape((-1, 1))
    X_plot = np.linspace(0, 255, 1000)[:, np.newaxis]
    kde = KernelDensity(kernel='gaussian', bandwidth=5).fit(X)
    log_dens = kde.score_samples(X_plot)

    # Find the indices of the peaks of the signal
    peak_inds = argrelmax(np.exp(log_dens))[0]

    # Find the beginning of the second peak by checking gradient deviation after first peak (grad > mean + n * std)
    peak0 = peak_inds[0] if len(peak_inds) != 1 else 0
    grad = np.gradient(np.exp(log_dens))
    std = np.std(grad[peak0:])
    mean = np.mean(grad[peak0:])
    n = 1
    while True:
        try:
            threshold_i = np.argwhere(grad[peak0:] >= mean + n*std)[0, 0] + peak0  # shift to global index by adding peak0
        except IndexError:
            n /= 2
        
        tot_dens = np.sum(np.exp(log_dens))
        curr_dens = np.sum(np.exp(log_dens[:threshold_i]))
        if curr_dens / tot_dens >= 0.66:
            n /= 2
        else:
            break

    threshold = X_plot[threshold_i, 0]
    filtered = img.copy()
    filtered[img > threshold] = 0
    filtered[img <= threshold] = 255

    if display or plot_outfile:
        plt.figure(figsize=(30, 5))
        
        plt.subplot(141)
        plt.title("Original image")
        plt.imshow(img, cmap='gray')
        
        plt.subplot(142)
        plt.title("Histogram derivative")
        plt.plot(X_plot[:, 0], grad)
        #plt.scatter(X_plot[peak0], grad[peak0], fc='#00ff00')
        plt.scatter(X_plot[threshold_i], grad[threshold_i], fc='#ff0000', 
                    label='Start of 2nd peak')
        plt.legend()

        plt.subplot(143)
        plt.title("Histogram")
        plt.plot(X_plot[:threshold_i, 0], np.exp(log_dens)[:threshold_i],
                 '#ff8c00', label='Selected subpart')
        plt.plot(X_plot[threshold_i:, 0], np.exp(log_dens)[threshold_i:],
                 '#0073ff')
        plt.scatter(X_plot[peak0], np.exp(log_dens)[peak0], fc='#00ff00',
                    label='First peak')
        plt.scatter(X_plot[threshold_i], np.exp(log_dens)[threshold_i],
                    fc='#ff0000', label='Start of 2nd peak')
        plt.legend()

        plt.subplot(144)
        plt.title("Result of thresholding")
        plt.imshow(filtered, cmap='gray')

        if plot_outfile:
            plt.savefig(plot_outfile)
        
        if not display:
            plt.close()

    return filtered


def extract_veins(filtered, display=False):
    
    _, labels, stats, _ = cv2.connectedComponentsWithStats(filtered,
                                                           connectivity=4)
    veins = np.zeros(filtered.shape)
    label = np.argsort(stats[:, cv2.CC_STAT_AREA])[-2]  # take second largest component (after background)
    veins[labels == label] = 255

    if display:
        plt.figure(figsize=(30,30))

        plt.subplot(121)
        plt.title('Original image')
        plt.imshow(filtered, cmap='gray')

        plt.subplot(122)
        plt.title('Veins image')
        plt.imshow(veins, cmap='gray')

    return veins