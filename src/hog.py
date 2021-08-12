from matplotlib import patches
from skimage import exposure
from skimage.feature import hog
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from .preprocessing import adaptive_global_thresholding, extract_veins
from .utils import read_img


def compute_hog(img, orientations, pixels_per_cell, cells_per_block, display=False):

    if not display:
        blocks = hog(img, orientations=orientations, 
                     pixels_per_cell=pixels_per_cell,
                     cells_per_block=cells_per_block, visualize=display, 
                     feature_vector=False)
        return blocks, None

    else:
        blocks, hog_image = hog(img, orientations=9, 
                                pixels_per_cell=pixels_per_cell,
                                cells_per_block=cells_per_block,
                                visualize=display, feature_vector=False)

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

        return blocks, hog_image_rescaled


def get_blocks_bounds(n_blocks_row, n_blocks_col, pixels_per_cell, cells_per_block):
    c_row, c_col = pixels_per_cell
    b_row, b_col = cells_per_block
    
    blocks_bounds, center_bounds = [], []
    for r in range(n_blocks_row):
        for c in range(n_blocks_col):
            row_start = r + b_row // 2
            col_start = c + b_col // 2
            blocks_bounds.append(((r*c_row, (r + b_row)*c_row), 
                                  (c*c_col, (c + b_col)*c_row)))
            center_bounds.append(((row_start*c_row, (row_start + 1)*c_row), 
                                  (col_start*c_col, (col_start + 1)*c_row)))

    return blocks_bounds, center_bounds


def blocks_gt(pts_coord, center_bounds):

    blocks_gt = []
    for ((bx_low, bx_high), (by_low, by_high)) in center_bounds:
        found = False
        for (x, y) in pts_coord:
            if (bx_low <= x <= bx_high) and (by_low <= y <= by_high):
                found = True
                break
        blocks_gt.append(found)
        
    blocks_gt = np.array(blocks_gt)

    return blocks_gt


def hog_data(filename, pixels_per_cell, cells_per_block, display=False,
             plot_dir=None, mode='predict'):

    img = read_img(f'{filename}.jpg')

    plot_outfile = os.path.join(plot_dir, filename + '_mask.png') if plot_dir else None
    filtered = adaptive_global_thresholding(img, display=display,
                                            plot_outfile=plot_outfile)
    veins_mask = extract_veins(filtered) != 0
    img[~veins_mask] = 0

    if display or plot_dir:
        plt.figure()
        plt.title('Filtered image')
        plt.imshow(img)

        if plot_dir:
            plt.savefig(os.path.join(plot_dir, filename + '_filtered.png'))
        
        if not display:
            plt.close()

    blocks, hog_img = compute_hog(img, orientations=9, 
                                  pixels_per_cell=pixels_per_cell,
                                  cells_per_block=cells_per_block, display=display)

    (n_blocks_row, n_blocks_col, b_row, b_col, orientations) = blocks.shape

    blocks_bounds, center_bounds = get_blocks_bounds(n_blocks_row, n_blocks_col,
                                                     pixels_per_cell, 
                                                     cells_per_block)

    if mode == 'train':
        true = pd.read_csv(f'{filename}.csv')
        pts_coords = list(true.itertuples(name=None, index=False))
        gt = blocks_gt(pts_coords, center_bounds)
    else:
        gt = None

    fd = np.reshape(blocks, (n_blocks_row*n_blocks_col, b_row*b_col*orientations))
    blocks_bounds = np.array(blocks_bounds)

    return img, hog_img, fd, gt, blocks_bounds


def plot_hog_data(img, hog_img, fd, bb, block_i):
    plt.figure(figsize=(20, 5))

    ax = plt.subplot(141)
    plt.title('Block position')
    plt.imshow(img)

    x1, x2 = bb[block_i][0, :]
    y1, y2 = bb[block_i][1, :]
    rect = patches.Rectangle((y1, x1), y2-y1, x2-x1, linewidth=2, edgecolor='r',
                             facecolor='none')
    ax.add_patch(rect)

    plt.subplot(142)
    plt.title('Block')
    plt.imshow(img[x1:x2, y1:y2])

    plt.subplot(143)
    plt.title('Block gradients')
    plt.imshow(hog_img[x1:x2, y1:y2])

    plt.subplot(144)
    plt.title('HOG block descriptor histogram')
    plt.bar(np.arange(len(fd[block_i])), fd[block_i])


def create_hog_dataset(files, pixels_per_cell=(16, 16), cells_per_block=(7, 7)):
    x, y, blocks_bounds  = [], [], []

    for fn in files:
        img, hog_img, fd, gt, bb = hog_data(fn, plot_dir='plots',
                                            pixels_per_cell=pixels_per_cell,
                                            cells_per_block=cells_per_block,
                                            mode='train')
        mask = fd.any(axis=1)
        fd = fd[mask]
        gt = gt[mask]
        bb = bb[mask]
        x.append(fd)
        y.append(gt)
        blocks_bounds.append(bb)

    return x, y, blocks_bounds
    

def select_subset(x, y):
    pts_inds = np.where(y == 1)[0]
    x_pts = x[pts_inds]
    y_pts = y[pts_inds]

    # Ensure there is no repetition
    rand_inds = np.array(list(set(np.arange(x.shape[0])) - set(pts_inds)))
    rand_inds = np.random.choice(rand_inds, len(y_pts)*20, replace=False)

    x_sub = np.concatenate([x[rand_inds], x_pts])
    y_sub = np.concatenate([y[rand_inds], y_pts])

    return x_sub, y_sub


def predict(clf, x, blocks_bound, overlap_threshold):
    y_pred = clf.predict(x)
    blocks_bound = blocks_bound[np.where(y_pred == 1)[0]]

    # Replace blocks which overlap too much by there average
    overlapping = []

    for bounds in blocks_bound:
        pushed = False
        x1, x2 = bounds[0, :]
        y1, y2 = bounds[1, :]
        pt = np.array([x1 + (x2-x1) // 2, y1 + (y2-y1) // 2])

        for i in range(len(overlapping)):
            centroid, l = overlapping[i]
            dist = np.linalg.norm(pt - centroid, ord=1)
            if dist <= overlap_threshold:
                l.append(bounds)
                n = len(l)
                overlapping[i][0] = (centroid * (n - 1) / n) + (pt * 1 / n)
                pushed = True
                break

        if not pushed:
            overlapping.append([pt, [bounds]])

    blocks_bound = np.array([np.mean(l, axis=0) for _, l in overlapping])

    # Retrieve middle point of blocks as prediction
    pred_pts = []
    for bounds in blocks_bound:
        x1, x2 = bounds[0, :]
        y1, y2 = bounds[1, :]
        pt = (x1 + (x2-x1) // 2), (y1 + (y2-y1) // 2)
        pred_pts.append(pt)

    return pred_pts