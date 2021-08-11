import cv2
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random


def load_annotations(annot_json, out_dir):

    with open(annot_json) as f:
        data = json.load(f)

    annotated_files = []
    for fn in data:
        name = fn.split('.')[:-1]
        name = '.'.join(name)
        annotated_files.append(name)
        points = {'y': [], 'x': []}
        for pt in data[fn]['regions']:
            att = pt['shape_attributes']
            points['x'].append(att['cx'])
            points['y'].append(att['cy'])
        df = pd.DataFrame(points, index=None)
        df.to_csv(os.path.join(out_dir, name + '.csv'), index=False,
                  header=False)

    return annotated_files


def read_img(img_file):
    img = cv2.imread(img_file)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img_gray


def plot_pts(img, pts):
    plt.imshow(img, cmap='gray')
    for y, x in pts:
        plt.scatter(x, y, fc='#ff0000')


def train_test_split(files):
    random.shuffle(files)
    split_i = int(len(files) * 0.3)
    test_files, train_files = files[:split_i], files[split_i:]
    return test_files, train_files


def score(true, pred):
    """ Computes the f1 score by checking if predicted points coordinates are in
    the neighbourhood of a point in the ground truth.
    """

    found = np.full((len(true),), False)
    box_w, box_h = (20, 20)
    FP = 0

    for (yp, xp) in pred:
        matched = []
        for i, (yt, xt) in enumerate(true):
            if (xt - box_h <= xp <= xt + box_h) and (yt - box_w <= yp <= yt + box_w):
                matched.append(i)

        if not matched:
            FP += 1
        else:
            min_i = np.argmin([np.linalg.norm(np.array((yp, xp)) - np.array(true[m]))
                               for m in matched])
            if found[matched[min_i]]:
                FP += 1
            else:
                found[matched[min_i]] = True
    
    FN = len(np.where(found == False)[0])
    TP = len(np.where(found == True)[0])

    try:
        P = TP / (TP+FP)
        R = TP / (TP+FN)
        F1 = 2 * (P*R) / (P+R)

        print(f"Precision : {P:.3f}")
        print(f"Recall : {R:.3f}")
        print(f"F1 score : {F1:.3f}")
    
    except ZeroDivisionError:
        print("Cannot compute scores because of division by 0.")