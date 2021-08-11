from joblib import load
import glob
import os
import sys

import src.hog as hog


def method2(input_path, clf):

    pix_per_cell = (16,16)
    _, _, fd, _, blocks_bounds = hog.hog_data(input_path,
                                              pixels_per_cell=pix_per_cell,
                                              cells_per_block=(7,7))

    pred = hog.predict(clf, fd, blocks_bounds, overlap_threshold=pix_per_cell[0]*2)

    return pred


def read_input():
    input_path = sys.argv[1]
    if not os.path.isdir(input_path):
        print(f"Input path '{input_path}' is not a directory.")
        exit(1)

    return input_path


if __name__ == '__main__':
    input_path = read_input()
    files = glob.glob(os.path.join(input_path, '*'))

    method = 'method2'

    if method == 'method2':
        clf = load('model/model.joblib')
        files = [os.path.splitext(os.path.basename(fn))[0] for fn in files]

        for fn in files:
            pred = method2(fn, clf)

    print(pred)
    # TODO : save to csv
    
