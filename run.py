import os
import sys

def read_input():
    input_path = sys.argv[1]
    if not os.path.isdir(input_path):
        print(f"Input path '{input_path}' is not a directory.")
        exit(1)
    return input_path


if __name__ == '__main__':
    input_path = read_input()