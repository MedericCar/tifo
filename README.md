# UnBEEtable
-------------

By Mederic Carriat and Maya El Gemayel.
IMAGE 2022.
Group project for TIFO (Traitement d'Image Fondamentale).

## Goal

Detection of bee's veins intersections.

## How

First of all, let's start by creating an environment:
`python3 -m venv .env && . .env/bin/activate`

Now that it is active, let's install all the requirements using the package
manager `pip`: `pip install -r requirements.txt`

To use the script for bee's veins intersctions detection, please run using the
following command (after activation and installation of all requirements):
`python run.py INPUT_PATH` with INPUT_PATH being the path to the directory
containing all JPG images of bee wings.

## Output

The output created is contained in a directory at the root of the call named
`OUTPUT`. CSV files (image_name.csv) are saved in it with all the intersections
coordinates.

## Some puns

- To bee or not to bee, that is the question.
- What do you call a bee that was born in May? A may-bee.
- What do you call a bee having a bad hair day? A frizz-bee.
- What do you call a bee that’s returned from the dead? A zom-bee.
