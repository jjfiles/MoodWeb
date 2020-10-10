import os
import cv2
import time
import argparse
import csv
import numpy as np

IMAGESIZE = 48
IMAGEDIR = './data/train'
OUTDIR = './datasets/train'
CANNYMIN = 100
CANNYMAX = 200

CATEGORIES = [
    'Angery', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
]

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="path to image data", action="store", default=IMAGEDIR)
parser.add_argument("-s", "--size", help="specify final image size", type=int, default=IMAGESIZE)
parser.add_argument("-o", "--output", help="path to save csv to", action="store", default=OUTDIR)
parser.add_argument("--min", help="specify canny min value", type=int, default=CANNYMIN)
parser.add_argument("--max", help="specify canny max value", type=int, default=CANNYMAX)
args = parser.parse_args()

print("Generating csv from images...")
with open(f"{args.output}/er_data_{time.time()}_{args.size}px.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow([0] * (args.size ** 2 + 1))
    directories = sorted(os.listdir(args.data))
    for directory in directories:
        path = os.path.join(args.data, directory)
        if not os.path.isdir(path):
            continue
            images = sorted(os.listdir(path))
            for img in images:
                percent = int((images.index(img) + 1) / len(images) * 20)
                print(f"{directory}:\t{directories.index(directory)+1}/{len(directories)}\t[{('#' * percent) + (' ' * (20-percent))}] {percent * 5}%\r", end='')
                img_path = os.path.join(path, img)
                img_array = cv2.resize(img_array, (args.size, args.size))
                img_array = np.array(img_array).reshape(-1) / 255.0
                img_array = np.insert(img_array, 0, CATEGORIES.index(directory))
                writer.writerow(img_array)
            print('')
        print(f"Created dataset: {file.name}")
