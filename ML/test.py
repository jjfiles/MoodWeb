import os
import cv2
import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model

IMAGESIZE = 80
DATADIR = './datasets/test'
MODELDIR = './models'

CATEGORIES = [
    'Angery', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
]

#Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("d", "--data", help="path to testing dataset", action="store", default=DATADIR)
parser.add_argument("-m", "--model", help="path to model directory", action="store", default=MODELDIR)
parser.add_argument("-s", "--size", help="specify training image size", type=int, default=IMAGESIZE)
parser.add_argument("-p", "--prediction", help="output predictions", action="store_true")
args = parser.parse_args()

class InvalidPathError(Exception):
    """Raised when path to training dataset is invalid"""
    pass

def get_test(tDir):
    if os.path.isfile(tDir):
        data_path = tDir
    elif os.path.isdir(tDir):
        # get last modified file in directory
        data_path = max([os.path.join(tDir, file) for file in os.listdir(tDir)], key=os.path.getctime)
        if not os.path.isfile(data_path):
            raise InvalidPathError
    else:
        raise InvalidPathError
    
    print(f"Reading testing data from {data_path}...")
    data = pd.read_csv(data_path, delimiter=',').values[1:]
    xTest = [[i[1:]] for i in data]
    yTest = [i[:1] for i in data]
    
    print("Shaping and shuffling data...")
    data = list(zip(xTest, yTest))    
    random.shuffle(data)
    
    xTest, yTest = zip(*data)
    xTest = np.asarray(xTest).reshape(-1, args.size, args.size, 1)
    yTest = np.asarray(yTest).reshape(-1)
    
    return xTest, yTest

def get_model(mDir):
    if os.path.isfile(mDir):
        model_path = mDir
    elif os.path.isdir(mDir):
        model_path = max([os.path.join(mDir, file) for file in os.listdir(mDir)], key=os.path.getctime)
        if not os.path.isfile(model_path):
            raise InvalidPathError
    else:
        raise InvalidPathError
    
    return tf.keras.models.load_model(model_path)
    
    
xTest, yTest = get_test(args.data)    
model = get_model(args.model)

if args.prediction:
    predictions = model.predict([xTest])
    for i in range(len(xTest)):
        correct = int(yTest[i]) == np.argmax(predictions[i])
        print(f"{'✓' if correct else 'X'} {i}\tValue: {CATEGORIES[int(yTest[i])]}\tPrediction: {CATEGORIES[np.argmax(predictions[i])]}")

loss, acc = model.evaluate(xTest, yTest)
print(f"loss: {loss}\taccuracy: {acc}")

