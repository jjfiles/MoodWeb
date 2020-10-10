import os
import cv2
import random
import pickle
import time
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard

IMAGESIZE = 48
DATADIR = './datasets/train/train.csv'
LOGDIR = './logs'
MODELDIR = './models'
EPOCH = 10

CATEGORIES = [
    'Angery', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'
]

#xTrain is wrong size

# Command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", help="path to training dataset", action="store", default=DATADIR)
parser.add_argument("-l", "--log", help="path to log directory", action="store", default=LOGDIR)
parser.add_argument("-m", "--model", help="path to model directory", action="store", default=MODELDIR)
parser.add_argument("-s", "--size", help="specify training image size", type=int, default=IMAGESIZE)
parser.add_argument("-e", "--epoch", help="specify number of epochs", type=int, default=EPOCH)
args = parser.parse_args()

class InvalidPathError(Exception):
    """Raised when path to training dataset is invalid"""
    pass

def get_train(tDir):
    if os.path.isfile(tDir):
        data_path = tDir
    elif os.path.isdir(tDir):
        # get last modified file in directory
        data_path = max([os.path.join(tDir, file) for file in os.listdir(tDir)], key=os.path.getctime)
        if not os.path.isfile(data_path):
            raise InvalidPathError
    else:
        raise InvalidPathError
    
    print("Reading training data from {data_path}...")
    data = pd.read_csv(data_path, delimiter=',').values[1:]
    xTrain = [[i[1:]] for i in data]
    yTrain = [i[:1] for i in data]
    
    print("Shaping and shuffling data...")
    data = list(zip(xTrain, yTrain))
    random.shuffle(data)
    
    xTrain, yTrain = zip(*data)
    xTrain = np.asarray(xTrain).reshape(-1, args.size, args.size, 1)
    yTrain = np.asarray(yTrain).reshape(-1)
    
    return xTrain, yTrain

if not os.path.isdir(args.log):
    os.makedirs(args.log)
if not os.path.isdir(args.model):
    os.makedirs(args.model)
    
NAME = f"er-cnn-64x2-{time.time()}_{args.size}px"
tb = TensorBoard(log_dir=f"{os.path.join(args.log, NAME)}")

xTrain, yTrain = get_train(args.data)

model = Sequential()

#Layer 1
model.add(Conv2D(64, (3,3),input_shape = xTrain.shape[1:]))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 2
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

#Layer 3
model.add(Conv2D(64, (3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation("relu"))

model.add(Dense(29))
model.add(Activation("softmax"))

model.complie(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
    )

model.fit(xTrain, yTrain, batch_size=5, epochs=args.epoch, validation_split=0.1, callbacks=[tb])

model.save(f"{os.path.join(args.model, NAME+'.h5')}")