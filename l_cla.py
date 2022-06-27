# WIP 
# PROBABLY WILL DELETE

import numpy as np
import sys
# from random import randint
import random
from numpy import linalg as LinAl

# import 

# basic percepteron model implementation
class Perceptron():
    def __init__(self, xTrain, yTrain) -> None:
        self.xTrain = xTrain
        self.yTrain = yTrain
        self._eta = 0.01
        self._epochs = 20

    def train(self):
        wts = np.zeros([2, 2])
        for epoch in range(self._epochs):
            mistake = 0
            for i in range(len(self.xTrain)):
                x, y = shuffle(self.xTrain, self.yTrain)
                yHat = int(np.argmax(np.dot(wts, x)))
                
                if y != yHat:
                    mistake+=1
                    wts[y, :] = wts[y, :] + self._eta * x
                    wts[yHat, :] = wts[yHat, :] - self._eta * x
        return mistake, wts

# utility functions
def shuffle(x, y):
    p = random.randint(0, len(x)-1)
    return x[p], y[p]

def loadData():
    with open("examples/earthquake-clean.data.txt", 'r') as f:
        col1 = []
        col2 = []
        # col3 = []
        for line in f:
            first, sec, thrd = line.split(",")
            col1.append([first, sec])
            col2.append(sec)
            # col3.append(thrd)
    return(col1, col2)

def test(xTest, yTest, wts):
    miss = 0
    for i in range(len(xTest)):
        res =  int(np.argmax(np.dot(wts, xTest[i])))
        if res != yTest[i]:
            miss+=1
    return miss

trainX, trainY = loadData()

percp = Perceptron(trainX, trainY)
w, mistakes = percp.train()
miss = test(trainX, trainY)

print(f"Perceptron accuracy: {(1 - miss / len(trainX)) * 100}")
