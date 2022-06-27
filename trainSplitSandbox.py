# SANDBOX FILE TO UNDERSTAND AND IMPLEMENT TEST-TRAIN SPLIT IN DATASET.

import numpy as np
import random

def getDataSet():
    col1 = []
    col2 = []
    col3 = []
    dataset = []
    with open("examples/earthquake-clean.data.txt", 'r') as f:
        for line in f:
            first, sec, thrd = line.split(",")
            col1.append(first)
            col2.append(sec)
            col3.append(thrd)
            dataset.append([float(first), float(sec), float(thrd)])
    print("1st:")
    print(col1)
    print("2nd:")
    print(col2)
    print("third:")
    print(col3)
    print(dataset[0:5])
    random.shuffle(dataset)
    return dataset

def testTrainSplit():
    x = np.random.rand(100, 5)
    print(f"Original: \n{x}")
    np.random.shuffle(x)
    training, test = x[:80,:], x[80:,:]
    return training, test

def get_train_test_inds(y,train_proportion=0.7):
    '''Generates indices, making random stratified split into train-test subsets
    with proportions -> train_proportion (and 1 - train_proportion) of the initial sample.
    y is any iterable indicating classes of each observation in the given sample.
    (stratified sampling).
    '''

    y=np.array(y)
    train_inds = np.zeros(len(y),dtype=bool)
    test_inds = np.zeros(len(y),dtype=bool)
    values = np.unique(y)
    for value in values:
        value_inds = np.nonzero(y==value)[0]
        np.random.shuffle(value_inds)
        n = int(train_proportion*len(value_inds))
        train_inds[value_inds[:n]]=True
        test_inds[value_inds[n:]]=True

    return train_inds,test_inds

def trainTestSplit2(dataset, ratio):
#     THIS ONE IS BETTER AND MUCH MORE SIMPLE!
    elems = len(dataset)
    middle = int(elems * ratio)
    return dataset[:middle], dataset[middle:]

# train, test = testTrainSplit()
# print(f"train:\n {train}")
# print(f"test:\n {test}")

y = getDataSet()
train, test = trainTestSplit2(getDataSet(), 0.3)
# train_inds,test_inds = get_train_test_inds(y,train_proportion=0.5)
# print (y[train_inds])
# print (y[test_inds])
print(f"original: \n{y}")
print(f"train: \n{train}")
print(f"test: \n{test}")

print(f"\n\nTotal N: {len(y)}")
print(f"\nTrain N {len(train)} + Test N {len(test)} = {len(train) + len(test)}")
