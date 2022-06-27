# EXTRA FILE USED AS A SANDBOX TO PLAY WITH DATASET. TRANSFORM IT INTO SOMETHING USABLE BY THE ALGO. 
# WIP!

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

    return dataset
datasetO = list(getDataSet())
datasetS = getDataSet()
random.shuffle(datasetS)

print(f"full data set: \n{datasetO}")
print(f"randomized dataset: \n{datasetS}")
# [2.7810836,2.550537003,0],


def trainTestSplit2(dataset, ratio):
    elems = len(dataset)
    middle = int(elems * ratio)
    return dataset[:middle], dataset[middle:]

