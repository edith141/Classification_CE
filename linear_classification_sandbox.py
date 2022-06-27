from math import exp
from random import shuffle
# prepare dataset
def getDataSet():
    # get the dataset from the file. transform it into a suitable form.
    # then shuffle it and return.
    dataset = []
    with open("examples/earthquake-clean.data.txt", 'r') as f:
        for line in f:
            first, sec, thrd = line.split(",")
            dataset.append([float(first), float(sec), float(thrd)])
    shuffle(dataset)
    return dataset
    # dataset of an OR gate.
    # return [[1,1,1],[1,0,1],[0,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,0],[0,1,1],[1,1,1],[1,0,1],[0,0,0],[0,1,1]]
 
#  spilt data set into train, test acc to the given ratio.
def trainTestSplit2(dataset, ratio):
    elems = len(dataset)
    middle = int(elems * ratio)
    print(f"\n\n\nDataSet Split:")
    print(f"\nTrain Data: \n{dataset[:middle]}")
    print(f"\nTest Data: \n{dataset[middle:]}")
    print("\n")
    return dataset[:middle], dataset[middle:]

def get_avg_from_arr(arr):
    # returns avg value from an array.
    sum = 0
    N = len(arr)
    for elem in arr:
        sum+=elem
    return (sum/N)

def getAccuracy(act, pred):
    # get the accuracy of the model.
    # simply compare pred and actual values.
    corr = 0
    for i in range(len(act)):
        if act[i] == pred[i]:
            corr+=1
    return (corr / float(len(act)) * 100)

# func Predict -> Make a prediction of y (yHat) with the given coeffficients
# yHat = 1.0 / (1.0 + e^(-(b0 + b1 * x1))) 
# b0 -> bias/intercept
# b1 -> coefff. for current (single) value of x
# yHat is nothing but the h(x) as given in the readme.
def predict(row, coeffficients): 
	yhat = coeffficients[0]
	for i in range(len(row)-1):
		yhat += coeffficients[i + 1] * row[i]
    # the sigmoid(x) function -> 1 / (1 + e^(x-1))
	return 1.0 / (1.0 + exp(-yhat))
 
# Estimate logistic regression coeffficients using stochastic gradient descent
def coefficientsSGD(train, LR, n_epoch):
	coeff = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		summError = 0
		for row in train:
			yhat = predict(row, coeff)
			error = row[-1] - yhat
            # error sum squared for logging
			summError += error**2
            # coefff one updating here - the y-intercept or the bias
            # this is in the outer loop as this doesn't depend on individual values of x input.
			coeff[0] = coeff[0] + LR * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
                # this is the weight coefff. to the input x and is hence dependent on each value of x.
                # so it is updated in each iteration..
				coeff[i + 1] = (coeff[i + 1] + LR * error * yhat * (1.0 - yhat) * row[i])
		print('>epoch=%d, lrate=%.3f, error=%.3f, b=%.3f, w=%.3f' % (epoch, LR, summError, coeff[0], coeff[i+1]))
	return coeff


def runLogisticRegression(train, test, LR, epochs):
    #  the performance of the logReg.
    preds = list()
    coeff = coefficientsSGD(train, LR, epochs)
    for row in test:
        yHat = predict(row, coeff)
        yHat = round(yHat)
        preds.append(yHat)
    return preds


def evalAlgo(dataset, algo, *args):
    trainDataSet, testDataSet = trainTestSplit2(getDataSet(), 0.7)
    predictedVals = algo(trainDataSet, testDataSet, *args)
    actualVals = [int(row[-1]) for row in testDataSet]
    print(f"\n\n\nActual Y Values: \n{actualVals}")
    print(f"\nPredicted Y Values: \n{predictedVals}")
    accuracy = getAccuracy(actualVals, predictedVals)

    return accuracy

LR = 0.17
epochs = 8000

print("\n\n\n")
# print(f"Final Accuracy: {scores}")

runs = []

for run in range(5):
    score = evalAlgo(getDataSet(), runLogisticRegression, LR, epochs)
    runs.append(score)
    
print(f'''\n\n
            Total runs to get an avg accuracy of the model: 5\n
            Accuracy array: {runs}\n
            Avg accuracy: {get_avg_from_arr(runs)}%\n\n''')