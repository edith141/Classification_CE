
from random import shuffle

# Function(s) to get data from file/


def getDataSet():
	# get the dataset from the file. transform it into a suitable form.
	# then shuffle it and return.
	dataset = []
	X1 = []
	X2 = []
	y = []
	XT1 = []
	XT2 = []
	yT = []

	with open("examples/earthquake-clean.data.txt", 'r') as f:
		for line in f:
			first, sec, thrd = line.split(",")
			dataset.append([float(first), float(sec), float(thrd)])
		shuffle(dataset)
		TrainDataSet, TestDataSet = trainTestSplit2(dataset, 0.5)
		X1, X2, y = getDataSetInXXZ(dataSet=TrainDataSet)
		XT1, XT2, yT = getDataSetInXXZ(TestDataSet)
			# dataset.append([float(first), float(sec), float(thrd)])
	# shuffle(dataset)
	return X1, X2, y, XT1, XT2, yT

def getDataSetInXXZ(dataSet):
	X1 = []
	X2 = []
	y = []
	for i in range(len(dataSet)):
		X1.append(dataSet[i][0])
		X2.append(dataSet[i][1])
		y.append(dataSet[i][2])

	return X1, X2, y

def trainTestSplit2(dataset, ratio):
	elems = len(dataset)
	middle = int(elems * ratio)
	print(f"\nDataSet Split: {ratio*100} : {(1-ratio)*100}")
	# print(f"\nTrain Data: \n{dataset[:middle]}")
	# print(f"\nTest Data: \n{dataset[middle:]}")
	# print("\n")

	print(f"Train Data Size: {len(dataset[:middle])}")
	print(f"Test Data Size: {len(dataset[middle:])}")

	return dataset[:middle], dataset[middle:]