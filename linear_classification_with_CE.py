import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from random import shuffle


def getDataSet():
	# get the dataset from the file. transform it into a suitable form.
	# then shuffle it and return.
	dataset = []
	X1 = []
	X2 = []
	y = []
	with open("./earthquake-clean.data.txt", 'r') as f:
		for line in f:
			first, sec, thrd = line.split(",")
			dataset.append([float(first), float(sec), float(thrd)])
		shuffle(dataset)
		for i in range(len(dataset)):
			X1.append(float(dataset[i][0]))
			X2.append(float(dataset[i][1]))
			y.append(float(dataset[i][2]))
			# dataset.append([float(first), float(sec), float(thrd)])
	# shuffle(dataset)
	return X1, X2, y


def random():
	X1 = []
	X2 = []
	y = []

	np.random.seed(1)
	for i in range(0,20):
		X1.append(i)
		X2.append(np.random.randint(100))
		y.append(0)

	for i in range(20,50):
		X1.append(i)
		X2.append(np.random.randint(80,300))
		y.append(1)

	print(f"X1: {X1}\n" )
	print(f"X2: {X2}\n")
	print(f"Y: {y}\n")
	return X1,X2,y


def standardize(data):
	data -= np.mean(data)
	data /= np.std(data)
	return data


def plot(X):
	plt.scatter(X[:,0],X[:,1])
	plt.xlabel('X1',fontweight="bold",fontsize = 15)
	plt.ylabel('X2',fontweight="bold",fontsize = 15)
	plt.title("Scatter Data",fontweight="bold",fontsize = 20)
	plt.show()

def sigmoid(X,theta):
	z = np.dot(X,theta.T)
	return 1.0/(1+np.exp(-z))


def cost_function(h,y):
	loss = ((-y * np.log(h))-((1-y)* np.log(1-h))).mean()
	return loss


def gradient_descent(X,h,y):
	return np.dot(X.T,(h-y))/y.shape[0]


def update_loss(theta,learning_rate,gradient):
	return theta-(learning_rate*gradient)


def predict(X,theta):
	threshold = 0.5
	outcome = []
	result = sigmoid(X,theta)
	for i in range(X.shape[0]):
		if result[i] <= threshold:
			outcome.append(0)
		else:
			outcome.append(1)
	return outcome


def plot_cost_function(cost):
	plt.plot(cost,label="loss")
	plt.xlabel('Iteration',fontweight="bold",fontsize = 15)
	plt.ylabel('Loss',fontweight="bold",fontsize = 15)
	plt.title("Cost Function",fontweight="bold",fontsize = 20)
	plt.legend()
	plt.show()


def plot_predict_classification(X,theta):
	plt.scatter(X[:,1],X[:,2])
	plt.xlabel('X1',fontweight="bold",fontsize = 15)
	plt.ylabel('X2',fontweight="bold",fontsize = 15)
	x = np.linspace(-1.5, 1.5, 50)
	y = -(theta[0] + theta[1]*x)/theta[2]
	plt.plot(x,y,color="red",label="Decision Boundary")
	plt.title("Decision Boundary for Logistic Regression",fontweight="bold",fontsize = 20)
	plt.legend()
	plt.show()




# main

if __name__ == "__main__":

	X1,X2,y = getDataSet()

	X1 = standardize(X1)
	X2 = standardize(X2)

	X = np.array(list(zip(X1,X2)))
	# print(X)

	y = np.array(y)

	plot(X)

	# Feature Length
	m = X.shape[0]

	# No. of Features
	n = X.shape

	# No. of Classes
	k = len(np.unique(y))

	# Initialize intercept with ones
	intercept = np.ones((X.shape[0],1))

	X = np.concatenate((intercept,X),axis= 1)

	# Initialize theta with zeros
	theta = np.zeros(X.shape[1])

	num_iter = 500

	cost = []

	for i in range(num_iter):
		h = sigmoid(X,theta)
		cost.append(cost_function(h,y))
		gradient = gradient_descent(X,h,y)
		theta = update_loss(theta,0.1,gradient)


	outcome = predict(X,theta)

	plot_cost_function(cost)

	print("theta_0 : {} , theta_1 : {}, theta_2 : {}".format(theta[0],theta[1],theta[2]))

	metric = confusion_matrix(y,outcome)

	print(metric)

	plot_predict_classification(X,theta)
