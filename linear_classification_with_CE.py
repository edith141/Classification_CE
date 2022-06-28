
from cProfile import label
import numpy as np
import pandas as pd
from pkg_resources import ResolutionError
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from random import shuffle
from get_data import getDataSet

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


def plot(X, XT):
	plt.scatter(X[:,0],X[:,1])
	plt.scatter(XT[:, 0], XT[:, 1], color='hotpink')
	plt.xlabel('X1',fontsize = 12)
	plt.ylabel('X2',fontsize = 12)
	plt.title("Scatter Data",fontweight="bold",fontsize = 12)
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

def getAccuracy(result, original, details = False):
	errors = 0
	N = len(original)
	for i in range(N):
		if result[i] != original[i]:
			errors += 1
	if details: return N, errors, ((N - errors) / N) * 100
	return ((N - errors) / N) * 100


def plot_cost_function(cost):
	plt.plot(cost,label="loss")
	plt.xlabel('Iteration',fontweight="bold",fontsize = 12)
	plt.ylabel('Loss',fontweight="bold",fontsize = 12)
	plt.title("Cost Function",fontweight="bold",fontsize = 12)
	plt.legend()
	plt.show()


def plot_predict_classification(X, XT, theta):
	plt.scatter(X[:,1],X[:,2], label='Train Data')
	plt.scatter(XT[:, 1], XT[:, 2], color='hotpink', label='Test Data')
	plt.xlabel('X1',fontsize = 12)
	plt.ylabel('X2',fontsize = 12)
	x = np.linspace(-1.5, 1.5, 50)
	y = -(theta[0] + theta[1]*x)/theta[2]
	plt.plot(x,y,color="red",label="Decision Boundary")
	plt.title("Decision Boundary for Logistic Regression",fontweight="bold",fontsize = 12)
	plt.legend()
	plt.show()




# main

def runTest(tRuns):
	accArr = []
	for _ in range(tRuns):
		X1,X2,y, XT1, XT2, yT = getDataSet()

		X1 = standardize(X1)
		X2 = standardize(X2)

		XT1 = standardize(XT1)
		XT2 = standardize(XT2)

		X = np.array(list(zip(X1,X2)))
		XT = np.array(list(zip(XT1, XT2)))
		# print(f"\n\n\nX: {XT}")
		# print(X)

		y = np.array(y)
		yT = np.array(yT)

		# plot(X, XT)
		

		# Feature Length
		m = X.shape[0]

		# No. of Features
		n = X.shape

		# No. of Classes
		k = len(np.unique(y))

		# Initialize intercept with ones
		intercept = np.ones((X.shape[0],1))
		interceptT = np.ones((XT.shape[0],1))

		X = np.concatenate((intercept,X),axis= 1)
		XT = np.concatenate((interceptT, XT), axis=1)

		# Initialize theta with zeros
		theta = np.zeros(X.shape[1])

		num_iter = 500

		cost = []

		for i in range(num_iter):
			h = sigmoid(X,theta)
			cost.append(cost_function(h,y))
			gradient = gradient_descent(X,h,y)
			theta = update_loss(theta,0.1,gradient)

		print(f"\nDimensions:\nX dim: {X.shape}")
		print(f"XT dim: {XT.shape}")
		print(f"Theta dim: {theta.shape}")
		# print(f"\n\nX: {X}")

		# print(f"\n\n\nXLAST: {X.shape} \n {X}")
		# print(f"\n\n\nXTLAST: {XT.shape} \n {XT}")
		# plot_predict_classification(X, XT, theta)

		outcome = predict(XT,theta)
		print(f"\nTheta: {theta}")
		# plot_cost_function(cost)
		
		print("theta_0 : {} , theta_1 : {}, theta_2 : {}".format(theta[0],theta[1],theta[2]))
		print(f"Iterations: {num_iter}")
		print("\nMetrics:")
		metric = confusion_matrix(yT,outcome)
		N, errorsN, accuracy = getAccuracy(outcome, yT, True)
		print(f"Confusion Matrix:")
		print(metric)

		print(f"\nAccuracy:")
		print(f"N: {N}\nErrors: {errorsN}\nAccuracy: {accuracy}")
		accArr.append(accuracy)
	return accArr

tRuns = 100
accArr = runTest(tRuns)
print(f"Average accuracy in {tRuns}: {sum(accArr) / tRuns}")
