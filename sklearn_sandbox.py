# TRIED TO SEE HOW DATASET OF A TYPICAL PROBLEM LOOKS LIKE.
# WIP!
# WILL PROBABLY DELETE

from sklearn.datasets import load_breast_cancer

#Loading the data
data = load_breast_cancer()

#Preparing the data
x = data.data
y = data.target

print(x[0])
print(x[1])
print("Y now")
print(y)

no_of_x_att = str(len(x[0])).split(" ")
print(f"number of attributes = {no_of_x_att}")
