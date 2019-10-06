from matplotlib import pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from FaultySteelPlates.preprocess import X_train, X_test, Y_train, Y_test, max_values
# apply fixed random seed 7
from numpy.random import seed
seed(7)
data = X_train[6:7]
print(data)
print(max_values)
print(data*max_values)