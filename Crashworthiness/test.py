import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adadelta
from pandas import HDFStore
from sklearn.metrics import r2_score
# apply fixed random seed 7
import numpy as np
import random
np.random.seed(7)

store = HDFStore('data_normalized.h5')
X_max = store["X_max"].values
X_min = store["X_min"].values
Y_max = store["Y_max"].values
Y_min = store["Y_min"].values
model_tandem = load_model("tandem_NN_normalized.h5")
intermediate_layer_model = Model(inputs=model_tandem.input, outputs=model_tandem.get_layer('intermediate').output)

def calculate_discrete_degree(input):
    if len(input) == 0:
        print('Invalid input!')
        return
    res = 0
    for i in input:
        res += (i-0.5)**2
    return (res/len(input))**0.5/0.5

mse_scores = HDFStore('mse.h5')
mse_scores = pd.DataFrame(mse_scores['df'])
mse_scores = mse_scores.sort_values(by=['mse'])
discrete_degree_Y = []
for row in mse_scores.values:
    discrete_degree_Y.append(calculate_discrete_degree(row[0:3]))

# print(discrete_degree_Y)
data = np.load("data_Y_space.npy")
print(data)
