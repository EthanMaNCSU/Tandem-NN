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

mse_scores = HDFStore('mse_bad_prediction_corner.h5')
mse_scores = pd.DataFrame(mse_scores['df'])
mse_scores = mse_scores.sort_values(by=['mse'])
discrete_degree_Y = []
discrete_degree_X = []
discrete_degree_f1 = []
discrete_degree_f2 = []
discrete_degree_f3 = []

for row in mse_scores.values:
    discrete_degree_Y.append(calculate_discrete_degree(row[0:3]))
    discrete_degree_X.append(calculate_discrete_degree(row[10:15]))
    discrete_degree_f1.append(calculate_discrete_degree(row[0:1]))
    discrete_degree_f2.append(calculate_discrete_degree(row[1:2]))
    discrete_degree_f3.append(calculate_discrete_degree(row[2:3]))

# mse_scores.insert(6, "discrete_f1", discrete_degree_f1, True)
# mse_scores.insert(7, "discrete_f2", discrete_degree_f2, True)
# mse_scores.insert(8, "discrete_f3", discrete_degree_f3, True)
# mse_scores.insert(9, "discrete_Y", discrete_degree_Y, True)
# mse_scores.insert(15, "discrete_X", discrete_degree_X, True)

print(mse_scores)