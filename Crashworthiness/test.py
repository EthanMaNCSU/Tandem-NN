import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adadelta
from pandas import HDFStore
from sklearn.metrics import r2_score
# apply fixed random seed 7
from numpy.random import seed
seed(7)
store = HDFStore('data.h5')
X_train = store["X_train"]
X_test = store["X_test"]
Y_train = store["Y_train"]
Y_test = store["Y_test"]
X_max = X_train.max().values
X_min = X_train.min().values
Y_max = Y_train.max().values
Y_min = Y_train.min().values

model_forward = load_model("forward_NN_normalized.h5")
model_tandem = load_model("tandem_NN_normalized.h5")

store = HDFStore('data_normalized.h5')
X_train = store["X_train"]
X_test = store["X_test"]
Y_train = store["Y_train"]
Y_test = store["Y_test"]

# input_layer1 = Input(shape=(3,))
# x1 = Dense(50, name='i1')(input_layer1)
# x1 = Dense(5, name='intermediate')(x1)
# o1 = model_forward(x1)
# model_tandem = Model(input=input_layer1, output=o1, name='tandem NN')
# model_tandem.load_weights("tandem_NN_constrained_weights.h5")
intermediate_layer_model = Model(inputs=model_tandem.input, outputs=model_tandem.get_layer('intermediate').output)
Y_test_curr = Y_test[20:21]
Y_test_curr_true = Y_test_curr*(Y_max - Y_min) + Y_min
X_pred = intermediate_layer_model.predict(Y_test_curr)
X_pred_true = X_pred*(X_max - X_min) + X_min
x1 = X_pred_true[0][0]
x2 = X_pred_true[0][1]
x3 = X_pred_true[0][2]
x4 = X_pred_true[0][3]
x5 = X_pred_true[0][4]
f1 = 1640.2823 + 2.3573285 * x1 + 2.3220035 * x2 + 4.5688768 * x3 + 7.7213633 * x4 + 4.4559504 * x5
f2 = 6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4 + \
     0.0861 * x1 * x5 + 0.3628 * x2 * x4 - 0.1106 * x1 * x1 - 0.3437 * x3 * x3 + 0.1764 * x4 * x4
f3 = -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.024 * x2 * x3 - 0.0118 * x2 * x4 \
     - 0.0204 * x3 * x4 - 0.008 * x3 * x5 - 0.0241 * x2 * x3 + 0.0109 * x4 * x4
Y_cal = [f1, f2, f3]

print('Y_test_curr')
print(Y_test_curr.values)
print('Y_test_curr_true')
print(Y_test_curr_true.values)
print('X_pred')
print(X_pred)
print('X_pred_true')
print(X_pred_true)
Y_pred = model_forward.predict(X_pred)
print('Y_pred')
print(Y_pred)
print('Y_cal')
print(Y_cal)

# print("[Tandem NN] The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_train_true, Y_train_cal)))
# print("[Tandem NN] The R2 score on the Test set is:\t{:0.3f}".format(r2_score(Y_test_true, Y_test_cal)))