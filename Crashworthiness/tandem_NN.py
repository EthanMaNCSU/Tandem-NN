from matplotlib import pyplot as plt
from keras import backend as K
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Input, Activation
from keras.optimizers import Adam, Adadelta
from pandas import HDFStore
from sklearn.metrics import r2_score
# apply fixed random seed 7
from numpy.random import seed
from keras.utils.generic_utils import get_custom_objects
seed(7)
store = HDFStore('data_normalized.h5')
X_train = store["X_train"]
X_test = store["X_test"]
Y_train = store["Y_train"]
Y_test = store["Y_test"]

model_forward = load_model("forward_NN_normalized.h5")
input_layer1 = Input(shape=(3,))
x1 = Dense(50, activation='relu', name='i1')(input_layer1)
x1 = Dense(5, activation = 'sigmoid', name='intermediate')(x1)
o1 = model_forward(x1)
model_tandem = Model(input=input_layer1, output=o1, name='tandem NN')

# step 5: compile and fit tandem NN
# model_tandem.compile(loss='mse', optimizer = Adam(lr = 0.00001))
model_tandem.compile(loss='mse', optimizer = "Adam")

history = model_tandem.fit(Y_train, Y_train, epochs=500, validation_split=0.2)
print("[Tandem NN summary]")
print(model_tandem.summary())
model_tandem.save("tandem_NN_normalized.h5")

# Plots loss vs. epoch
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(loss_values, 'b', label='training loss')
plt.plot(val_loss_values, 'r', label='training loss val')
plt.show()
# Calculates and prints r2 score of training and testing data
Y_train_pred = model_tandem.predict(Y_train)
Y_test_pred = model_tandem.predict(Y_test)
print("[Tandem NN] The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_train, Y_train_pred)))
print("[Tandem NN] The R2 score on the Test set is:\t{:0.3f}".format(r2_score(Y_test, Y_test_pred)))