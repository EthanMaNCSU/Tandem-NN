from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adadelta
from pandas import HDFStore
from sklearn.metrics import r2_score
# apply fixed random seed 7
from numpy.random import seed
seed(7)
store = HDFStore('data_normalized.h5')
X_train = store["X_train"]
X_test = store["X_test"]
Y_train = store["Y_train"]
Y_test = store["Y_test"]

# step 1: build forward NN
input_layer = Input(shape=(5,))
x = Dense(50, activation='relu', name='f1')(input_layer)
o = Dense(3, name='output')(x)
model_forward = Model(input=input_layer, output=[o])
# show forward NN summary
print("[Forward NN summary]")
print(model_forward.summary())
# step 2: compile and fit forward NN\
model_forward.compile(loss='mse', optimizer = "Adam")
history = model_forward.fit(X_train, Y_train, epochs=500, validation_split=0.2)
for layer in model_forward.layers:
    layer.trainable = False
model_forward.save("forward_NN_normalized.h5")
# Plots loss vs. epoch
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(loss_values, 'b', label='training loss')
plt.plot(val_loss_values, 'r', label='training loss val')
plt.show()
# Calculates and prints r2 score of training and testing data
Y_train_pred = model_forward.predict(X_train)
Y_test_pred = model_forward.predict(X_test)
print("[Forward NN] The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_train, Y_train_pred)))
print("[Forward NN] The R2 score on the Test set is:\t{:0.3f}".format(r2_score(Y_test, Y_test_pred)))
