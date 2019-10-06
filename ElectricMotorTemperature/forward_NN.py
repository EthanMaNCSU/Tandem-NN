from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from ElectricMotorTemperature.preprocess import X_train, X_test, Y_train, Y_test
from sklearn.metrics import r2_score
# apply fixed random seed 7
from numpy.random import seed
seed(7)

# step 1: build forward NN
input_layer = Input(shape=(7,))
x = Dense(10, input_dim=7, activation='relu', name='f1')(input_layer)
x = Dense(10, activation='relu', name='f2')(x)
x = Dense(10, activation='relu', name='f3')(x)
o = Dense(5, name='output')(x)
model_forward = Model(input=input_layer, output=[o], name='forward NN')

# step 2: compile and fit forward NN
model_forward.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
history = model_forward.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=0)

# show forward NN summary
print("[Forward NN summary]")
print(model_forward.summary())
# for layer in model_forward.layers: print(layer.get_config(), layer.get_weights())

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
