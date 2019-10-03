import numpy as np
from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.models import Sequential, Model
from keras.layers import Dense, Input
from preprocess import X_train, X_test, Y_train, Y_test, X_train_inv, X_test_inv, Y_train_inv, Y_test_inv
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
model_forward = Model(input=input_layer, output=o)

# step 2: compile and fit forward NN
model_forward.compile(Adam(lr=0.001), 'mean_squared_error')
history = model_forward.fit(X_train, Y_train, epochs=100, validation_split=0.2, verbose=0)
# show forward NN summary
print("[Forward NN summary]")
print(model_forward.summary())
# Calculates and prints r2 score of training and testing data
Y_train_pred = model_forward.predict(X_train)
Y_test_pred = model_forward.predict(X_test)
print("[Forward NN] The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_train, Y_train_pred)))
print("[Forward NN] The R2 score on the Test set is:\t{:0.3f}".format(r2_score(Y_test, Y_test_pred)))

# step 3: freeze forward NN
for layer in model_forward.layers:
    layer.trainable = False

# step 4: build tandem NN
input_layer1 = Input(shape=(5,))
x1 = Dense(10, input_dim=7, activation='relu', name='i1')(input_layer1)
x1 = Dense(10, activation='relu', name='i2')(x1)
x1 = Dense(10, activation='relu', name='i3')(x1)
x1 = Dense(7, name='intermediate')(x1)
o1 = model_forward(x1)
model_tandem = Model(input=input_layer1, output=o1)
# intermediate_layer_weights = np.ones((10, 7), dtype='int32')
# intermediate_layer = model_tandem.get_layer('intermediate')
# intermediate_layer.set_weights([intermediate_layer_weights, intermediate_layer.get_weights()[1]])

# step 5: compile and fit tandem NN
model_tandem.compile(Adam(lr=0.001), 'mean_squared_error')
history = model_tandem.fit(Y_train, Y_train, epochs=100, validation_split=0.2, verbose=0)
# show tandem NN summary
print("[Tandem NN summary]")
print(model_tandem.summary())
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

# step 6: predict X given Y using intermediate layer
intermediate_layer_model = Model(inputs=model_tandem.input,
                                 outputs=model_tandem.get_layer('intermediate').output)

# Calculates and prints r2 score of training and testing data
X_train_pred = intermediate_layer_model.predict(Y_train)
X_test_pred = intermediate_layer_model.predict(Y_test)
print("[Intermediate layer] The R2 score on the Train set is:\t{:0.3f}".format(r2_score(X_train, X_train_pred)))
print("[Intermediate layer] The R2 score on the Test set is:\t{:0.3f}".format(r2_score(X_test, X_test_pred)))