from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from Crashworthiness.preprocess import X_train, X_test, Y_train, Y_test, max_values
from sklearn.metrics import r2_score
# apply fixed random seed 7
from numpy.random import seed
seed(7)

# step 1: build forward NN
input_layer = Input(shape=(5,))
x = Dense(50, input_dim=7, activation='relu', name='f1')(input_layer)
o = Dense(3, name='output')(x)
model_forward = Model(input=input_layer, output=o)
# show forward NN summary
print("[Forward NN summary]")
print(model_forward.summary())

# step 2: compile and fit forward NN\
model_forward.compile(loss='mse', optimizer='adam')
model_forward.fit(X_train, Y_train, epochs=500, validation_split=0.2, verbose=0)

# Calculates and prints r2 score of training and testing data
Y_train_pred = model_forward.predict(X_train)
Y_test_pred = model_forward.predict(X_test)
print("[Forward NN] The R2 score on the Train set is:\t{:0.3f}".format(r2_score(Y_train, Y_train_pred)))
print("[Forward NN] The R2 score on the Test set is:\t{:0.3f}".format(r2_score(Y_test, Y_test_pred)))

# step 3: freeze forward NN
for layer in model_forward.layers:
    layer.trainable = False

# step 4: build tandem NN
input_layer1 = Input(shape=(3,))
x1 = Dense(100, input_dim=7, name='i1')(input_layer1)
x1 = Dense(100, input_dim=7, name='i2')(x1)
x1 = Dense(100, input_dim=7, name='i3')(x1)
x1 = Dense(5, name='intermediate')(x1)
o1 = model_forward(x1)
model_tandem = Model(input=input_layer1, output=o1, name='tandem NN')

# step 5: compile and fit tandem NN
model_tandem.compile(loss='mse', optimizer='adam')
history = model_tandem.fit(Y_train, Y_train, epochs=20000, validation_split=0.2, verbose=0)
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

Y_test_curr = Y_test[0:1]
X_pred = intermediate_layer_model.predict(Y_test_curr)
print(X_pred)
X_pred *= max_values
x1 = X_pred[0][0]
x2 = X_pred[0][1]
x3 = X_pred[0][2]
x4 = X_pred[0][3]
x5 = X_pred[0][4]
f1 = 1640.2823 + 2.3573285 * x1 + 2.3220035 * x2 + 4.5688768 * x3 + 7.7213633 * x4 + 4.4559504 * x5
f2 = 6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4 + \
     0.0861 * x1 * x5 + 0.3628 * x2 * x4 - 0.1106 * x1 * x1 - 0.3437 * x3 * x3 + 0.1764 * x4 * x4
f3 = -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.024 * x2 * x3 - 0.0118 * x2 * x4 \
     - 0.0204 * x3 * x4 - 0.008 * x3 * x5 - 0.0241 * x2 * x3 + 0.0109 * x4 * x4
Y_true = [f1, f2, f3]
print('Y_test_curr')
print(Y_test_curr)
print('X_pred')
print(X_pred)
print('Y_true')
print(Y_true)
