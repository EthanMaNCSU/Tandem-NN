import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Model
from keras.layers import Dense, Input
from FaultySteelPlates.preprocess import X_train, X_test, Y_train, Y_test, max_values
# apply fixed random seed 7
from numpy.random import seed
seed(7)

# step 1: build forward NN
input_layer = Input(shape=(27,))
x = Dense(100, activation='relu', name='f1')(input_layer)
o = Dense(7, activation='softmax', name='output')(x)
model_forward = Model(input=input_layer, output=[o])

# step 2: compile and fit forward NN
model_forward.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# show forward NN summary
print("[Forward NN summary]")
print(model_forward.summary())

results = model_forward.fit(X_train, Y_train, epochs = 500, validation_split=0.2, verbose=0)

# Calculate scores on training and testing data
scores = model_forward.evaluate(X_train, Y_train, verbose=0)
print('[forward NN] Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
scores2 = model_forward.evaluate(X_test, Y_test, verbose=0)
print('[forward NN] Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

# step 3: freeze forward NN
for layer in model_forward.layers:
    layer.trainable = False

# step 4: build tandem NN
input_layer1 = Input(shape=(7,))
x1 = Dense(100, activation='relu', name='i1')(input_layer1)
x1 = Dense(27, name='intermediate')(x1)
o1 = model_forward(x1)
model_tandem = Model(input=input_layer1, output=[o1], name='tandem NN')

# step 5: compile and fit tandem NN
model_tandem.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# show tandem NN summary
print("[Tandem NN summary]")
print(model_tandem.summary())
results = model_tandem.fit(Y_train, Y_train, epochs = 50, validation_split=0.2, verbose=0)

# Plots loss vs. epoch
history_dict = results.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(loss_values, 'b', label='training loss')
plt.plot(val_loss_values, 'r', label='training loss val')
plt.show()

# Calculate scores on training and testing data
scores = model_tandem.evaluate(Y_train, Y_train, verbose=0)
print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))
scores2 = model_tandem.evaluate(Y_test, Y_test, verbose=0)
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))

# step 6: predict X given Y using intermediate layer
intermediate_layer_model = Model(inputs=model_tandem.input,
                                 outputs=model_tandem.get_layer('intermediate').output)
test_data = Y_test[0:1]
X_test_pred = intermediate_layer_model.predict(test_data)
print(test_data)
print(X_test_pred * max_values)
print(model_forward.predict(X_test_pred))

