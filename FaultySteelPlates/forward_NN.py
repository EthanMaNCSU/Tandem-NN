from matplotlib import pyplot as plt
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Dense, Input
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score, KFold
from FaultySteelPlates.preprocess import X_train, X_test, Y_train, Y_test
# apply fixed random seed 7
from numpy.random import seed
seed(7)

# step 1: build forward NN
input_layer = Input(shape=(27,))
x = Dense(100, activation='relu', name='f1')(input_layer)
o = Dense(7, activation='softmax', name='output')(x)
model_forward = Model(input=input_layer, output=[o], name='forward NN')

# step 2: compile and fit forward NN
model_forward.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# show forward NN summary
print("[Forward NN summary]")
print(model_forward.summary())

results = model_forward.fit(X_train, Y_train, epochs = 500, validation_split=0.2, verbose=0)

# Plots loss vs. epoch
history_dict = results.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
plt.plot(loss_values, 'b', label='training loss')
plt.plot(val_loss_values, 'r', label='training loss val')
plt.show()

# Calculate scores on training and testing data
pred_train = model_forward.predict(X_train)
scores = model_forward.evaluate(X_train, Y_train, verbose=0)
print('Accuracy on training data: {} \n Error on training data: {}'.format(scores[1], 1 - scores[1]))

pred_test = model_forward.predict(X_test)
scores2 = model_forward.evaluate(X_test, Y_test, verbose=0)
print('Accuracy on test data: {} \n Error on test data: {}'.format(scores2[1], 1 - scores2[1]))