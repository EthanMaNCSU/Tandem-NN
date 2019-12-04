import pandas as pd
from pandas import HDFStore
from sklearn.model_selection import train_test_split
import random
import numpy as np
# generate data set

num_of_instances = 10000
count = 0
data = {'x1':[], 'x2':[], 'x3':[], 'x4':[], 'x5':[], 'f1':[], 'f2':[], 'f3':[]}
while (count < num_of_instances):
    count = count + 1
    x1 = random.uniform(1, 3)
    x2 = random.uniform(1, 3)
    x3 = random.uniform(1, 3)
    x4 = random.uniform(1, 3)
    x5 = random.uniform(1, 3)
    f1 = 1640.2823 + 2.3573285 * x1 + 2.3220035 * x2 + 4.5688768 * x3 + 7.7213633 * x4 + 4.4559504 * x5
    f2 = 6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4 + \
         0.0861 * x1 * x5 + 0.3628 * x2 * x4 - 0.1106 * x1 * x1 - 0.3437 * x3 * x3 + 0.1764 * x4 * x4
    f3 = -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.024 * x2 * x3 - 0.0118 * x2 * x4 \
         - 0.0204 * x3 * x4 - 0.008 * x3 * x5 - 0.0241 * x2 * x3 + 0.0109 * x4 * x4
    data['x1'].append(x1)
    data['x2'].append(x2)
    data['x3'].append(x3)
    data['x4'].append(x4)
    data['x5'].append(x5)
    data['f1'].append(f1)
    data['f2'].append(f2)
    data['f3'].append(f3)
df = pd.DataFrame(data)
target_column = ['f1', 'f2', 'f3']
predictors = ['x1', 'x2', 'x3', 'x4', 'x5']

# normalize data
predictors_max_values = df[predictors].max().values
predictors_min_values = df[predictors].min().values
df[predictors] = (df[predictors]-predictors_min_values)/(predictors_max_values-predictors_min_values)
target_column_max_values = df[target_column].max().values
target_column_min_values = df[target_column].min().values
df[target_column] = (df[target_column]-target_column_min_values)/(target_column_max_values - target_column_min_values)
X_train, X_test, Y_train, Y_test = train_test_split(df[predictors], df[target_column], test_size=0.25)
store1 = HDFStore('data_normalized.h5')
store1["X_train"] = X_train
store1["X_test"] = X_test
store1["Y_train"] = Y_train
store1["Y_test"] = Y_test
store1["X_max"] = df[predictors].max()
store1["X_min"] = df[predictors].min()
store1["Y_max"] = df[target_column].max()
store1["Y_min"] = df[target_column].min()

print('')
