import pandas as pd
from sklearn.model_selection import train_test_split
import random
# generate data set

num_of_instance = 4000
count = 0
data = {'x1':[], 'x2':[], 'x3':[], 'x4':[], 'x5':[], 'f1':[], 'f2':[], 'f3':[]}
while (count < num_of_instance):
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
max_values = df[predictors].max().values
df[predictors] = df[predictors]/max_values
X = df[predictors].values
Y = df[target_column].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
