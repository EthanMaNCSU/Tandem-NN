# Electric Motor Temperature
import pandas as pd
from sklearn.model_selection import train_test_split
num_of_instance = -1
df = pd.read_csv('faults.csv')
if num_of_instance != -1:
    df = df.iloc[:num_of_instance]
target_column = ['Pastry', 'Z_Scratch', 'K_Scatch', 'Stains', 'Dirtiness', 'Bumps',	'Other_Faults']
predictors = list(set(list(df.columns))-set(target_column))
max_values = df[predictors].max().values
df[predictors] = df[predictors]/max_values
X = df[predictors].values
Y = df[target_column].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.30, random_state=40)