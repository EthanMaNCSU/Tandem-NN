# Electric Motor Temperature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
num_of_instance = 3000
data = pd.read_csv('pmsm_temperature_data.csv')
if num_of_instance == -1:
    data = data.loc[data['profile_id'] == 57]
else:
    data = data.loc[data['profile_id'] == 57].iloc[:num_of_instance]
attr_for_train = ['ambient', 'coolant', 'u_d', 'u_q', 'i_d', 'i_q', 'motor_speed']
attr_for_test = ['torque', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
X = data.loc[:, attr_for_train]
Y = data.loc[:, attr_for_test]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)
X_train_inv, X_test_inv, Y_train_inv, Y_test_inv = train_test_split(Y, X, test_size=0.25)