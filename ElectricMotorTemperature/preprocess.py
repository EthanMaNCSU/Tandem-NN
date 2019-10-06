# Electric Motor Temperature
import pandas as pd
from sklearn.model_selection import train_test_split
num_of_instance = 3000
data = pd.read_csv('pmsm_temperature_data.csv')
if num_of_instance == -1:
    data = data.loc[data['profile_id'] == 57]
else:
    data = data.loc[data['profile_id'] == 57].iloc[:num_of_instance]
attr_for_X = ['ambient', 'coolant', 'u_d', 'u_q', 'i_d', 'i_q', 'motor_speed']
attr_for_Y = ['torque', 'pm', 'stator_yoke', 'stator_tooth', 'stator_winding']
X = data.loc[:, attr_for_X]
Y = data.loc[:, attr_for_Y]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25)