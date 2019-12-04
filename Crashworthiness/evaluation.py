import pandas as pd
from matplotlib import pyplot as plt
from keras.models import Model, load_model
from keras.layers import Dense, Input
from keras.optimizers import Adam, Adadelta
from pandas import HDFStore
from sklearn.metrics import r2_score
# apply fixed random seed 7
import numpy as np
import random
np.random.seed(7)
store = HDFStore('data_normalized.h5')
X_max = store["X_max"].values
X_min = store["X_min"].values
Y_max = store["Y_max"].values
Y_min = store["Y_min"].values
X_train = store["X_train"]
X_test = store["X_test"]
Y_train = store["Y_train"]
Y_test = store["Y_test"]
model_tandem = load_model("tandem_NN_normalized.h5")
intermediate_layer_model = Model(inputs=model_tandem.input, outputs=model_tandem.get_layer('intermediate').output)

def calculate_Y(x1, x2, x3, x4, x5):
     f1 = 1640.2823 + 2.3573285 * x1 + 2.3220035 * x2 + 4.5688768 * x3 + 7.7213633 * x4 + 4.4559504 * x5
     f2 = 6.5856 + 1.15 * x1 - 1.0427 * x2 + 0.9738 * x3 + 0.8364 * x4 - 0.3695 * x1 * x4 + \
          0.0861 * x1 * x5 + 0.3628 * x2 * x4 - 0.1106 * x1 * x1 - 0.3437 * x3 * x3 + 0.1764 * x4 * x4
     f3 = -0.0551 + 0.0181 * x1 + 0.1024 * x2 + 0.0421 * x3 - 0.0073 * x1 * x2 + 0.024 * x2 * x3 - 0.0118 * x2 * x4 \
          - 0.0204 * x3 * x4 - 0.008 * x3 * x5 - 0.0241 * x2 * x3 + 0.0109 * x4 * x4
     return [f1, f2, f3]

def calculate_mse(Y_input, res):
     X_pred = intermediate_layer_model.predict(Y_input)
     X_pred_original = X_pred * (X_max - X_min) + X_min
     Y_cal = {'f1': [], 'f2': [], 'f3': []}
     for row in X_pred_original:
          Y = calculate_Y(row[0], row[1], row[2], row[3], row[4])
          Y_cal['f1'].append(Y[0])
          Y_cal['f2'].append(Y[1])
          Y_cal['f3'].append(Y[2])
     Y_cal = pd.DataFrame(Y_cal)
     Y_cal_normalized = (Y_cal - Y_min)/ (Y_max - Y_min)
     score = np.square(np.subtract(Y_cal_normalized.values, Y_input.values)).mean()
     Y_input = Y_input.values
     Y_cal_normalized = Y_cal_normalized.values
     res['f1'].append(Y_input[0][0])
     res['f2'].append(Y_input[0][1])
     res['f3'].append(Y_input[0][2])
     res['f1_cal'].append(Y_cal_normalized[0][0])
     res['f2_cal'].append(Y_cal_normalized[0][1])
     res['f3_cal'].append(Y_cal_normalized[0][2])
     res['f1_diff'].append(abs(Y_cal_normalized[0][0]-Y_input[0][0]))
     res['f2_diff'].append(abs(Y_cal_normalized[0][1]-Y_input[0][1]))
     res['f3_diff'].append(abs(Y_cal_normalized[0][2]-Y_input[0][2]))
     return score

def append_Y(f1, f2, f3, data):
     data['f1'].append(f1)
     data['f2'].append(f2)
     data['f3'].append(f3)
     return data

def generate_margin_input(range, num_of_instance):
     if range>0.5:
          print('range must less than 0.5!!')
          return
     data = {'f1': [], 'f2': [], 'f3': []}
     count = 0
     while (count < num_of_instance):
          f1 = random.uniform(0, range)
          f2 = random.uniform(0, range)
          f3 = random.uniform(0, range)
          data = append_Y(f1, f2, f3, data)
          count += 1
     data = pd.DataFrame(data)
     return data

# calculate R2 score of Y_input and Y_cal
def calculate_R2(Y_input):
     Y_input_original = Y_input * (Y_max - Y_min) + Y_min
     X_pred = intermediate_layer_model.predict(Y_input)
     X_pred_original = X_pred * (X_max - X_min) + X_min
     Y_cal = {'f1': [], 'f2': [], 'f3': []}
     for row in X_pred_original:
          Y = calculate_Y(row[0], row[1], row[2], row[3], row[4])
          Y_cal['f1'].append(Y[0])
          Y_cal['f2'].append(Y[1])
          Y_cal['f3'].append(Y[2])
     Y_cal = pd.DataFrame(Y_cal)
     return r2_score(Y_input_original, Y_cal)


# divide input data along each dimension and calculate R2 for each part
def divide_data(division_num, num_of_instances):
     result = []
     if((not isinstance(division_num, int)) or (division_num < 1) or (not isinstance(num_of_instances, int))or (num_of_instances<1)):
          print("division_num must be a positive integer!")
          return
     subdiv_range = 1/division_num
     half_subdiv_range = subdiv_range/2
     for i in range(0, division_num):
          f1_min = i/division_num
          for i in range(0, division_num):
               f2_min = i / division_num
               for i in range(0, division_num):
                    f3_min = i / division_num
                    # calculate R2 of generated instances within current subdivision
                    print("===========================================================")
                    print([f1_min+half_subdiv_range, f2_min+half_subdiv_range, f3_min+half_subdiv_range])
                    data = {'f1': [], 'f2': [], 'f3': []}
                    count = 0
                    while (count < num_of_instances):
                         f1 = random.uniform(f1_min, f1_min+subdiv_range)
                         f2 = random.uniform(f2_min, f2_min+subdiv_range)
                         f3 = random.uniform(f3_min, f3_min+subdiv_range)
                         data['f1'].append(f1)
                         data['f2'].append(f2)
                         data['f3'].append(f3)
                         # print([f1, f2, f3])
                         count+=1
                    Y_input = pd.DataFrame(data)
                    R2 = calculate_R2(Y_input)
                    print("R2: "+ str(R2))
                    print("===========================================================\n\n")
                    result.append([f1_min+half_subdiv_range, f2_min+half_subdiv_range, f3_min+half_subdiv_range, R2])
     return result

# # calculate R2 of each subdivision
# division_num = 7
# num_of_instances = int(round(10000/(division_num**3)))
# subdivision_R2_scores = divide_data(division_num, num_of_instances)
# np.save("subdivision_R2_"+str(division_num), subdivision_R2_scores)
# print("\n"+str(num_of_instances))


# calculate MSE of each point
res = {'f1': [], 'f1_cal': [], 'f1_diff': [], 'f2': [], 'f2_cal': [], 'f2_diff': [],'f3': [], 'f3_cal': [], 'f3_diff': [],
       "mse": []
       }
Y_input = generate_margin_input(0.1, 1000)
# Y_input = Y_test
for i in range(len(Y_input)):
# for i in range(5):
     curr_Y = Y_input[i:i+1]
     mse = calculate_mse(curr_Y, res)
     res['mse'].append(mse)
res = pd.DataFrame(res)
score = HDFStore('mse_bad_prediction_corner.h5')
score['df'] = res
print(res)
print(calculate_R2(Y_input))