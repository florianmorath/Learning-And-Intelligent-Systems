import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

print(df_train)

def meanOfRowTest(i):
    sum = 0
    for xi in range (1,11):
        sum += df_test.iloc[i][xi]
    return sum/10

def meanOfRowTrain(i):
    sum = 0
    for xi in range (2,12):
        sum += df_train.iloc[i][xi]
    return sum/10

# correct y-values of training data
y_train = df_train.iloc[:,[1]]

# predicted y-values on training data
y_pred_train = []
for i in range (0,10000):
    y_pred_train.append(meanOfRowTrain(i))

RMSE_train = mean_squared_error( y_train , y_pred_train ) ** 0.5
print('RMSE on training data = ', RMSE_train)

# predicted y-values on test data
y_pred = []
for i in range (0,2000):
    y_pred.append(meanOfRowTest(i))

id = []
for i in range(10000,12000):
    id.append(i)

df_pred = pd.DataFrame(columns=['Id', 'y'])
df_pred['y'] = y_pred
df_pred['Id'] = id
print(df_pred)
df_pred.to_csv('pred01_fmorath.csv', index = False)
