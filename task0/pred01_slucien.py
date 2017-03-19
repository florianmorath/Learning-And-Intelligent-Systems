import csv
import pandas as pn
import numpy as np
from decimal import *
from sklearn.metrics import mean_squared_error
test = pn.read_csv('C:/Users/Work/Desktop/ethz/Semester6/LIS/project/task0/test.csv')
train = pn.read_csv('C:/Users/Work/Desktop/ethz/Semester6/LIS/project/task0/train.csv')


def meanRowTest(i):
    s = 0
    for xi in range(1,11):
        s += test.iloc[i][xi]
    return decimal(s / 10.0)

def meanRowTrain(i):
    s = 0
    for xi in range(2,12):
        s += train.iloc[i][xi]
    return s / 10.0


pred_train = []
for i in range(0,10000):
    pred_train.append(meanRowTrain(i))


pred_test = []
for i in range(0,2000):
    pred_test.append(meanRowTest(i))

y_train = train.iloc[:,[1]]
RMSE = mean_squared_error(y_train, pred_train)**0.5
print(RMSE)

id = []
for i in range (10000, 12000):
    id.append(i)

res = pn.DataFrame(columns = ['Id', 'y'])
res['Id'] = id
res['y'] = pred_test
res.to_csv("pred01_slucien.csv")
#df_train = pd.read_csv('Users/Work/Desktop/ethz/Semester6/LIS/project/task0)
'''
with open('C:/Users/Work/Desktop/ethz/Semester6/LIS/project/task0/train.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for line in readCSV:
        t = line[0],line[1]
        print(t)

with open('C:/Users/Work/Desktop/ethz/Semester6/LIS/project/task0/test.csv') as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for line in readCSV:
        test_t = line[0],line[1]
        print(t)

def meanRowTest(i):
    s = 0
    for xi in range(1,11):
        sum += test_t.row
#test = pd.csv_read('Users\Work\Desktop\ethz\Semester6\LIS\project\task0\test.csv')
#train = pd.csv_read('Users\Work\Desktop\ethz\Semester6\LIS\project\task0\train.csv')

#print(train)
#RMSE = mean_squared_error(y, y_pred)**0.5
'''
