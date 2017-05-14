import pandas as pd
import numpy as np
import csv

# read in data from csv files
df_train = pd.read_hdf('train_labeled.h5')
df_test = pd.read_hdf('test.h5')

# prepare feature matrix X and response vector y
feature_cols = []
for x in range(1,129):
    feature_cols.append('x%i' % (x))

y_train = df_train['y']
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

# build the NN
from sknn.mlp import Classifier, Layer
from sknn.platform import gpu32

nn = Classifier(
	layers = [Layer('Rectifier', units = 1000), Layer('Rectifier', units = 400), Layer('Rectifier', units = 50), Layer('Softmax', units = 10)],
	learning_rate = 0.03,
	n_iter = 70,
	batch_size = 10,
	#learning_rule = 'momentum',
	#weight_decay = 0.0005,
	#regularize = 'L2' 
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

pipeline = Pipeline([
        ('scaler', MinMaxScaler()), # also try MinMaxScaler() and MaxAbsScaler() StandardScaler()
        ('neural network', nn)
])
'''
# print out cross validation mean score for the chosen model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
scores = cross_val_score(pipeline, X_train, y_train, cv = 10, scoring = 'accuracy', verbose = 10)
print('cross val mean score = ', scores.mean())
print('cross val deviation = ', scores.std() * 2)
'''

# from sklearn.model_selection import GridSearchCV
# param_grid = {
#     'neural network__learning_rate': [0.05, 0.01, 0.005, 0.001],
#     'neural network__hidden0__units': [4, 8, 12],
#     'neural network__hidden0__type': ["Rectifier", "Sigmoid", "Tanh"]
# }
# grid_search = GridSearchCV(estimator = pipeline, param_grid =  param_grid, cv = 10, scoring = 'accuracy')
# grid_search.fit(X_train, y_train)
# #print('available parameters = ', pipeline.get_params().keys())
# print('grid search best params = ', grid_search.best_params_)

pipeline.fit(X_train, y_train)

 # print accuracy
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, pipeline.predict(X_train))
print('Accuracy on training data = ', accuracy_train)

# predict values on test data
prediction = pipeline.predict(df_test.as_matrix())

# write predictions to csv file
outfile = open('pred01_slucien.csv','w')
writer = csv.writer(outfile)
writer.writerow(['Id', 'y'])
for i in range(prediction.shape[0]):
 	writer.writerow([i+30000, prediction[i][0]])
outfile.close()
