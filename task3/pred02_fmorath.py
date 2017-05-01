import pandas as pd
import numpy as np
import csv

# read in data from csv files
df_train = pd.read_hdf('train.h5')
df_test = pd.read_hdf('test.h5')

# prepare feature matrix X and response vector y
feature_cols = []
for x in range(1,101):
    feature_cols.append('x%i' % (x))

y_train = df_train['y']
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

# build the NN
from sknn.mlp import Classifier, Layer

nn = Classifier(
	layers = [ Layer('Rectifier', units = 500), Layer('Rectifier', units = 50), Layer('Softmax', units = 5)],
	learning_rate = 0.03,
	n_iter = 70,
	batch_size = 10
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('neural network', nn)
])

pipeline.fit(X_train, y_train)

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

 # print accuracy
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, pipeline.predict(X_train))
print('Accuracy on training data = ', accuracy_train)

# print out cross validation mean score for the chosen model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(pipeline, X_train, y_train, cv = 10, scoring = 'accuracy')
print('cross val mean score = ', scores.mean())


# predict values on test data
prediction = nn.predict(df_test.as_matrix())

# write predictions to csv file
outfile = open('pred02_fmorath.csv','w')
writer = csv.writer(outfile)
writer.writerow(['Id', 'y'])
ids = df_test.ix[:, 0:1]
for i in range(prediction.shape[0]):
 	writer.writerow([i+45324, prediction[i][0]])
outfile.close()
