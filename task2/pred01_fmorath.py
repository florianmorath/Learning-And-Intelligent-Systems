import pandas as pd
import numpy as np

# read in data from csv files
df_train = pd.read_csv('train.csv', float_precision = 'round_trip')
df_test = pd.read_csv('test.csv', float_precision = 'round_trip')

# prepare feature matrix X and response vector y
feature_cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15']
y_train = df_train['y']
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

# build the classifier

# feature selection
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
sel = SelectKBest(f_classif, k = 11)

# SVC
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#param_grid = {'gamma': np.linspace(0.008, 0.009, 1000)}
clf = svm.SVC(kernel = 'rbf', gamma = 0.0085175175175175172, C = 1.9)
sel_clf = make_pipeline(sel, clf)
# grid_search = GridSearchCV(estimator = sel_clf, param_grid =  param_grid, cv = 10, scoring = 'accuracy')
# grid_search.fit(X_train, y_train)
# print('grid search best params = ', grid_search.best_params_)

# train the classifier
sel_clf.fit(X_train, y_train)

# # print out cross validation mean score for the chosen model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(sel_clf, X_train, y_train, cv = 10, scoring = 'accuracy')
print('cross val mean score = ', scores.mean())

# # print accuracy
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, sel_clf.predict(X_train))
print('Accuracy on training data = ', accuracy_train)

# predict values on test data
y_test = sel_clf.predict(X_test)

# write predictions to csv file
id = []
for i in range(1000,4000):
    id.append(i)

df_pred = pd.DataFrame(columns=['Id', 'y'])
df_pred['y'] = y_test
df_pred['Id'] = id
df_pred.to_csv('pred01_fmorath.csv', index = False)
