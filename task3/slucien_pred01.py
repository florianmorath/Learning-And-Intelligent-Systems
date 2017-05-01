import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier

#read in data hd5 files
df_train = pd.read_hdf("train.h5", "train", float_precision='round_trip')
df_test = pd.read_hdf("test.h5", "test", float_precision='round_trip')

'''
# read in data from csv files
df_train = pd.read_csv('train.csv', float_precision = 'round_trip')
df_test = pd.read_csv('test.csv', float_precision = 'round_trip')
'''

# prepare feature matrix X and response vector y
feature_cols = []
for i in range(1,101):
    feature_cols.append('x' + str(i))

#feature_cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15']
y_train = df_train['y']
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

#build the classifier

alphas = 10 ** np.linspace(10,-5,100) * 0.5
param_grid = [{'alpha': alphas}]

'''
# feature selection
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif,chi2,mutual_info_classif
sel = SelectKBest(f_classif, k = 11)
'''

#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes(5, 2), random_state=1)
clf = MLPClassifier(alpha=1.4240179342178966e-05, activation = 'logistic')
'''
sel_clf = make_pipeline(sel, clf)

from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator = clf, param_grid =  param_grid, cv = 10, scoring = 'accuracy')
grid_search.fit(X_train, y_train)
print('grid search best params = ', grid_search.best_params_)
'''
clf.fit(X_train, y_train)


# print out cross validation mean score for the chosen model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, X_train, y_train, cv = 10, scoring = 'accuracy')
print('cross val mean score = ', scores.mean())


# print accuracy
from sklearn.metrics import accuracy_score
accuracy_train = accuracy_score(y_train, clf.predict(X_train))
print('Accuracy on training data = ', accuracy_train)

# predict values on test data
y_test = clf.predict(X_test)

# write predictions to csv file
id = []
for i in range(45324, 53461):
    id.append(i)

df_pred = pd.DataFrame(columns=['Id', 'y'])
df_pred['y'] = y_test
df_pred['Id'] = id
df_pred.to_csv('pred01_slucien.csv', index = False)
