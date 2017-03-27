import pandas as pd
import numpy as np
import seaborn as sns

# read in data from csv files
df_train = pd.read_csv('train.csv', float_precision = 'round_trip')
df_test = pd.read_csv('test.csv', float_precision = 'round_trip')

# prepare feature matrix X and response vector y
feature_cols = ['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12','x13','x14','x15']
y_train = df_train['y']
X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

# visualize the relationship between the features and the response using scatterplots
#sns.pairplot(df_train, x_vars = feature_cols, y_vars = 'y', size = 7, aspect = 0.7)
#sns.plt.show()

# non-linear feature transformation
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree = 3)
phi_train = poly.fit_transform(X_train)
phi_test = poly.fit_transform(X_test)

# build the classifier
# alphas = 10 ** np.linspace(10,-5,100) * 0.5
alphas = np.linspace(280,290,100)
print('alphas = ', alphas)

# hyperparameter alpha selection with cross-validation
from sklearn.linear_model import Ridge, RidgeCV
ridgecv = RidgeCV(fit_intercept = True, cv = 10,
    alphas = alphas, scoring = 'neg_mean_squared_error', normalize = False)
ridgecv.fit(phi_train, y_train)

print('RidgeCV alpha = ', ridgecv.alpha_)

# build Ridge classifier with the chosen hyperparameter alpha and train it
ridge = Ridge(alpha = ridgecv.alpha_, copy_X = True, fit_intercept = True, max_iter = None,
    normalize = False, solver = 'auto', tol = 0.001)
ridge.fit(phi_train, y_train)

# print out cross validation mean score for the chosen model
from sklearn.model_selection import cross_val_score
scores = cross_val_score(ridge, phi_train, y_train, cv = 10)
print('cross val mean score = ', scores.mean())
print('cross val std (+/-) = ', scores.std() * 2)

# print training error
from sklearn.metrics import mean_squared_error
RMSE_train = mean_squared_error(y_train, ridge.predict(phi_train)) ** 0.5
print('RMSE on training data = ', RMSE_train)

# predict values on test data
y_test = ridge.predict(phi_test)

# write predictions to csv file
id = []
for i in range(900,2900):
    id.append(i)

df_pred = pd.DataFrame(columns=['Id', 'y'])
df_pred['y'] = y_test
df_pred['Id'] = id
df_pred.to_csv('pred03_fmorath.csv', index = False)


''' ordered according to public score (lowest first) '''

'''
public score = 16.1646519373
with alphas = 10 ** np.linspace(10,-5,100) * 0.5
ridgecv alpha =  266.83496156
cross val mean score =  0.920652272746
RMSE on training data =  7.47898298079
'''

'''
public score = 16.1649181817
with alphas = np.linspace(280,290,100)
RidgeCV alpha =  283.939393939
cross val mean score =  0.920654652416
cross val std (+/-) =  0.0308225860317
RMSE on training data =  7.57469179853
'''
