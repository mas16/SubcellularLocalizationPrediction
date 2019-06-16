"""
Python 3.6

Use scikit-learn to train a ML model to predict protein subcellular
localization exclusively from sequence features.

by MAS 06.2019
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

seed = 1234

df = pd.read_csv("./ecoli_proteome_features.csv")

# Final data set shape
print(df.shape)

# Number of features
print(len(list(df)[3:]))

# We have a total of ~ 2000 observations.
# We should make sure our features are ~10x less than number of observations.

# Based on the box plots, we can eliminate U, X, A,
df = df.drop(["U", "X", "A"], axis=1)

# Drop colinear features with abs(r) > 0.5
df = df.drop(["ss_mean", "hydro_mean"], axis=1)

## Number of soluble proteins
print("soluble:" , df[df["Local"]==0].shape)

## Number of membrane proteins
print("membrane:" , df[df["Local"]==1].shape)

## Little bit of a class imbalance
df = shuffle(df, random_state=seed)
balance_membrane_df = df[df["Local"]==1].iloc[0:874,]
balance_soluble_df = df[df["Local"]==0]
df = pd.concat([balance_membrane_df, balance_soluble_df])

##############

# Final number of features is now 10x less than number of observations
print(len(list(df)[3:]))

## Ok Let's start testing some models
# Shuffle data
df = shuffle(df, random_state=seed)

# Convert to np array
array = df.values

# Features
X = array[:, 3:]
X = X.astype("float")

# Classes
Y = array[:, 2]
Y = Y.astype("int")

# Size of test set
test_size = 0.40

# Test options and evaluation metric
scoring = 'accuracy'

# Set up train and validation sets
X_train, X_test, Y_train, Y_test = \
    model_selection.train_test_split(X, Y, test_size=test_size,
                                     random_state=seed)

# Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# Try different models
models = []
models.append(('LR', LogisticRegression(solver="lbfgs")))
models.append(('CART', DecisionTreeClassifier()))
models.append(("RF", RandomForestClassifier(n_estimators=100, random_state=seed)))
models.append(("GB", GradientBoostingClassifier(n_estimators=100, random_state=seed)))
models.append(('SVM', SVC(gamma="scale")))
#models.append(("XGBRF", xgb.XGBRFRegressor(random_state=seed)))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # 5x cross validation
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    print(name, cv_results.mean(), cv_results.std())



# SVM parameter tuning using 5 fold cross validation
kernels = ["rbf", "linear", "poly", "sigmoid"]
cv_results = []
results = []
nfolds = 5
for kernel in kernels:
    Cs = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]
    gammas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1]
    param_grid = {'C': Cs, 'gamma': gammas}
    grid_search = GridSearchCV(SVC(kernel=kernel), param_grid, cv=nfolds)
    grid_search.fit(X_train, Y_train)
    c = grid_search.best_params_["C"]
    gamma = grid_search.best_params_["gamma"]
    kfold = model_selection.KFold(n_splits=5, random_state=seed)
    cv_results = model_selection.cross_val_score(SVC(kernel=kernel, C=c, gamma=gamma), X_train, Y_train, cv=kfold, scoring=scoring)
    results.append(cv_results)
    print(kernel)
    print(grid_search.best_params_)
    print(cv_results.mean(), cv_results.std())


# Apply final model to validation set
classifier = SVC(kernel="rbf", C=5, gamma=0.1)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
print(accuracy_score(Y_test, Y_pred))
