"""
Python 3.6

Use scikit-learn to train a ML model to predict protein subcellular
localization exclusively from sequence features.

by MAS 06.2019
"""

import pandas as pd
import numpy as np
from pandas.tools.plotting import scatter_matrix
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

# Shuffle training set
new_df = shuffle(new_df, random_state=0)

array = new_df.values

# Features
X = array[:, 3:]
X = X.astype("float")
# Classification
Y = array[:, 2]
Y = Y.astype("int")

# Size of validation set
validation_size = 0.40
seed = 7

# Set up train and validation sets
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size,
                                     random_state=seed)

# Try standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)


# Random Forests
n_vals = [10, 20, 30, 50, 100, 150, 200]
iterations = np.arange(0, 10)

for n in n_vals :
    scores = []
    for _ in iterations:
        classifier = RandomForestClassifier(n_estimators=n, random_state=0)
        classifier.fit(X_train, Y_train)
        Y_pred = classifier.predict(X_validation)
        scores.append(accuracy_score(Y_validation, Y_pred))
        #print(confusion_matrix(Y_validation,Y_pred))
        #print(classification_report(Y_validation,Y_pred))
        #print(accuracy_score(Y_validation, Y_pred))
    print(n)
    print(np.mean(scores))
    print(np.std(scores, ddof=1))

### MODEL ###
classifier = RandomForestClassifier(n_estimators=50, random_state=0)
classifier.fit(X_train, Y_train)
feature_imp = pd.Series(classifier.feature_importances_).sort_values(ascending=False)
print(feature_imp)

Y_pred = classifier.predict(X_validation)
print(confusion_matrix(Y_validation,Y_pred))
print(classification_report(Y_validation,Y_pred))
print(accuracy_score(Y_validation, Y_pred))