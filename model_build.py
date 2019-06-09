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

df = pd.read_csv("./ecoli_proteome_features.csv")

# Final data set shape
print(df.shape)

# Number of features
print(len(list(df)[3:]))

# We have a total of ~ 2000 observations.
# We should make sure our features are ~10x less than number of observations.
#

# Check to see which features change the least between classifications
for feature in list(df)[3:]:
    #df.boxplot(column=feature, by="Local")
    sns.boxplot(x="Local", y=feature, data=df)
    plt.savefig("./" + feature + ".png")
    plt.clf()

# Based on the above, we can safely eliminate U, X, A,
df = df.drop(["U", "X", "A"], axis=1)

# Check for multi-colinearity
# Features start at column 3
plt.figure(figsize=(12,10))
cor = df[df.columns[3:]].corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.savefig("./correlation_matrix.png")

# Drop colinear features with abs(r) > 0.5
df = df.drop(["ss_mean", "hydro_mean"], axis=1)

# Final number of features is now 10x less than number of observations
print(len(list(df)[3:]))

## Ok Let's start testing some models

# Shuffle training set
df = shuffle(df, random_state=0)

# Convert to np array
array = df.values

# Features
X = array[:, 3:]
X = X.astype("float")

# Classification
Y = array[:, 2]
Y = Y.astype("int")

# Size of validation set
validation_size = 0.40
seed = 7

# Test options and evaluation metric
scoring = 'accuracy'

# Set up train and validation sets
X_train, X_validation, Y_train, Y_validation = \
    model_selection.train_test_split(X, Y, test_size=validation_size,
                                     random_state=seed)

# Try standardization
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_validation = sc.transform(X_validation)


# Try different models
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(("RF", RandomForestClassifier(n_estimators=100, random_state=0)))
models.append(("GB", GradientBoostingClassifier(n_estimators=100, random_state=0)))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))

# Evaluate each model in turn
results = []
names = []
for name, model in models:
    # 10x cross validation
    kfold = model_selection.KFold(n_splits=10, random_state=seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train,
                                                 cv=kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg = (name, cv_results.mean(), cv_results.std())
    print(msg)

"""
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
"""