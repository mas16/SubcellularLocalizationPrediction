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

# Set seed
SEED = 1234

def check_df():
    try:
        df = pd.read_csv("./ecoli_proteome_features.csv")
        # Based on the box plots, we can eliminate U, X, A,
        df = DF.drop(["U", "X", "A"], axis=1)
        # Drop colinear features with abs(r) > 0.5
        df = DF.drop(["ss_mean", "hydro_mean"], axis=1)
        return df
    except FileNotFoundError:
        print("Dataframe File Not Found")


def preprocess_df(df=DF, seed=SEED):
    df = shuffle(df, random_state=seed)
    # Convert to np array
    array = df.values
    # Features
    X = array[:, 3:]
    X = X.astype("float")
    # Classes
    Y = array[:, 2]
    Y = Y.astype("int")
