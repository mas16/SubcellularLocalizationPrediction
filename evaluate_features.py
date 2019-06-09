"""
Python 3.6

This script will evaluate the features extracted from feature_extract.py
Specifically it will:
    - Generate box plots for all features (by classification)
    - Calculate correlations across all features to check for colinearity

Libraries:
    - Pandas
    - NumPy
    -seaborn

by MAS 06.2019
"""

import os
import pandas as pd
import seaborn as sns
import time
import pylab as plt

# Timer
start_time = time.time()


def mk_dir():
    """
    Function to make "plots" directory to put plots in.
    :return: None
    """
    path = os.getcwd()
    try:
        os.mkdir(path+"/plots/")
    # In case folder already made
    except OSError:
        pass


def fetch_csv():
    """
    Fetch csv data file from working directory.
    :return: dataframe if exits, None otherwise
    """
    # Read in sequence data with classifications
    try:
        df = pd.read_csv("./ecoli_proteome_features.csv")
        return df
    except FileNotFoundError:
        print("File not found.")


def gen_boxplots(df):
    """
    Generate boxplots by classification for each feature.
    Save box plots in "plots" directory
    :param df: dataframe of IDs, sequences, classifications, features
    :return: None
    """
    # Features start at column 3
    for feature in list(df)[3:]:
        sns.boxplot(x="Local", y=feature, data=df)
        plt.savefig("./plots/" + feature + ".png")
        plt.clf()


def check_colinearity(df):
    """
    Generate correlation matrix for all features. Save plot in
    "plots directory".
    :param df: dataframe of IDs, sequences, classifications, features
    :return: None
    """
    # Check for multi-colinearity
    # Features start at column 3
    plt.figure(figsize=(12, 10))
    cor = df[df.columns[3:]].corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.savefig("./plots/correlation_matrix.png")


def run():
    """
    Run functions if data exist
    :return:
    """
    df = fetch_csv()
    if df.all:
        df = df.dropna(how="any")
        gen_boxplots(df)
        check_colinearity(df)
    else:
        print("No Data!")


if __name__ == '__main__':
    run()
    print('run time = ' + str(time.time() - start_time) + ' s')
