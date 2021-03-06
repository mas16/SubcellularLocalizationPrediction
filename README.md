# Building and Tuning a Machine Learning Model to Predict Protein Subcellular Localization from Amino Acid Sequence
by MAS16, 2019

## Introduction
We now have access to an unprecedented amount of genomic information. However, harnessing the full potential of that information still requires a lot of slow and expensive experiments. In an ideal scenario, we would employ supervised machine learning to make predictions about biology using the data that's already been collected and readily available genomic information. Here, I build and tune a machine learning model to predict the subcellular localization of proteins based on amino acid sequences. The end result is a support vector machine (SVM) model that predicts soluble and membrane proteins with an out-of-sample prediction accuracy of 0.84, precision of 0.85, and recall of 0.84.

## Preprocessing Scripts
The data come from the proteome database of Uniprot for the bacterium *E.Coli*. The scripts for scraping, preprocessing, and feature engineering are:

```01_scrape_uniprot.py```  

```02_extract_features.py```  

```03_evaluate_features.py```  

## Model Building Script
The script used for model building is"
```04_model_build_tune.py```

## Model Building and Prediction Jupyter Notebook
To see how I built the model, tuned hyperparameters, and used it for prediction, see the ```05_model_build_tune_describe.ipynb``` notebook.
