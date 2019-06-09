"""
Python 3.6

This script will extract numerical features from the sequence:
    - Amino acid count
    - Average hydrophobicity
    - Median hydrophobicity
    - Average helical propensity
    - Median helical propensity

by MAS 06.2019
"""

import time
import pandas as pd
import numpy as np

# Timer
start_time = time.time()

# Read in sequence data with classifications
DF = pd.read_csv("./ecoli_proteome.csv")

# Drop any data that does not have a classification
# There are no classifications if the subcellular localization is not
# annotated in the uniprot database
DF = DF.dropna(how="any")

# Dictionary of hydrophobicity scores
aa_hydro = {"I": 4.5,
            "V": 4.2,
            "L": 3.8,
            "F": 2.8,
            "C": 2.5,
            "M": 1.9,
            "A": 1.8,
            "G": -0.4,
            "T": -0.7,
            "S": -0.8,
            "W": -0.9,
            "Y": -1.3,
            "P": -1.6,
            "H": -3.2,
            "E": -3.5,
            "Q": -3.5,
            "D": -3.5,
            "N": -3.5,
            "K": -3.9,
            "R": -4.5,
            "X": 0,
            "U": 0
            }

# Dictionary of helical propensity scores
aa_secon = {"I": 0.97,
            "V": 0.91,
            "L": 1.3,
            "F": 1.07,
            "C": 1.11,
            "M": 1.47,
            "A": 1.29,
            "G": 0.56,
            "T": 0.82,
            "S": 0.82,
            "W": 0.99,
            "Y": 0.72,
            "P": 0.52,
            "H": 1.22,
            "E": 1.44,
            "Q": 1.27,
            "D": 1.040,
            "N": 0.9,
            "K": 1.23,
            "R": 0.96,
            "X": 0,
            "U": 0
            }

AAS = tuple(aa_hydro.keys())


def aa_count(aa, dataframe):
    scores = ["hydro_mean", "ss_mean"]
    aa = list(aa)
    aa.extend(scores)
    blank_df = pd.DataFrame(columns=aa)
    for index, row in dataframe.iterrows():
        counts = [row["Sequence"].count(a) / len(row["Sequence"])
                  for a in aa[:-len(scores)]]
        counts.extend(score(row["Sequence"]))
        blank_df.loc[index] = counts
    return blank_df


def score(sequence):
    hscore = [aa_hydro[res] for res in sequence]
    sscore = [aa_secon[res] for res in sequence]
    return [np.mean(hscore), np.mean(sscore)]


def get_all_features(aas=AAS, df=DF):
    temp_df = aa_count(aas, df)
    final_df = pd.concat([df, temp_df], axis=1)
    final_df.to_csv("./ecoli_proteome_features.csv", index=False)


if __name__ == '__main__':
    get_all_features()
    print('run time = ' + str(time.time() - start_time) + ' s')
