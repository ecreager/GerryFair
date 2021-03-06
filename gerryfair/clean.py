import argparse
import pdb

import numpy as np
import pandas as pd

def setup():
    parser = argparse.ArgumentParser(description='Fairness Data Cleaning')
    parser.add_argument('-n', '--name', type=str,
                        help='name of the to store the new datasets (Required)')
    parser.add_argument('-d', '--dataset', type=str,
                        help='name of the original dataset file (Required)')
    parser.add_argument('-a', '--attributes', type=str,
                        help='name of the file representing which attributes are protected (unprotected = 0, protected = 1, label = 2) (Required)')
    parser.add_argument('-c', '--centered', default=False, action='store_true', required=False,
                        help='Include this flag to determine whether data should be centered')
    args = parser.parse_args()
    return [args.name, args.dataset, args.attributes, args.centered]



'''
Clean a dataset, given the filename for the dataset and the filename for the attributes.

Dataset: Filename for dataset. The dataset should be formatted such that categorical variables use one-hot encoding
and the label should be 0/1

Attributes: Filename for the attributes of the dataset. The file should have each column name in a list, and under this
list should have 0 for an unprotected attribute, 1 for a protected attribute, and 2 for the attribute of the label.

'''
def clean_dataset(dataset, attributes, centered, age_cutoff):
    if not 'adult' in dataset and 'adult' in attributes:
        raise ValueError(
                'the purpose of this fork is to binarize the adult dataset' \
                + 'other functionality is not supported')
    df = pd.read_csv(dataset)
    sens_df = pd.read_csv(attributes)

    ## Get and remove label Y
    y_col = [str(c) for c in sens_df.columns if sens_df[c][0] == 2]
    print('label feature: {}'.format(y_col))
    if(len(y_col) > 1):
        raise ValueError('More than 1 label column used')
    if (len(y_col) < 1):
        raise ValueError('No label column used')

    y = df[y_col[0]]

    # handle case where last column is formatted as string not int
    if y_col[0] == 'income' and ' >50K' in set(y.values):
         y = pd.Series((y.values == ' >50K').astype(np.int_))

    ## Do not use labels in rest of data
    X = df.loc[:, df.columns != y_col[0]]
    X = X.loc[:, X.columns != 'Unnamed: 0']
    ## get protected attributes
    sens_cols = [str(c) for c in sens_df.columns if sens_df[c][0] == 1]
    print('sensitive features: {}'.format(sens_cols))
    sens_dict = {c: 1 if c in sens_cols else 0 for c in df.columns}
    X, sens_dict, categorical_cols \
            = one_hot_code(X, sens_dict)
    sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    print('there are {} sensitive features including derivative features'.format(len(sens_names)))

    continuous_cols = ['fnlwgt', 'eduction-num', 'capital-gain', 
            'capital-loss', 'hours-per-week']  # note education was misspelled in the csv

    # binarize age according to cutoff
    X['age'] = (X['age'] > age_cutoff).astype(np.int_)

    # binarize race by non-white/white; note that a=1 is non-white
    # "White" happens to be the last race alphabetically
    sens_name_maj_race = sorted([n for n in sens_names if 'race' in n])[-1]
    X['race'] = 1 - X[sens_name_maj_race]  # a=1 indicates minority race
    # get rid of old race attribute names
    for n in sens_names: 
        if 'race.' in n:
            del X[n]
    del categorical_cols['race']

    # split dataframe into sens/non-sens
    A = X.loc[:, sens_cols]
    for c in sens_cols:
        del X[c]

    categorical_idx = {}  # maps attr names to col indices
    for k, v in categorical_cols.items():
        categorical_idx[k] = [X.columns.tolist().index(vv) for vv in v]

    continuous_idx = {}  # maps attr names to col indices
    for c in continuous_cols:
        continuous_idx[c] = X.columns.tolist().index(c)

    # sex is already binarized; a=1 is female, a=0 is male

    if(centered):
        X = center(X, continuous_cols)

    #sens_idx = sorted([X.columns.tolist().index(c) for c in sens_cols])
    #sens_idx = [X.columns.tolist().index(c) for c in sens_cols]
    sens_names = sens_cols

    return X, A, sens_names, categorical_idx, continuous_idx, y  



def center(X, continuous_cols):
    for col in X.columns:
        if col in continuous_cols:
            X.loc[:, col] -= np.mean(X.loc[:, col])
            X.loc[:, col] /= X.loc[:, col].std()
            #X.loc[:, col] /= X.loc[:, col].abs().max()
    return X

def one_hot_code(df1, sens_dict):
    cols = df1.columns
    categorical_cols = {}  # maps attr names to one-hot column names
    _sens_names = [key for key in sens_dict.keys() if sens_dict[key] == 1]
    for c in cols:
        if isinstance(df1[c][0], str):
            column = df1[c]
            df1 = df1.drop(c, 1)
            unique_values = list(sorted(set(column)))
            n = len(unique_values)
            if n > 2:
                categorical_cols[c] = []
                for i in range(n):
                    col_name = '{}.{}'.format(c, i)
                    categorical_cols[c].append(col_name)
                    col_i = [1 if el == unique_values[i] else 0 for el in column]
                    df1[col_name] = col_i
                    sens_dict[col_name] = sens_dict[c]
                del sens_dict[c]
            else:
                col_name = c
                col = [1 if el == unique_values[0] else 0 for el in column]
                df1[col_name] = col

    return df1, sens_dict, categorical_cols


# Helper for main method
'''
Given name of dataset, load in the three datasets associated from the clean.py file
'''


def get_data(dataset):
    X = pd.read_csv('dataset/' + dataset + '_features.csv')
    X_prime = pd.read_csv('dataset/' + dataset + '_protectedfeatures.csv')
    y = pd.read_csv('dataset/' + dataset + '_labels.csv', names=['index', 'label'])
    y = y['label']
    return X, X_prime, y





