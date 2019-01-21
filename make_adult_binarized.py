import pdb

import numpy as np

import gerryfair


full_dataset = True
centered = True
age_cutoff = 50
if full_dataset:
    dataset = "./dataset/full_adult.csv"
    attributes = "./dataset/adult_protected.csv"
    output_filename = './dataset/full_adult_binarized'
else:
    dataset = "./dataset/adult.csv"
    attributes = "./dataset/adult_protected.csv"
    output_filename = './dataset/adult_binarized'
X, A, sens_names, categorical_idx, continuous_idx, y \
        = gerryfair.clean.clean_dataset(dataset, attributes, centered, age_cutoff)
np.savez(output_filename, 
        x=X.values,
        columns=X.columns.tolist(),
        y=y.values,
        a=A.values,
        sens_names=sens_names,
        categorical_idx=categorical_idx,
        continuous_idx=continuous_idx,
        age_cutoff=age_cutoff)
print('done')
