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
cat_names, cat_idx = [], []
for k, v in categorical_idx.items():
    cat_names.append(k)
    cat_idx.append(v)
cont_names, cont_idx = [], []
for k, v in continuous_idx.items():
    cont_names.append(k)
    cont_idx.append(v)
np.savez(output_filename, 
        x=X.values,
        columns=X.columns.tolist(),
        y=y.values,
        a=A.values,
        sens_names=sens_names,
        cont_names=cont_names,
        cont_idx=cont_idx,
        cat_names=cat_names,
        cat_idx=cat_idx,
        age_cutoff=age_cutoff)
print('done')
