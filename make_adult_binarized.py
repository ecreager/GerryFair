import random
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

TRAIN_SPLIT = [0, 30975]
VALIDATION_SPLIT = [30976, 43959]
TEST_SPLIT = [43960, 48842]
shuffled_test_idx = list(range(TEST_SPLIT[0], TEST_SPLIT[1]))
random.shuffle(shuffled_test_idx)

idx = list(range(TRAIN_SPLIT[0], TRAIN_SPLIT[1]+1)) \
        + list(range(VALIDATION_SPLIT[0], VALIDATION_SPLIT[1]+1)) \
        + shuffled_test_idx
print(len(idx))

X = X.iloc[idx, :]
A = A.iloc[idx, :]
y = y.iloc[idx]

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
        #cat_idx=cat_idx,
        cat_idx0=cat_idx[0],
        cat_idx1=cat_idx[1],
        cat_idx2=cat_idx[2],
        cat_idx3=cat_idx[3],
        cat_idx4=cat_idx[4],
        cat_idx5=cat_idx[5],
        age_cutoff=age_cutoff)
print('done')
