import numpy as np

import gerryfair


dataset = "./dataset/adult.csv"
attributes = "./dataset/adult_protected.csv"
centered = True
age_cutoff = 50
X, sens_idx, y = gerryfair.clean.clean_dataset(dataset, attributes, centered, age_cutoff)
np.savez('./dataset/adult_binarized', 
        x=X.values, sens_idx=sens_idx, y=y.values)
print('done')
