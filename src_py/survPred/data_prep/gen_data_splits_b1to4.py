# 
# (1) clean image data, save to target dir structure: data_clean/[images/labels]
# (2) generate data splits for training, validation, testing


import os
import sys
sys.path.append(os.getcwd()) # cuz config module is in the working dir
import json

import numpy as np
import pandas as pd

import survPred.config as config

root = os.path.dirname(os.getcwd())

## file paths
surv_dir = os.path.join(root, 'data_cleaned_nonImage', 'survival')
cases_included_surv_f = os.path.join(surv_dir, 'cases_included_random_split_by_recur_death_surgeryYear.csv')

folds_n = config.folds_n # folds number for cross-validation
test_prob = config.test_prob

#### 1. generate data splits ####
# get img IDs for only those have annotations
# patIDs_with_seg = [i.split('_A')[0] for i in os.listdir(liver_seg_dir) if '_A' in i]
b_tag = 'b1to4'
cases_included_surv_df_all = pd.read_csv(cases_included_surv_f)
cases_included_surv_df = cases_included_surv_df_all # [cases_included_surv_df_all['batch']=='b1to2']
pats_dev = cases_included_surv_df['pat_id'][cases_included_surv_df['split_group']=='dev'].values.tolist()
pats_test = cases_included_surv_df['pat_id'][cases_included_surv_df['split_group']=='test'].values.tolist()
patIDs = pats_dev + pats_test

patIDs.sort()
print('total num:{}'.format(len(patIDs))) # 820

##------- save data splits indices to .json ------##
data_splits = dict()
np.random.seed(99)
np.random.shuffle(patIDs)
# data_splits['test'] = patIDs[0:int(test_prob*len(patIDs))]
data_splits['test'] = pats_test
print('test num:{}'.format(len(data_splits['test']))) # 178

dev = dict()
# devIDs = [i for i in patIDs if i not in data_splits['test']]
devIDs = pats_dev
dev['foldBFsplit'] = dict()
dev['foldBFsplit']['train'] = devIDs
dev['foldBFsplit']['val'] = devIDs

# split 'dev' data to one time CV folds
# np.random.seed(99)
# np.random.shuffle(devIDs)
for fold in range(folds_n):
    split_tag = 'fold{}'.format(fold)
    dev[split_tag] = dict()
    val_ids = cases_included_surv_df['pat_id'][cases_included_surv_df['dev_CV_folds']=='Fold{}'.format(fold+1)].values.tolist() # sorted([x for i,x in enumerate(devIDs) if i%folds_n != fold])
    train_ids = sorted([x for i,x in enumerate(devIDs) if x not in val_ids])
    print('fold{}: train num, {}; val num, {}'.format(fold, len(train_ids), len(val_ids)))
    dev[split_tag]['val'] = val_ids
    dev[split_tag]['train'] = train_ids

# 20 times random non-replaced resampling
for i in range(20):
    split_tag = 'Resample{}'.format(str(i+1).zfill(2))
    dev[split_tag] = dict()
    val_ids = cases_included_surv_df['pat_id'][cases_included_surv_df[split_tag]=='val'].values.tolist()
    train_ids = cases_included_surv_df['pat_id'][cases_included_surv_df[split_tag]=='train'].values.tolist()
    print('{}: train num, {}; val num, {}'.format(split_tag, len(train_ids), len(val_ids)))
    dev[split_tag]['val'] = val_ids
    dev[split_tag]['train'] = train_ids

# multiple CV folds
CV_times = 5# 4
CV_k = 4#5
for i in range(CV_times):
    for j in range(CV_k):
        split_tag = 'mCVsFold{}.Rep{}'.format(j+1,i+1)
        dev[split_tag] = dict()
        val_ids = cases_included_surv_df['pat_id'][cases_included_surv_df[split_tag]=='val'].values.tolist()
        train_ids = cases_included_surv_df['pat_id'][cases_included_surv_df[split_tag]=='train'].values.tolist()
        print('{}: train num, {}; val num, {}'.format(split_tag, len(train_ids), len(val_ids))) # train:510~520, val:122~132
        dev[split_tag]['val'] = val_ids
        dev[split_tag]['train'] = train_ids

#
data_splits['dev'] = dev

with open(os.path.join(surv_dir, 'data_splits_survPred_{}.json'.format(b_tag)), 'w') as f:
    json.dump(data_splits, f, indent=4)

### 5-fold CV ####
# total num:795
# test num:147
# fold0: train num, 517; val num, 131
# fold1: train num, 517; val num, 131
# fold2: train num, 519; val num, 129
# fold3: train num, 520; val num, 128
# fold4: train num, 519; val num, 129
# Resample01: train num, 529; val num, 119
# Resample02: train num, 529; val num, 119
# Resample03: train num, 529; val num, 119
# Resample04: train num, 529; val num, 119
# Resample05: train num, 529; val num, 119
# Resample06: train num, 529; val num, 119
# Resample07: train num, 529; val num, 119
# Resample08: train num, 529; val num, 119
# Resample09: train num, 529; val num, 119
# Resample10: train num, 529; val num, 119
# Resample11: train num, 529; val num, 119
# Resample12: train num, 529; val num, 119
# Resample13: train num, 529; val num, 119
# Resample14: train num, 529; val num, 119
# Resample15: train num, 529; val num, 119
# Resample16: train num, 529; val num, 119
# Resample17: train num, 529; val num, 119
# Resample18: train num, 529; val num, 119
# Resample19: train num, 529; val num, 119
# Resample20: train num, 529; val num, 119
# mCVsFold1.Rep1: train num, 517; val num, 131
# mCVsFold2.Rep1: train num, 517; val num, 131
# mCVsFold3.Rep1: train num, 519; val num, 129
# mCVsFold4.Rep1: train num, 520; val num, 128
# mCVsFold5.Rep1: train num, 519; val num, 129
# mCVsFold1.Rep2: train num, 521; val num, 127
# mCVsFold2.Rep2: train num, 515; val num, 133
# mCVsFold3.Rep2: train num, 522; val num, 126
# mCVsFold4.Rep2: train num, 520; val num, 128
# mCVsFold5.Rep2: train num, 514; val num, 134
# mCVsFold1.Rep3: train num, 520; val num, 128
# mCVsFold2.Rep3: train num, 517; val num, 131
# mCVsFold3.Rep3: train num, 518; val num, 130
# mCVsFold4.Rep3: train num, 520; val num, 128
# mCVsFold5.Rep3: train num, 517; val num, 131
# mCVsFold1.Rep4: train num, 517; val num, 131
# mCVsFold2.Rep4: train num, 517; val num, 131
# mCVsFold3.Rep4: train num, 519; val num, 129
# mCVsFold4.Rep4: train num, 520; val num, 128
# mCVsFold5.Rep4: train num, 519; val num, 129



### 4-fold CV ####
# total num:795
# test num:187
# mCVsFold1.Rep1: train num, 454; val num, 154
# mCVsFold2.Rep1: train num, 456; val num, 152
# mCVsFold3.Rep1: train num, 457; val num, 151
# mCVsFold4.Rep1: train num, 457; val num, 151
# mCVsFold1.Rep2: train num, 460; val num, 148
# mCVsFold2.Rep2: train num, 455; val num, 153
# mCVsFold3.Rep2: train num, 457; val num, 151
# mCVsFold4.Rep2: train num, 452; val num, 156
# mCVsFold1.Rep3: train num, 454; val num, 154
# mCVsFold2.Rep3: train num, 457; val num, 151
# mCVsFold3.Rep3: train num, 456; val num, 152
# mCVsFold4.Rep3: train num, 457; val num, 151
# mCVsFold1.Rep4: train num, 458; val num, 150
# mCVsFold2.Rep4: train num, 456; val num, 152
# mCVsFold3.Rep4: train num, 454; val num, 154
# mCVsFold4.Rep4: train num, 456; val num, 152
# mCVsFold1.Rep5: train num, 456; val num, 152
# mCVsFold2.Rep5: train num, 462; val num, 146
# mCVsFold3.Rep5: train num, 454; val num, 154
# mCVsFold4.Rep5: train num, 452; val num, 156