# 
# (1) clean image data, save to target dir structure: data_clean/[images/labels]
# (2) generate data splits for training, validation, testing


import os
import sys
sys.path.append(os.getcwd()) # cuz config module is in the working dir
import json
import copy

import numpy as np
import pandas as pd

import survPred.config as config

root = os.path.dirname(os.getcwd())
# root="/HCC_new_std1/HCC_proj"
## file paths
surv_dir = os.path.join(root, 'data_cleaned_nonImage', 'survival')


folds_n = config.folds_n # folds number for cross-validation
test_prob = config.test_prob

#### 1. generate data splits ####
# get img IDs for only those have annotations
# patIDs_with_seg = [i.split('_A')[0] for i in os.listdir(liver_seg_dir) if '_A' in i]
b_tags = ['HUZHOU','TCGA_LIHC','SRRH']

for b_tag in b_tags:
    # b_tag = b_tags[0]
    cases_included_surv_f = os.path.join(surv_dir, '{}_cases_included.csv'.format(b_tag))

    cases_included_surv_df_all = pd.read_csv(cases_included_surv_f)
    cases_included_surv_df = cases_included_surv_df_all # [cases_included_surv_df_all['batch']=='b1to2']
    # pats_dev = cases_included_surv_df['pat_id'][cases_included_surv_df['split_group']=='dev'].values.tolist()
    # pats_test = cases_included_surv_df['pat_id'][cases_included_surv_df['split_group']=='test'].values.tolist()
    # patIDs = pats_dev + pats_test

    patIDs = cases_included_surv_df['pat_id'].values.tolist()

    patIDs.sort()
    print('total num:{}'.format(len(patIDs))) # 820

    ##------- save data splits indices to .json ------##
    data_splits = dict()
    # data_splits['test'] = patIDs[0:int(test_prob*len(patIDs))]
    pats_test = copy.deepcopy(patIDs)
    data_splits['test'] = pats_test
    print('test num:{}'.format(len(data_splits['test']))) # 178

    with open(os.path.join(surv_dir, 'data_splits_survPred_{}.json'.format(b_tag)), 'w') as f:
        json.dump(data_splits, f, indent=4)



# total num:25
# test num:25
# total num:16
# test num:16
# total num:292
# test num:292