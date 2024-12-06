
import os
import sys
sys.path.append(os.getcwd())
import json
import shutil
from datetime import datetime
import warnings
import copy
import argparse
import csv

############################# start for block same as main.py #########################
import numpy as np
import torch


from ccToolkits.MySummaryWriter import MySummaryWriter
from ccToolkits import logger
from ccToolkits import tinies

import survPred.config as config
from importlib import reload
reload(config)
from survPred.models.getModel import get_model
from survPred.training import train, predict

#
root = os.path.dirname(os.getcwd())
src_root = os.getcwd()
data_surv_dir = os.path.join(config.data_root_dict['nonImg'],'survival')
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
##
parser = argparse.ArgumentParser('set scome args in the command line')
parser.add_argument('-date','--date_tag', required=False, type=str, default='surv_debug')
parser.add_argument('-batch','--batch_tag', required=False, type=str, default='b1to4')
parser.add_argument('-bs','--batch_size', required=False, type=int, default=16)
parser.add_argument('-model','--model_name', required=False, type=str, default='LiverNet_SramFalse_zSpacing5_xyzMedian') # LiverNet_CSAM27_zSpacing5_xyDownsamp4
parser.add_argument('-t', '--task_names', nargs="+", required=False, default=['recur'])
parser.add_argument('-split','--split', required=False, type=str, default='mCVsFold4.Rep1') # choices= fold0-4, foldBFsplit, Resample01-20
parser.add_argument('-lr','--learningRate', required=False, type=float, default=0.001)
parser.add_argument('-l2','--l2Sigma', required=False, type=str, default='None') # 0.01 #'None'
parser.add_argument('-loss','--loss', required=False, type=str, default='Cox', choices=['Cox','CE']) # 0.01
parser.add_argument('-downSamp','--liver_xyDownSamp', required=False, type=str, default='liver_xyzSpacingMedian') #liver_zSpacing5_xyDownsamp4
parser.add_argument('-epochMethod','--epoch_method', required=False, type=str, default='finite') # infinite: each batch of samples is randomly sampled the whole training set, number of iterations should be pre-set. finite: samples from the training set are sample to form a batch in turn, almost all training cases will be chosen after an epoch.
parser.add_argument('-return_incomplete','--return_incomplete', required=False, type=bool, default=False)
parser.add_argument('-lrPatience','--lrPatience', required=False, type=int, default=2, help='learning Rate Reduce Patience')
parser.add_argument('-lrScheduler','--lrScheduler', required=False, type=str, default='ReduceLROnPlateau') # 'CyclicLR'
parser.add_argument('-max_epoch','--max_epoch', required=False, type=int, default=1000) 
parser.add_argument('-addSegTask','--addSegTask', required=False, type=bool, default=False)
parser.add_argument('-addClin','--addClin', required=False, type=bool, default=False)
parser.add_argument('-modality', '--modality', nargs="+", required=False, default=['ART','PV'])

args = parser.parse_args()

batch_tag = args.batch_tag
batch_size = args.batch_size #30
#---------------------- settings for every experiment ------------------------#

expe_config = config.set_experiment_config(
    out_tag = '{}'.format(batch_tag), # output dir tag
    train = True, # whether to train
    val = True, # whether to validate
    test = False,# whether to test
    split = args.split,
    input_src = args.liver_xyDownSamp, # liver_xyNoDownSamp, liver_xyDownSamp2, liver_xyDownSamp4
    addSegTask = args.addSegTask,
    debug = False
)

#------------------------ train config -------------------------#
train_config = config.set_train_config(
    step_per_epoch = 50,
    epoch_method = args.epoch_method,
    start_val_epoch = 0,
    val_epoch_interval = 1, #10, # val every # epochs.
    # 3
    # start_test_epoch = 0, #10,
    # test_epoch_interval = 1, #10, # test every # epochs.

    max_epoch = args.max_epoch, # max training epochs # for debug
    save_epoch = 1,
    base_lr = args.learningRate, # 0.0005,
    # L1_reg_lambda = 0.01
    lrPatience = args.lrPatience,
    lrScheduler = args.lrScheduler,
    L2_reg_lambda = args.l2Sigma, #0.01
    multiTaskUncertainty = 'Kendall',
    loss_type=args.loss
)

#------------------------- model config -------------------------#
model_config = config.set_model_config(
    model_name = args.model_name, #liverNet_3, DenseNet_Wang5
    task_names = args.task_names, # ['death'], # 'recur', 'death'
    modality = args.modality,
    model_loc= None, # '/data/cHuang/HCC_proj/results/res_surv_20221115_liverNet_4_deathb1to2_server38/checkpoint/epoch20.pth.tar'
    batch_size = args.batch_size,
    addClin = args.addClin
)

if len(model_config.task_names)==1:
    args.multiTaskUncertainty = 'Not multi task'

# renew out_dir
config.result_dir = os.path.join(config.result_root, '_SEP_'.join(['res_{}'.format(args.date_tag),'bs{}'.format(args.batch_size), model_config.model_name, '_'.join(model_config.task_names), '{}'.format(expe_config.split), expe_config.out_tag, 'lr{}'.format(str(args.learningRate).replace('.','')), 'lrPatience{}'.format(args.lrPatience), 'l2is{}'.format(str(args.l2Sigma).replace('.','')), '_'.join(model_config.modality)]))
tinies.sureDir(config.result_dir)
logger.info('Result dir: {}'.format(config.result_dir))

# renew ckpt_dir
config.ckpt_dir = os.path.join(config.result_dir, 'checkpoint')
tinies.sureDir(config.ckpt_dir)

# prep log_dir 
config.log_dir = os.path.join(config.result_dir, 'train_log')
if os.path.exists(config.log_dir):
    bkp_log_dir = config.log_dir + datetime.now().strftime('%m%d_%H%M%S')
    shutil.move(config.log_dir, bkp_log_dir)

# init log_dir
config.writer = MySummaryWriter(log_dir=config.log_dir) # creates log_dir and a tensorboard file named after 'events.out.tfevents.'
logger.set_logger_dir(os.path.join(config.log_dir, 'logger'), action="d") # creates 'logger' in log_dir, and create file 'log.log' in 'logger'.


##################################################################
logger.info('task name: {}'.format(model_config.task_names))

train_config.use_gpu = train_config.use_gpu and torch.cuda.is_available()


with open(os.path.join(data_surv_dir, 'data_splits_survPred_{}.json'.format(batch_tag)), mode='r') as f:
    data_splits = json.load(f)

# seed
np.random.seed(1993)

# dataset_config = config.set_dataset_config()

tinies.ForceCopyDir(src_root, os.path.join(config.result_dir, 'src_py'), ignore=shutil.ignore_patterns('*.pyc', '*.pth', '*git*')) # ,'tumorSurvPred'

config.val_out_dir = os.path.join(config.result_dir, "val_out")
tinies.sureDir(config.val_out_dir)
config.test_out_dir = os.path.join(config.result_dir, "test_out")
tinies.sureDir(config.test_out_dir)

############################# end for block same as main.py #########################
warnings.filterwarnings('ignore')



############################# predict with trained model #########################
# global settings
out_root = os.path.join(root, 'res_test')
tinies.sureDir(out_root)

### inputs ###
# same settings from xxx.sh file and are useful for prediction
expe_config.split = 'mCVsFold1.Rep2'
expe_config.patch_size = [48,352,480]
expe_config.imgs_dir = os.path.join(config.data_root_img, 'liverROI_3d_xyzSpacingMedian_no_zscore/alligned') # data_root_img

model_config.model_name = 'LiverNet_CsamMmtm_zSpacing5_xyzMedian'
model_config.task_names = ['recur']
model_config.modality = ['ART', 'PV']

# ckp src
model_config.model_loc = ''
model_config.addClin = False

### outputs ####
out_dir_tag = [i for i in model_config.model_loc.split('/') if i.startswith('res_surv_')][0]
pred_out_dir = os.path.join(out_root,out_dir_tag)
tinies.newDir(pred_out_dir)
out_f = os.path.join(pred_out_dir,'cindex_input_recur.csv')

##
pred_out_all_dict = dict()
pred_out_all_dict['split_tag'] = list()
pred_out_all_dict['pat_id'] = list()
pred_out_all_dict['status'] = list()
pred_out_all_dict['time'] = list()
pred_out_all_dict['model_logits'] = list()
if model_config.model_loc is None:
    print('Please specify model path to predict')
else:
    logger.info('Testing with ckpt from :{}'.format(str(model_config.model_loc)))
    # instantialize model
    torch.manual_seed(1)
    model = get_model(model_config)

    for split_tag in ['train','val','test']:
        # split_tag = 'train'
        if split_tag=='train':
            cases = data_splits['dev'][expe_config.split]['train']
        elif split_tag =='val':
            cases = data_splits['dev'][expe_config.split]['val']
        elif split_tag =='test':
            cases = data_splits['test']

        # cases = ['BA_003069232', 'BA_003914054', 'BA_003081078', 'BA_003095436', 'BA_003423038'] # BA_2000074323肿瘤大小在BA_003069232之后，但是缺失了随访数据。所以剔除。
        pred_out = predict.predict(expe_config, cases, model, model_config, pred_out_dir, mode='test_offline')

        pred_out_all_dict['split_tag'].extend([split_tag]*len(cases))
        pred_out_all_dict['pat_id'].extend(cases)
        pred_out_all_dict['status'].extend(pred_out['recur']['cindex_input']['status'])
        pred_out_all_dict['time'].extend(pred_out['recur']['cindex_input']['time'])
        pred_out_all_dict['model_logits'].extend(pred_out['recur']['cindex_input']['model_logits'])

###
with open(out_f,'w') as f:
    w = csv.writer(f)
    w.writerow(pred_out_all_dict.keys())
    w.writerows(zip(*pred_out_all_dict.values()))



