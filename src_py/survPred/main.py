
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

import numpy as np
import torch
import torch.nn as nn

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
parser = argparse.ArgumentParser('set some args in the command line')
parser.add_argument('-date','--date_tag', required=False, type=str, default='surv_debug')
parser.add_argument('-batch','--batch_tag', required=False, type=str, default='b1to4')
parser.add_argument('-bs','--batch_size', required=False, type=int, default=16)
parser.add_argument('-model','--model_name', required=False, type=str, default='LiverSurv') # TumorSurv #LiverNet_segTx3Mmtm_xyzMedian_resizeToMedian
parser.add_argument('-t', '--task_names', nargs="+", required=False, default=['recur'], help='survival outcomes')
parser.add_argument('-split','--split', required=False, type=str, default='mCVsFold1.Rep1') # choices= fold0-4, foldBFsplit, Resample01-20
parser.add_argument('-lr','--learningRate', required=False, type=float, default=0.001)
parser.add_argument('-weight_decay','--weight_decay', required=False, type=float, default=0)
parser.add_argument('-l2','--l2Sigma', required=False, type=str, default='None') # 0.01 #'None'
parser.add_argument('-loss','--loss', required=False, type=str, default='Cox', choices=['Cox','CE']) # 0.01
parser.add_argument('-input_src','--input_src', required=False, type=str, default='liver_liverNoDilate_xyzMedian_tumorMaskByModel') #liver_zSpacing5_xyDownsamp4
parser.add_argument('-epochMethod','--epoch_method', required=False, type=str, default='finite') # infinite: each batch of samples is randomly sampled the whole training set, number of iterations should be pre-set. finite: samples from the training set are sample to form a batch in turn, almost all training cases will be chosen after an epoch.
parser.add_argument('-return_incomplete','--return_incomplete', required=False, type=bool, default=False)  # 注意 type 被设置为 bool 后，只要参数存在，无论其值如何，它都会始终返回True，不设置则为 None
parser.add_argument('-lrPatience','--lrPatience', required=False, type=int, default=10, help='learning Rate Reduce Patience')
parser.add_argument('-lrScheduler','--lrScheduler', required=False, type=str, default='ReduceLROnPlateau') # 'CyclicLR'
parser.add_argument('-max_epoch','--max_epoch', required=False, type=int, default=150)
parser.add_argument('-addSegTask','--addSegTask', required=False, type=int, default=0, choices=[0,1], help='0=False, 1=True')
parser.add_argument('-addSurvTask','--addSurvTask', required=False, type=int, default=0, choices=[0,1], help='0=False, 1=True')
parser.add_argument('-addClsTask','--addClsTask', required=False, type=int, default=0, choices=[0,1], help='0=False, 1=True')

parser.add_argument('-clinFeats', '--clinFeats', nargs="+", required=False, default=['ALBI','cirrhosis', 'AFPgt400']) # ['tumorMaxDiameter','ALBI','multiple_tumor','cirrhosis', 'AFPgt400'] ## ['ALBI','cirrhosis', 'AFPgt400']
parser.add_argument('-addClin','--addClin', required=False, type=int, default=0, choices=[0,1], help='0=False, 1=True')

parser.add_argument('-modality', '--modality', nargs="+", required=False, default=['ART','PV'])
parser.add_argument('-addTumorMask','--addTumorMask', required=False, type=int, default=0, choices=[0,1], help='0=False, 1=True')
parser.add_argument('-addLiverMask','--addLiverMask', required=False, type=int, default=0, choices=[0,1], help='0=False, 1=True')
parser.add_argument('-model_loc','--model_loc', required=False, type=str, default="") #需要retrain的模型
args = parser.parse_args()

#
args.addSegTask = bool(args.addSegTask)
args.addSurvTask = bool(args.addSurvTask)
args.addClsTask = bool(args.addClsTask)
args.addClin = bool(args.addClin)
args.addTumorMask = bool(args.addTumorMask)
args.addLiverMask = bool(args.addLiverMask)

#

batch_tag = args.batch_tag
batch_size = args.batch_size #30

# reset config parameters based on args.
config.clin_feats = args.clinFeats # when all model_config.clinFeats work, this line will be deprecated.

###### for debug #######
# args.addTumorMask = True
# args.addSurvTask = True
# args.addSegTask = False
# args.addClsTask = False
# args.addClin = True
# os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
# torch.cuda.set_device(0)
###### end of debug ######

#---------------------- settings for every experiment ------------------------#

expe_config = config.set_experiment_config(
    out_tag = '{}'.format(batch_tag), # output dir tag
    train = True, # whether to train
    trainEval = True,
    val = True, # whether to validate
    test = True,# whether to test
    split = args.split,
    input_src = args.input_src, # liver_xyNoDownSamp, liver_xyDownSamp2, liver_xyDownSamp4
    addSegTask = args.addSegTask,
    addSurvTask = args.addSurvTask,
    addLiverMask = args.addLiverMask,
    addTumorMask = args.addTumorMask,
    addClsTask=args.addClsTask,
    debug = False,
)

#------------------------ train config -------------------------#
train_config = config.set_train_config(
    step_per_epoch = 50,
    epoch_method = args.epoch_method,
    start_trainEval_epoch = 0,
    trainEval_epoch_interval= 1,
    start_val_epoch = 0,
    val_epoch_interval = 1, #10, # val every # epochs.
    # 3
    start_test_epoch = 0, #10,
    test_epoch_interval = 5, #10, # test every # epochs.

    max_epoch = args.max_epoch, # max training epochs # for debug
    save_epoch = 1,
    base_lr = args.learningRate, # 0.0005,
    weight_decay = args.weight_decay, 
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
    model_loc= args.model_loc, # '/data/cHuang/HCC_proj/results/res_surv_20221115_liverNet_4_deathb1to2_server38/checkpoint/epoch20.pth.tar'
    batch_size = args.batch_size,
    addClin = args.addClin,
    clinFeats = args.clinFeats
)

if len(model_config.task_names)==1:
    args.multiTaskUncertainty = 'Not multi task'

# renew out_dir
res_tag_addClin = 'addClin' if args.addClin else 'noClin'
config.result_dir = os.path.join(config.result_root, '_SEP_'.join(['res_{}'.format(args.date_tag),'bs{}'.format(args.batch_size), model_config.model_name, '_'.join(model_config.task_names), '{}'.format(expe_config.split), expe_config.out_tag, 'lr{}'.format(str(args.learningRate).replace('.','')), 'lrPatience{}'.format(args.lrPatience), 'l2is{}'.format(str(args.l2Sigma).replace('.','')), '_'.join(model_config.modality), res_tag_addClin]))
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

config.eval_out_dir = os.path.join(config.result_dir, "eval_out")
tinies.sureDir(config.eval_out_dir)

warnings.filterwarnings('ignore')


# print(DEVICE)
# model.to(DEVICE)  # send to GPU

if expe_config.train:
    ## train, can also val and/or test during training
    # train.train(expe_config, data_splits, dataset_config, model, model_config, train_config)
    # train.train(expe_config, data_splits, model, model_config, train_config)
    try:
        # instantialize model
        torch.manual_seed(1)
        model = get_model(model_config)
        logger.info(str(args))
        if args.addClin:
            logger.info('clin feats: {}'.format(str(expe_config.clinFeats)))
        else:
            logger.info('no clin')

        logger.info('Armed model: {}'.format(model_config.model_name))
        if args.addSegTask:
            logger.info('Will add tumor seg task')
        logger.info('lrScheduler:{}'.format(train_config.lrScheduler))
        # logger.info('addClin: {}'.format(model_config.addClin))
        # logger.info('addLiverMask: {}'.format(expe_config.addLiverMask))
        # logger.info('addTumorMask: {}'.format(expe_config.addTumorMask))
        #

        train.train(expe_config, data_splits, model, model_config, train_config)
    except Exception as e:
        logger.exception("Unexpected exception! %s",e)
else:
    pass

## predict with trained model
if expe_config.test:
    if model_config.model_loc is None:
        print('Please specify model path to predict')
    else:
        # test all train/val/test data?
        # test only test data?

        # predict.predict(data_splits, model, model_config, mode='test_offline')
        logger.info('Testing with ckpt from :{}'.format(str(model_config.model_loc)))
        ## predict on only test data
        cases = data_splits['test']
        # cases = data_splits['dev']['fold0']['train']
        # cases = data_splits['dev']['fold0']['val']
        # cases = data_splits['dev']['fold0']['train'] + data_splits['dev']['fold0']['val']

        # predict on all dev/val/test data
        # cases = data_splits['test'] + data_splits['dev']['fold0']['train'] + data_splits['dev']['fold0']['val']

        pred_out_dir = config.eval_out_dir

        config.pred_type = 'test'
        recur_cindex, death_cindex = predict.predict(expe_config, cases, model, model_config, pred_out_dir, mode='test_offline')
        # predict.predict(expe_config, cases, model, model_config, pred_out_dir, mode='infer')



