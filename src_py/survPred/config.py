import os

import numpy as np
import torch

from ccToolkits.MySummaryWriter import MySummaryWriter

#### inputs paths ####
proj_root = '/data/cHuang/HCC_proj/' # 

data_root_dict = {
    'img':os.path.join(proj_root, 'data_cleaned_CECT_annotations'),
    'nonImg':os.path.join(proj_root, 'data_cleaned_nonImage')
}
data_root_img = data_root_dict['img']
# 
imgs_dir = None # placeholder

#### outputs paths ####
# result path for saving models,test results,training log
# will add a subfolder res_[datetime] every time of excecution
result_root = '/data/cHuang/HCC_proj/results' 

########
folds_n = 5 # folds number for cross-validation
test_prob = 0.2
##
num_workers = 3 # number of workers for DataLoader of torch.

# 40, 200 was set as this is the mostly used wl and ww in batch3 dicom headers.
wl = 40 #55
ww = 200 # 250
norm_range = [0, 255]


#######
### tensorboard visualization
colorsList = ['#000000','#00FF00','#0000FF','#FF0000', '#FFFF00'] # hex color: Black, Green, Blue, Red, Yellow



#### global placeholder ####
writer = None
ckpt_dir = None
result_dir = None
log_dir = None
eval_out_dir = None
test_out_dir = None

pred_type = 'test'

clin_feats = None # place holder. will be given in main.py # ['tumorMaxDiameter','ALBI','multiple_tumor','cirrhosis', 'AFPgt400'] 

# data preparation settings
dila_iters_3D = 1
dilate_kernel_3D = np.ones((5, 5, 5), np.uint8)
liver_HU_range = [-100,250] # [-150,200] # [-200,200], [-100,250] # [-70,190] # 浙一图像常用的是两种窗宽窗位。一种是WL40，WW200；一种是WL60，WW260。这里取两个范围的并集以保留更多但不是特别多的像素值。# [-200,1000] #毛医生在标注时常用的窗宽是200，窗位是80~100。这里在依据上述几种窗宽窗位计算的最大HU范围后再扩大一点范围。


# experiment-level arguments. This is to replace the argparse usage.
class set_experiment_config(object):
    def __init__(self,
                 out_tag = 'E_recur', # output dir tag   
                 split = 'BFsplit', # 'BFsplit': all dev data for training. otherwise, int index of cross-validation np.random.uniform()fold splits
                 train = True, # whether to train
                 trainEval = True,
                 val = True, # whether to validate
                 test = True,# whether to test
                 infer = False, # wheter to infer new cases with no annotations
                 resume_ckp = '', # dir to load ckp for evaluation or re-training
                 resume_epoch = 0, # epoch of resume_ckp
                 aug_device = 'cpu', # image augmentation device:'gpu' or 'cpu,
                 input_src = 1,
                 addSegTask=False,
                 addSurvTask = False,
                 addTumorMask=False,
                 addLiverMask=False,
                 debug = False,
                 addClsTask=False,# if debug mode
                 imgs_dir=None,
                 ):
        self.out_tag = out_tag
        self.split = split

        self.train = train
        self.trainEval = trainEval
        self.val = val
        self.test = test
        self.infer = infer

        self.resume_ckp = resume_ckp
        self.resume_epoch = resume_epoch

        self.aug_device = aug_device
        # liver_xyzSpacing225, liver_xyDownSamp2, liver_xyDownSamp4
        if input_src=='liver_liverNoDilate_xyzMedian_tumorMaskByModel':
            self.patch_size = [48,352,480] # [48,384,512]#
            self.imgs_dir = os.path.join(data_root_img, 'liverROI_3d_liverNoDilate_xyzMedian_no_zscore_tumorMaskByModel/alligned')
        elif input_src == 'liver_liverNoDilate_xyzMedian_tumorMaskGT':
            self.patch_size = [48, 352, 480]  # [48,384,512]#
            self.imgs_dir = os.path.join(data_root_img,'liverROI_3d_liverNoDilate_xyzMedian_no_zscore_tumorMaskGT/alligned')

        elif input_src == 'tumor_liverNoDilate_xyzMedian_bbx':
            self.patch_size = [48, 80,80]  # [48,384,512]#
            self.imgs_dir = os.path.join(data_root_img,
                                         'tumorROI_3d_liverNoDilate_xyzMedian_no_zscore/alligned_bbx')

        elif input_src == 'tumor_liverNoDilate_xyzMedian_mask':
            self.patch_size = [48, 80,80]  # [48,384,512]#
            self.imgs_dir = os.path.join(data_root_img,
                                         'tumorROI_3d_liverNoDilate_xyzMedian_no_zscore/alligned')
            
        else:
            self.patch_size = [48, 256, 320]
            self.imgs_dir=imgs_dir
        self.addSegTask = addSegTask
        self.addSurvTask = addSurvTask

        self.debug = debug
        
        self.addTumorMask = addTumorMask
        self.addLiverMask = addLiverMask
        self.addClsTask= addClsTask
        
#### model level config ####
class set_model_config(object):
    def __init__(self,
                 model_name = 'LiverSurv', # 
                 task_names = ['recur'], # 
                 modality = ['ART', 'PV'],
                 numChannels = 2, 
                 nclass = 2,
                 batch_size = 6,
                #  patch_size=None, # [48,321,449]
                 nc = 64,
                 pre_train = False,
                 model_loc = None,
                 addClin = False, # will be deprecated later. will be totally replaced by clinFeats
                 clinFeats = ''
                ):
        self.model_name = model_name # 
        self.task_names = task_names
        self.modality = modality
        self.numChannels = numChannels ## number of input channels: 1 for gray image, 3 for 3 channels, 6 for DMP models
        self.nclass = nclass # number of classes
        self.batch_size = batch_size
        self.nc = nc # number of out channels for the first convolution? refer to xyy PA-ResSeg code.

        self.pre_train = pre_train
        self.model_loc = model_loc
        self.addClin = addClin
        self.clinFeats = clinFeats


## train/val/test level config
class set_train_config(object):
    def __init__(self,
                 step_per_epoch = 300, # iterations per epoch
                 epoch_method = 'infinite',
                 return_incomplete = False,
                 eval_num = 0,
                 start_trainEval_epoch = 50,
                 trainEval_epoch_interval = 2, # eval train set every # epochs
                 start_val_epoch = 50,
                 val_epoch_interval = 10, # val every # epochs.
                  # 3
                 start_test_epoch = 300,
                 test_epoch_interval = 20, # test every # epochs.

                 max_epoch = 1000, # 500, max training epochs # for debug
                 save_epoch = 5, # 50, save epoch interval

                 loss_type = 'ce',
                 trLoss_win = 20,
                 base_lr = 0.001, #0.001, 
                 final_lr = 1e-4,#1e-5 # 5e-6,1e-7：too small. results almost not changed after 5e-6
                 lrPatience = 3, # Patience for ReduceLROnPlateau()
                 lrScheduler = 'ReduceLROnPlateau', # CyclicLR
                 weight_decay = 0, #3e-5, #0.01 # to Adam, similar to but not same as l2 penalty # prevent overfitting
                 L1_reg_lambda = 'None',
                 L2_reg_lambda = '0.01',
                 post_processing = False,
                 use_gpu = True,
                 test_flip = False, # lead to CUDA OOM during test # TBD # Test time augmentation
                 multiTaskUncertainty = None # if apply loss based on Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR2018)
                ):
        debug_mode = False # True # 
        if debug_mode:
            self.step_per_epoch = 1
            self.eval_num = 0

            self.start_trainEval_epoch = 1
            self.trainEval_epoch_interval = 1

            self.start_val_epoch = 1
            self.val_epoch_interval = 1

            self.start_test_epoch = 1
            self.test_epoch_interval = 1
        else:
            # formal training
            self.step_per_epoch = step_per_epoch
            self.eval_num = eval_num
            
            self.start_trainEval_epoch = start_trainEval_epoch
            self.trainEval_epoch_interval = trainEval_epoch_interval

            self.start_val_epoch = start_val_epoch
            self.val_epoch_interval = val_epoch_interval

            self.start_test_epoch = start_test_epoch
            self.test_epoch_interval = test_epoch_interval

        self.epoch_method = epoch_method       
        self.return_incomplete = return_incomplete 

        self.max_epoch = max_epoch
        self.save_epoch = save_epoch

        self.loss_type = loss_type
        self.trLoss_win = trLoss_win
        self.base_lr = base_lr
        self.final_lr = final_lr
        self.lrPatience = lrPatience
        self.lrScheduler = lrScheduler
        self.weight_decay = weight_decay
        if L1_reg_lambda=='None':
            self.L1_reg_lambda = None
        else:
            self.L1_reg_lambda = float(L1_reg_lambda)
        # self.L1_reg_lambda = L1_reg_lambda
        if L2_reg_lambda=='None':
            self.L2_reg_lambda = None
        else:
            self.L2_reg_lambda = float(L2_reg_lambda)
        # self.L1_reg_lambda = L1_reg_lambda

        self.post_processing = post_processing
        self.use_gpu = use_gpu
        self.test_flip = test_flip

        self.multiTaskUncertainty = multiTaskUncertainty