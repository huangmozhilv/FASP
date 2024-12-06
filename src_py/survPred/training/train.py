# for survival prediction
import os
import copy
import csv
import time
import math
from collections import deque

from unittest.mock import patch

import numpy as np
import pandas as pd
import tqdm
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from lifelines.utils import concordance_index

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CyclicLR
import torch.nn.functional as F

from ccToolkits.torchsummary import summary
import ccToolkits.logger as logger
from ccToolkits import tinies

from utils.progress_bar import ProgressBar
from utils.utils_train import l_reg

import survPred.config as config
from survPred.training import train_utils, predict
from survPred.surv_dataloader import survDataLoader, get_train_transform
# from loss.surv_loss import cox_loss_cox_nnet
# from loss.surv_loss import CoxPHLoss
# from loss.surv_loss import NegativeLogLikelihood
# from loss.surv_loss import cox_loss_Olivier

from loss.surv_loss import NegLogPartialLikelihood
from loss.lovasz_loss import lovasz_softmax
from loss.focal_loss import FocalLoss
from loss.joinclsloss import JointClsLoss
from loss.joinclsloss import JointClsLoss2
from loss.joinclsloss import JointClsLoss3
from loss.joinclsloss import JointClsLoss4
# from loss.multiTaskLoss import multiTaskUncertaintySurvLoss
from loss.multiTaskLoss import multiTaskUncertaintySurvLoss_sameAsPaper

def save_model(epoch, model, optimizer, train_epoch_loss, out_dir, save_for_infer=True, save_for_resume=False):
    # out_dir: config.ckpt_dir
    # save model for inference
    if save_for_infer:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict()
        },os.path.join(out_dir, 'epoch{}.pth.tar'.format(epoch)))
    else:
        pass

    # save model for resuming training in case of unexpected interrupt of the training process
    if save_for_resume: # 这种时候就保存能用于继续训练的，以防模型训练被意外中断。
        torch.save({
            'epoch': epoch,
            # 'model': model,
            'model_state_dict': model.state_dict(),
            # 'optimizer': optimizer,
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_epoch_loss
        },os.path.join(out_dir, 'ckp_to_resume.pth.tar'))
    else:
        pass
    #Chao测试了几种情况，发现，只保存model或model_state_dict或optimizer或optimizer_state_dict的存储空间占用大小跟如上保存6项内容的大小都是152M。后面发现原因是我这里测试用的是正式训练前的，而后来用了训练过程中的，发现保存上面4项跟保存6项的大小是一样的，但是只保存epoch和model_state_dict的2.97倍，是只保存epoch、optimizer_state_dict、loss的1.5倍。这里采用torch.save官方推荐的方法，保存epoch、model_state_dict、optimizer_state_dict和train_epoch_loss。


def train(expe_config, data_splits, model, model_config, train_config):
    # ## debug start
    # loss_lr_f = os.path.join(config.result_dir, 'epochs_loss_lr.csv')
    # loss_lr_fo = open(loss_lr_f, 'w')
    # loss_lr_wo = csv.writer(loss_lr_fo, delimiter=',')
    # loss_lr_wo.writerow(['epoch', 'step', 'lr', 'loss'])
    # loss_lr_fo.flush()
    # ## debug end

    torch.backends.cudnn.benchmark=True
    model = nn.parallel.DataParallel(model)
    if model_config.model_loc!="":
        ckpt_dir = model_config.model_loc
        ckpt = torch.load(ckpt_dir)
        try:
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in ckpt['model_state_dict'].items():
                name = k.split('module.')[1]  # remove 'module.' of dataparallel # don't use .strip as it will remove the leading or trailing characters same as in .strip()
                new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
        except:
            model.load_state_dict(ckpt['model_state_dict'])

            # model_name = 'Resnet_2d-150-0000.pth'
        model_name = ckpt_dir.split('/')[-1]
        logger.info('current model: {}'.format(model_name))


    if train_config.use_gpu:
        model.cuda() # required bofore optimizer?
    
    # print(model) # especially useful for debugging model structure.

    # lr
    lr = train_config.base_lr
    if_lr_decay = True if train_config.final_lr is not None else False

    if expe_config.resume_ckp != '':
        logger.info('==> loading checkpoint: {}'.format(expe_config.resume_ckp))
        checkpoint = torch.load(expe_config.resume_ckp)
        optimizer = checkpoint['optimizer']
    else:
        if train_config.multiTaskUncertainty=='Kendall' and len(model_config.task_names)>1:
            # MTU_module = multiTaskUncertaintySurvLoss(len(model_config.task_names))
            MTU_module = multiTaskUncertaintySurvLoss_sameAsPaper(len(model_config.task_names))
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': MTU_module.parameters(), 'weight_decay': 0}	
            ], lr=lr, weight_decay=train_config.weight_decay) # 
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=train_config.weight_decay)

    if train_config.lrScheduler == 'ReduceLROnPlateau':
        lrScheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0, patience=train_config.lrPatience, factor=0.5, verbose=True)
    elif train_config.lrScheduler == 'CyclicLR':
        lrScheduler = CyclicLR(optimizer, base_lr=train_config.base_lr, max_lr=0.01, step_size_down=100, step_size_up=100, mode='triangular2', cycle_momentum=False)

    # loss_fn
    if train_config.loss_type=='Cox':
        # loss_fn = cox_loss_cox_nnet()
        # loss_fn = CoxPHLoss()
        # loss_fn = NegativeLogLikelihood()
        # loss_fn = cox_loss_Olivier()
        loss_fn = NegLogPartialLikelihood()
    elif train_config.loss_type=='CE':
        loss_fn = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2)

    if expe_config.addClsTask:
        # jointClsLoss=JointClsLoss(bins=model.bins)
        jointClsLoss=JointClsLoss3(bins=model.bins)
    else:
        pass


    ### prep case ids
    if expe_config.out_tag == "SRRH_FT":
        cases_dict = {
            # 'dev':data_splits['dev'][expe_config.split]['train'] + data_splits['dev'][expe_config.split]['val'],
            'train':data_splits['dev'][expe_config.split]['train'],
            'val':data_splits['dev'][expe_config.split]['val'],
            # 'test':data_splits['test']
        }
        surv_endpoint_fname = "SRRH_surv_endpoint.csv"
    else:
        cases_dict = {
            # 'dev':data_splits['dev'][expe_config.split]['train'] + data_splits['dev'][expe_config.split]['val'],
            'train':data_splits['dev'][expe_config.split]['train'],
            'val':data_splits['dev'][expe_config.split]['val'],
            'test':data_splits['test']
        }
        surv_endpoint_fname = None

    # settings for training dataloader
    num_threads_in_multithreaded = config.num_workers 
    num_cached_per_queue = 2
    train_transforms = get_train_transform(expe_config.patch_size)

    seeds_candidates = list(range(0,100))
    if_prep_tumorMask=expe_config.addSegTask or expe_config.addClsTask or expe_config.addTumorMask
    if_prep_liverMsak=expe_config.addSegTask or expe_config.addClsTask or expe_config.addLiverMask

    if train_config.epoch_method == 'infinite':
        # separate recur and death sampling as it is rather difficult to obtain a batch containing both negative and positive cases in respect to both recur and death since the positive cases are rare in the datasets. however, this is required to compute cox loss. as a result Chao adopt the round-robin method to update the model by updating the model with the loss of recur and loss of death in turn.
        train_loader = survDataLoader(model_config.task_names, config.data_root_dict, expe_config.imgs_dir, cases_dict['train'], batch_size=model_config.batch_size, patch_size=expe_config.patch_size,
                                      mode='train',clin=model_config.addClin,  num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=False, shuffle=True, infinite=True,
                                      surv_endpoint_fname=surv_endpoint_fname,if_prep_tumorMask=if_prep_tumorMask)

        train_gen = MultiThreadedAugmenter(train_loader, train_transforms, num_processes=num_threads_in_multithreaded, num_cached_per_queue=num_cached_per_queue, pin_memory=False)
    elif train_config.epoch_method == 'finite':
        if train_config.return_incomplete:
            train_config.step_per_epoch = math.ceil(len(cases_dict['train'])/model_config.batch_size)
        else:
            train_config.step_per_epoch = math.floor(len(cases_dict['train'])/model_config.batch_size)

    steps_per_epoch = train_config.step_per_epoch # len(train_loader)
    print('steps per epoch:', steps_per_epoch)

    iter_count = 1
    train_stop = False
    train_epoch_loss = 0

    val_f = os.path.join(config.result_dir, 'epochs_eval_metrics.csv')
    val_fo = open(val_f, 'w')
    val_wo = csv.writer(val_fo, delimiter=',')
    val_wo.writerow(['epoch', 'split_tag', 'recur_cindex', 'death_cindex','mean_dice_art_liver', 'mean_dice_art_tumor', 'mean_dice_pv_liver','mean_dice_pv_tumor'])
    val_fo.flush()

    # torch.autograd.set_detect_anomaly(True) # this will slow the training. useful to track errors during backward?forward? and auto_grad

    for epoch in tqdm.tqdm(range(train_config.max_epoch)):
        # epoch = 0
        if not train_stop:
            # logger.info('    ----- training epoch {} -----'.format(epoch))

            if train_config.epoch_method == 'finite':
                train_loader = survDataLoader(model_config.task_names, config.data_root_dict, expe_config.imgs_dir, cases_dict['train'], batch_size=model_config.batch_size, patch_size=expe_config.patch_size, mode='train',clin=model_config.addClin,
                                              num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=train_config.return_incomplete, shuffle=True, infinite=False,
                                              surv_endpoint_fname=surv_endpoint_fname,if_prep_tumorMask=if_prep_tumorMask)

                train_gen = MultiThreadedAugmenter(train_loader, train_transforms, num_processes=num_threads_in_multithreaded, num_cached_per_queue=num_cached_per_queue, pin_memory=False)
            else:
                pass

            train_progressor = ProgressBar(mode='train_{}_{}'.format(model_config.model_name, expe_config.split), epoch=epoch, total_epoch=train_config.max_epoch, model_name=model_config.model_name, total=steps_per_epoch)
            model.train()
            

            iter_loss_sum = 0
            for step in tqdm.tqdm(range(steps_per_epoch)):
                # step = 0
                train_progressor.current = step

                loss = torch.tensor(0, dtype=torch.float32)
                loss = loss.cuda() # non_blocking=True

                # import ipdb; ipdb.set_trace()
                # time_dataloader = time.time()
                train_batch = next(train_gen)


                sample_ids = train_batch['names']
                # # if train_batch is None:
                if len(sample_ids) < model_config.batch_size:
                    logger.info('\n skip steps with train_batch containing no events, skip step{}'.format(step))
                    continue # skip steps with train_batch containing no events. in survDataLoader, Chao set all batches with no envent to return only the first sample. So that the len(sample_ids) of such batches will be 1, which is easy to identify here.
                else:
                    pass
                # # logger.info('\nelapsed time_dataloader:{}s'.format(time.time()-time_dataloader)) # ?s for ? workers

                images_all = train_batch['data'] # b, c, z,y,x
                if train_config.use_gpu:
                    images_all = torch.tensor(images_all).cuda()  # inputs to GPU # non_blocking=True
                
                lab = train_batch['surv']
                # lab
                if model_config.addClin:
                    clin_data = torch.tensor(train_batch['clin_data']).cuda()
                else:
                    clin_data = None

                # print('clin_data:{}'.format(str(clin_data)))

                allSeg_all = train_batch['seg']

                # if if_prep_tumorMask and not if_prep_liverMsak:
                #     tumorMasks_all = ((allSeg_all==2)*1).astype(allSeg_all.dtype) # original: background=0, liver=1, tumor=2; out: bacgroundORliver=0, tumor=1
                #     tumorMasks_all = torch.tensor(tumorMasks_all).cuda()
                # if if_prep_liverMsak and not if_prep_tumorMask:
                #     LiverMasks_all = ((allSeg_all>=1)*1).astype(allSeg_all.dtype) # original: background=0, liver=1, tumor=2; out: bacground=0, liverORtumor=1
                #     LiverMasks_all = torch.tensor(LiverMasks_all).cuda()

                # if if_prep_liverMsak and if_prep_tumorMask:
                #     LiverTumorMasks_all = copy.deepcopy(allSeg_all)
                #     LiverTumorMasks_all = torch.tensor(LiverTumorMasks_all).cuda()


                if if_prep_tumorMask:
                    tumorMasks_all = ((allSeg_all==2)*1).astype(allSeg_all.dtype) # original: background=0, liver=1, tumor=2; out: bacgroundORliver=0, tumor=1
                    tumorMasks_all = torch.tensor(tumorMasks_all).cuda()
                if if_prep_liverMsak:
                    LiverMasks_all = ((allSeg_all>=1)*1).astype(allSeg_all.dtype) # original: background=0, liver=1, tumor=2; out: bacground=0, liverORtumor=1
                    LiverMasks_all = torch.tensor(LiverMasks_all).cuda()
                if if_prep_liverMsak and if_prep_tumorMask:
                    LiverTumorMasks_all = copy.deepcopy(allSeg_all)
                    LiverTumorMasks_all = torch.tensor(LiverTumorMasks_all).cuda()


                # # 正向传播时：开启自动求导的异常侦测
                # torch.autograd.set_detect_anomaly(True) #开启后消耗时间，建议仅调试时使用 # 例如RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn（很多问题都会导致这个提示）

                # time_forward = time.time()
                torch.cuda.empty_cache()
                
                ## to determine what inputs to the model are, and run modelling
                images_ART = images_all[:,0,:,:,:].unsqueeze(1)
                images_PV = images_all[:,1,:,:,:].unsqueeze(1)

                if train_config.use_gpu:
                    images_ART = images_ART.cuda()
                    images_PV = images_PV.cuda()

                if 'mmtm' in model_config.model_name.lower():
                    if expe_config.addTumorMask and not expe_config.addLiverMask:
                        images_ART = torch.cat([images_ART, tumorMasks_all[:, 0, :, :, :].unsqueeze(1)], dim=1)
                        images_PV = torch.cat([images_PV, tumorMasks_all[:, 1, :, :, :].unsqueeze(1)], dim=1)
                    elif expe_config.addLiverMask and not expe_config.addTumorMask:  
                        images_ART = torch.cat([images_ART, LiverMasks_all[:, 0, :, :, :].unsqueeze(1)], dim=1)
                        images_PV = torch.cat([images_PV,LiverMasks_all[:, 1, :, :, :].unsqueeze(1)], dim=1)
                    elif expe_config.addLiverMask and expe_config.addTumorMask:  
                        # images_ART = torch.cat([images_ART, LiverTumorMasks_all[:, 0, :, :, :].unsqueeze(1)], dim=1)
                        # images_PV = torch.cat([images_PV,LiverTumorMasks_all[:, 1, :, :, :].unsqueeze(1)], dim=1)
                        bg_ART = (LiverTumorMasks_all[:, 0, :, :, :].unsqueeze(1)==0) *1 
                        liver_ART = (LiverTumorMasks_all[:, 0, :, :, :].unsqueeze(1)==1) *1 
                        tumor_ART = (LiverTumorMasks_all[:, 0, :, :, :].unsqueeze(1)==2) *1
                        images_ART = torch.cat([images_ART, bg_ART, liver_ART, tumor_ART], dim=1)

                        bg_PV = (LiverTumorMasks_all[:, 1, :, :, :].unsqueeze(1)==0) *1 
                        liver_PV = (LiverTumorMasks_all[:, 1, :, :, :].unsqueeze(1)==1) *1 
                        tumor_PV = (LiverTumorMasks_all[:, 1, :, :, :].unsqueeze(1)==2) *1
                        images_PV = torch.cat([images_PV, bg_PV, liver_PV, tumor_PV], dim=1)
                    else:
                        pass

                    model_res = model(images_ART, images_PV, clin_data)

                elif len(model_config.modality)==1 and 'ART' in model_config.modality:
                    raise ValueError('if expe_config.addTumorMask: TBD')
                    model_res = model(images_ART,None,clin_data)

                elif len(model_config.modality)==1 and 'PV' in model_config.modality:
                    raise ValueError('if expe_config.addTumorMask: TBD')
                    model_res = model(images_PV, None,clin_data)
                    
                else:
                    model_res = model(images_all,clin_data) # 对于不含MMTM的模型，目前forward中还没添加clin_data=None
                # logger.info('\nelapsed time_forward:{}s'.format(time.time()-time_forward)) # ?s for ? workers

                ## store data for tensorboard
                # if iter_count % 20 == 0:
                if (iter_count-1) % steps_per_epoch == 0:
                    # # for debug
                    # import SimpleITK as sitk
                    # for k in range(2):
                    #     sitk_img = sitk.GetImageFromArray(images_all.detach().cpu().numpy()[0,k,:])
                    #     sitk.WriteImage(sitk_img, '/data/cHuang/HCC_proj/results/{}_{}.nii.gz'.format(sample_ids[0],k))
                    # train_utils.tb_images([images_all[0, 0, :, :,:], images_all[0, 1, :, :,:]], [False, False], ['pv', 'art'], iter_count, tag='Train_{}_epoch{}_step{}'.format(sample_ids[0], str(epoch).zfill(3), str(step).zfill(3)), img_is_RGB=False)
                    if expe_config.addTumorMask or expe_config.addLiverMask:
                        train_utils.tb_images([images_ART[0, 0, :, :,:], images_ART[0, 1, :, :,:]], [False, True], ['art_img', 'art_lab'], iter_count, tag='Train_{}_epoch{}_step{}_art'.format(sample_ids[0], str(epoch).zfill(3), str(step).zfill(3)), img_is_RGB=False)

                        train_utils.tb_images([images_PV[0, 0, :, :,:], images_PV[0, 1, :, :,:]], [False, True], ['pv_img', 'pv_lab'], iter_count, tag='Train_{}_epoch{}_step{}_pv'.format(sample_ids[0], str(epoch).zfill(3), str(step).zfill(3)), img_is_RGB=False)
                    if expe_config.addSegTask:
                        case_idx = 0

                        train_utils.tb_images([images_ART[case_idx, 0, :, :,:], LiverTumorMasks_all[case_idx, 0, :, :,:]], [False, True], ['art_img', 'art_lab'], iter_count, tag='Train_{}_epoch{}_step{}_art'.format(sample_ids[case_idx], str(epoch).zfill(3), str(step).zfill(3)), img_is_RGB=False)

                        train_utils.tb_images([images_PV[case_idx, 0, :, :,:], LiverTumorMasks_all[case_idx, 1, :, :,:]], [False, True], ['art_img', 'art_lab'], iter_count, tag='Train_{}_epoch{}_step{}_art'.format(sample_ids[case_idx], str(epoch).zfill(3), str(step).zfill(3)), img_is_RGB=False)

                    elif expe_config.addSurvTask:
                        case_idx = 0

                        train_utils.tb_images([images_ART[case_idx, 0, :, :, :],images_PV[case_idx, 0, :, :, :]],[False,False], ['art_img','pv_img'], iter_count,tag='Train_{}_epoch{}_step{}'.format(sample_ids[case_idx], str(epoch).zfill(3),str(step).zfill(3)), img_is_RGB=False)
                        
                    else:
                        pass
                
                ## loss: regularization
                if train_config.L1_reg_lambda is not None:
                    loss += train_config.L1_reg_lambda*l_reg(model, 'L1')
                if train_config.L2_reg_lambda is not None:
                    loss += train_config.L2_reg_lambda*l_reg(model,'L2')

                # logger.info('{}'.format(str(sample_ids)))

                loss_tmp_list = list()
                
                ## loss: survival 
                surv_loss = 0
                if expe_config.addSurvTask:
                    logits_dict = model_res['logits']
                    for task_name in model_config.task_names:
                        # task_name = model_config.task_names[0]
                        surv = np.array([i['{}_surv'.format(task_name)] for i in lab])
                        surv_status_tensor = torch.tensor([i['status'] for i in surv]).cuda()
                        surv_time_tensor = torch.tensor([i['time'] for i in surv]).cuda()
                        # logger.info('\n--task:{}--'.format(task_name)) #detach().cpu().
                        # logger.info('\nlogits:{}'.format(str(logits_dict[task_name].tolist())))

                        # logger.info('status:{}'.format(str(surv_status_tensor.tolist()))) #detach().cpu().

                        logits_to_print = logits_dict[task_name].detach().cpu()[:,0]
                        # for i in range(len(logits_to_print)):
                        #     # if i >88, torch.exp(torch.tensor(i)) outputs inf. 上溢
                        #     # if i is negative and i <-103, torch.exp(torch.tensor(i)) outputs 0.
                        #     if abs(logits_to_print[i])>50:
                        #         logger.info('logits---{}: {}'.format(sample_ids[i], str(logits_to_print[i])))
                        # print(str(logits_to_print))
                        if all([i<=88 for i in logits_to_print]) and all([i>=-103 for i in logits_to_print]):
                            pass
                        else:
                            logger.info('\n sample_ids:{}'.format(str(sample_ids)))
                            logger.info('\n logits:{}'.format(str(logits_to_print)))
                            logger.info('\n time:{}'.format(str(surv_time_tensor)))
                            logger.info('\n status:{}'.format(str(surv_status_tensor)))

                        if train_config.loss_type=='Cox':
                            loss_tmp = loss_fn(logits_dict[task_name],surv_time_tensor, surv_status_tensor)
                        elif train_config.loss_type=='CE':
                            loss_tmp = loss_fn(torch.sigmoid(logits_dict[task_name]), surv_status_tensor)
                        loss_tmp_list.append(loss_tmp)

                        # logger.info('\n{}_loss_tmp:{}'.format(task_name, str(loss_tmp.tolist())))
                        # ystatus = surv_status_tensor.reshape(-1,1)
                        # num_positives += torch.sum(ystatus)

                        # config.writer.add_histogram('logits_'+task_name, logits_dict[task_name][:,0], iter_count)


                    current_loss_by_tasks = ['{}:{}'.format(model_config.task_names[i], round(loss_tmp_list[i].tolist(),4)) for i in range(len(loss_tmp_list))]
                    if step == 0:
                        logger.info('surv loss: {}'.format(str(current_loss_by_tasks)))

                    if len(model_config.task_names)>1:
                        if train_config.multiTaskUncertainty=='Kendall':
                            surv_loss += MTU_module(loss_tmp_list).cuda()
                            for name, param in MTU_module.named_parameters():
                                logger.info('\nMTU params: {}'.format(str(param.data)))
                        else:
                            multitask_loss = torch.tensor(0, dtype=torch.float32).cuda()
                            for loss_tmp in loss_tmp_list:
                                multitask_loss += loss_tmp
                            multitask_loss /= len(model_config.task_names)
                            surv_loss += multitask_loss
                    else:
                        surv_loss += loss_tmp_list[0]
                else:
                    current_loss_by_tasks = []
                #

                ## loss: CAAM based classification
                
                if expe_config.addClsTask and expe_config.addSegTask:
                    # seg_gt_art = LiverTumorMasks_all[:,0,:,:,:] # [2, 48,256,320]
                    # seg_gt_pv = LiverTumorMasks_all[:,1,:,:,:]

                    # # art_outputs = model_res['cls_score_art']
                    # # pv_outputs = model_res['cls_score_pv']

                    # art_outputs = model_res['seg_pred_art']
                    # pv_outputs = model_res['seg_pred_pv']

                    # target_dict_art = {
                    #         'target': seg_gt_art,
                    #         'featMapSize': model_res['featMapSize'],
                    #     }
                    # target_dict_pv = {
                    #         'target': seg_gt_pv,
                    #         'featMapSize': model_res['featMapSize'],
                    #     }

                    # cls_loss_art = jointClsLoss(art_outputs, target_dict_art)
                    # cls_loss_pv = jointClsLoss(pv_outputs, target_dict_pv)
                    # final_cls_loss=(cls_loss_art + cls_loss_pv)/2

                    # config.writer.add_scalar('train/loss_step_seg_pv', float(cls_loss_pv), iter_count)
                    # config.writer.add_scalar('train/loss_step_seg_art', float(cls_loss_art), iter_count)

                    # seg_lambda = 0.5
                    # loss += seg_lambda * final_cls_loss


                    #### use seg loss for CAAM model
                    # seg_weights = [1, 0.5, 0.4, 0.3, 0.2, 0.1] # for different resolution levels
                    seg_weights = [1,1,1,1,1,1,1] # for different resolution levels

                    seg_pred_list_art = model_res['seg_pred_art']
                    seg_pred_list_pv = model_res['seg_pred_pv']

                    featMapSize_list = model_res['featMapSize']

                    assert type(seg_pred_list_art) is list, 'seg_pred_list_art should be a list'

                    seg_loss_art = 0
                    seg_loss_pv = 0

                    seg_loss_art_tmp_list = list()
                    seg_loss_pv_tmp_list = list()

                    for i in range(1):
                        # i = 0
                        res_level = i+2 # starts from the 1/4 resolution level. suppose that seg from 1 and 1/2 resolution level are not correct. refer to Liu CVPR 2022
                        featMapSize = featMapSize_list[res_level]

                        seg_gt = F.interpolate(LiverTumorMasks_all, tuple(featMapSize[-3::]), mode='nearest') # if sizes same, no interpolation done.
                        seg_gt_art = seg_gt[:,0,:,:,:].contiguous() # [2, 48,256,320] # not sure why .contiguous() should be added here. by Chao
                        seg_gt_pv = seg_gt[:,1,:,:,:].contiguous()

                        seg_loss_art_tmp = (lovasz_softmax(F.softmax(seg_pred_list_art[i], dim=1), seg_gt_art, ignore=10) + focal_loss(seg_pred_list_art[i], seg_gt_art))/2

                        seg_loss_pv_tmp = (lovasz_softmax(F.softmax(seg_pred_list_pv[i], dim=1), seg_gt_pv, ignore=10) + focal_loss(seg_pred_list_pv[i], seg_gt_pv))/2

                        config.writer.add_scalar('train/loss_step_seg_art', float(seg_loss_art), iter_count)
                        config.writer.add_scalar('train/loss_step_seg_pv', float(seg_loss_pv), iter_count)

                        seg_loss_art += seg_weights[i] * seg_loss_art_tmp
                        seg_loss_pv += seg_weights[i] * seg_loss_pv_tmp

                        seg_loss_art_tmp_list.append(seg_loss_art_tmp)
                        seg_loss_pv_tmp_list.append(seg_loss_pv_tmp)

                    if step == 0:
                        logger.info('dice loss for reso levels: art---{}, pv---{}'.format(str([round(float(i.detach().cpu()),5) for i in seg_loss_art_tmp_list]), str([round(float(i.detach().cpu()),5) for i in seg_loss_pv_tmp_list])))

                    seg_loss_art /= len(seg_loss_art_tmp_list)
                    seg_loss_pv /= len(seg_loss_pv_tmp_list)

                    CAAM_loss = (seg_loss_art + seg_loss_pv)/2
                else:
                    CAAM_loss = 0

                ## loss: segmentation
                
                if not expe_config.addClsTask and expe_config.addSegTask:
                    seg_weights = [1, 0.5, 0.4, 0.3, 0.2, 0.1] # for different resolution levels

                    seg_pred_list_art = model_res['seg_pred_art']
                    seg_pred_list_pv = model_res['seg_pred_pv']

                    featMapSize_list = model_res['featMapSize']

                    assert type(seg_pred_list_art) is list, 'seg_pred_list_art should be a list'

                    seg_loss_art = 0
                    seg_loss_pv = 0

                    seg_loss_art_tmp_list = list()
                    seg_loss_pv_tmp_list = list()

                    for i in range(len(seg_pred_list_art)):
                        # i = 0
                        res_level = i+1 # starts from the second resolution level. Remove the highest level seg decoder level to save GPU memory
                        featMapSize = featMapSize_list[res_level]

                        seg_gt = F.interpolate(LiverTumorMasks_all, tuple(featMapSize[-3::]), mode='nearest') # if sizes same, no interpolation done.
                        seg_gt_art = seg_gt[:,0,:,:,:].contiguous() # [2, 48,256,320] # not sure why .contiguous() should be added here. by Chao
                        seg_gt_pv = seg_gt[:,1,:,:,:].contiguous()

                        seg_loss_art_tmp = (lovasz_softmax(F.softmax(seg_pred_list_art[-res_level], dim=1), seg_gt_art, ignore=10) + focal_loss(seg_pred_list_art[-res_level], seg_gt_art))/2

                        seg_loss_pv_tmp = (lovasz_softmax(F.softmax(seg_pred_list_pv[-res_level], dim=1), seg_gt_pv, ignore=10) + focal_loss(seg_pred_list_pv[-res_level], seg_gt_pv))/2

                        config.writer.add_scalar('train/loss_step_seg_art', float(seg_loss_art), iter_count)
                        config.writer.add_scalar('train/loss_step_seg_pv', float(seg_loss_pv), iter_count)

                        seg_loss_art += seg_weights[i] * seg_loss_art_tmp
                        seg_loss_pv += seg_weights[i] * seg_loss_pv_tmp

                        seg_loss_art_tmp_list.append(seg_loss_art_tmp)
                        seg_loss_pv_tmp_list.append(seg_loss_pv_tmp)

                    if step == 0:
                        logger.info('dice loss for reso levels: art---{}, pv---{}'.format(str([round(float(i.detach().cpu()),5) for i in seg_loss_art_tmp_list]), str([round(float(i.detach().cpu()),5) for i in seg_loss_pv_tmp_list])))

                    seg_loss_art /= len(seg_loss_art_tmp_list)
                    seg_loss_pv /= len(seg_loss_pv_tmp_list)

                    seg_loss = (seg_loss_art + seg_loss_pv)/2
                else:
                    seg_loss = 0


                #### gather all loss items
                if expe_config.addSurvTask and not expe_config.addClsTask and not expe_config.addSegTask: 
                    loss = 1 * surv_loss
                elif expe_config.addSurvTask and expe_config.addClsTask:
                    epoch_alphas = {
                        # 200:0.5,
                        # 220:1
                        10:0.5
                    }
                    surv_over_seg_alpha = 0
                    if epoch in epoch_alphas.keys():
                        surv_over_seg_alpha = epoch_alphas[epoch]
                    if surv_over_seg_alpha==epoch_alphas[list(epoch_alphas.keys())[0]]:
                        # when surv branch works, reset lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = train_config.base_lr
                    # surv_over_seg_alpha = min(1, (epoch//150)*0.25) # priority of the losses slowly change from seg>surv to surv>seg
                    # for epoch in range(500):
                    #     print('{}: {}'.format(epoch, surv_over_seg_alpha))

                    seg_lambda = 10 # seg loss is in [0,1], while surv_loss mostly around 1~10.
                    loss += surv_over_seg_alpha * surv_loss + (1-surv_over_seg_alpha)*seg_lambda*CAAM_loss
                elif expe_config.addSurvTask and expe_config.addSegTask:
                    epoch_alphas = {
                        100:0.5,
                        250:1
                    }
                    surv_over_seg_alpha = 0
                    if epoch in epoch_alphas.keys():
                        surv_over_seg_alpha = epoch_alphas[epoch]
                    if surv_over_seg_alpha==epoch_alphas[list(epoch_alphas.keys())[0]]:
                        # when surv branch works, reset lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = train_config.base_lr
                    # surv_over_seg_alpha = min(1, (epoch//150)*0.25) # priority of the losses slowly change from seg>surv to surv>seg
                    # for epoch in range(500):
                    #     print('{}: {}'.format(epoch, surv_over_seg_alpha))

                    seg_lambda = 10 # seg loss is in [0,1], while surv_loss mostly around 1~10.
                    loss += surv_over_seg_alpha * surv_loss + (1-surv_over_seg_alpha)*seg_lambda*seg_loss
                

                config.writer.add_scalar('train/loss_step', loss.item(), iter_count)
                iter_loss_sum += float(loss.item()) # .item() Returns the value of this tensor as a standard Python number

                # time_backward = time.time()
                optimizer.zero_grad()   # zero the parameter gradients

                # # 反向传播时：在求导时开启侦测 #开启后消耗时间，建议仅调试时使用
                # with torch.autograd.detect_anomaly():
                #     loss.backward()

                loss.backward() # gradients are computed

                # try:
                #     loss.backward()  # backward
                # # except RuntimeError:
                # #     logger.info('{}'.format(str(RuntimeError)))
                # #     return
                # except Exception as e:
                #     logger.exception("Unexpected exception! %s",e)
                #     return

                optimizer.step()  # optimize
                # logger.info('\nelapsed time_backward:{}s'.format(time.time()-time_backward)) # ?s for ? workers

                ##########需要打印每个iter的，否则数据更新不及时，无法反映报错的iter的情况#########
                # this op costs much time. 
                if expe_config.debug:
                    #这部分代码在lungnetOlivier17，batchsize=10时会占用3GB显存
                    for name, param in model.named_parameters():      #返回模型的参数
                        print(name + '_data')

                        print(param)
                        config.writer.add_histogram(name + '_data', param.data, iter_count)   #参数的权值
                        print(name+'_grad')
                        print(param.grad)
                        config.writer.add_histogram(name + '_grad', param.grad, iter_count)   #参数的梯度，need to be after loss.backward to get the grads #如果返回'The histogram is empty, please file a bug report.'，根据报错信息回溯到tensorboardX的一段代码，发现是由于 if counts.size == 0 or limits.size == 0: raise ValueError('The histogram is empty, please file a bug report.')

                # # debug start
                # lr = optimizer.param_groups[0]['lr']
                # # config.writer.add_scalar('train/loss_lr', loss.item(), lr*10000)
                # loss_lr_wo.writerow([epoch, iter_count, lr, loss.item()])
                # loss_lr_fo.flush()
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr * 2
                # # debug end

                # train_progressor.current_loss = loss.item()
                if expe_config.addSegTask:
                    train_progressor.current_loss = str(current_loss_by_tasks + ['total:{}'.format(round(float(loss.item()),4))] + ['PV diceLoss:{}'.format(round(float(seg_loss_pv),4))] + ['ART diceLoss:{}'.format(round(float(seg_loss_art),4))])
                else:
                    train_progressor.current_loss = str(current_loss_by_tasks + ['total:{}'.format(round(float(loss.item()),4))])

                train_progressor()


                ## end of iteration
                iter_count += 1

            # after one epoch of training
            train_progressor.done(os.path.join(config.log_dir,'train_progress_bar.txt'))

            train_epoch_loss = iter_loss_sum/steps_per_epoch

            config.writer.add_scalar('train/epoch_loss', train_epoch_loss, epoch)
            if expe_config.addSegTask:
                logger.info('epoch {}: PV dice loss={}, ART dice loss={}'.format(epoch,round(float(seg_loss_pv),4),round(float(seg_loss_art),4)))
            else:
                pass
            config.writer.add_scalar('train/lr', lr, iter_count-1)

            # inits
            eval_bool_train = False
            eval_bool_val = False
            eval_bool_test = False
            train_stop = False

            #### save checkpoint
            if epoch % train_config.save_epoch == 0 or epoch==train_config.max_epoch-1:
                save_for_infer = True
            else:
                save_for_infer = False

            if epoch % 5 == 0:
                save_for_resume = True
            else:
                save_for_resume = False

            #### debug start
            save_model(epoch, model, optimizer, train_epoch_loss, config.ckpt_dir, save_for_infer=save_for_infer, save_for_resume=save_for_resume)
            #### debug end

            #### determine if do inference or stop training
            if (lr < train_config.final_lr):
                eval_bool_train = True
                eval_bool_val = True
                eval_bool_test = True
                train_stop = True
                logger.info('epoch:{}. lr is reduced to {}. Will do the last evaluation'.format(epoch, lr))
            elif epoch==train_config.max_epoch-1:
                eval_bool_train = True
                eval_bool_val = True
                eval_bool_test = True
                train_stop = True
                logger.info('reach to max epoch {}. Will do the last evaluation'.format(train_config.max_epoch))
            else:
                pass

            ### eval
            if expe_config.trainEval and epoch >= train_config.start_trainEval_epoch and (epoch-train_config.start_trainEval_epoch) % train_config.trainEval_epoch_interval == 0:
                eval_bool_train = True
            else:
                pass
        
            if expe_config.val and epoch >= train_config.start_val_epoch and (epoch-train_config.start_val_epoch) % train_config.val_epoch_interval == 0:
                eval_bool_val = True
            else:
                pass

            if expe_config.test and epoch>=train_config.start_test_epoch and (epoch-train_config.start_test_epoch) % train_config.test_epoch_interval == 0:
                eval_bool_test = True
            else:
                pass
            
            #
            if eval_bool_train and eval_bool_val:
                dev_data2calMetrics = {
                    'pat_id':list(),
                    'surv':{
                        'recur':{
                            'time':list(),
                            'status':list(),
                            'logits':list()
                        },
                        'death':{
                            'time':list(),
                            'status':list(),
                            'logits':list()
                        }
                    },
                    'seg':{
                        'dice_art_liver':list(),
                        'dice_art_tumor':list(),
                        'dice_pv_liver':list(),
                        'dice_pv_tumor':list()
                    }
                }
            else:
                pass        
                # val_ids have to include both positive and negative cases.
            
            logger.info('start to eval after epoch {}'.format(epoch))

            val_loss_tmp = 0

            logger.info('cases_dict.keys(): {}'.format(str(cases_dict.keys())))
            for split_tag in cases_dict.keys():
                if split_tag=='train' and not eval_bool_train:
                    pass
                elif split_tag=='val' and not eval_bool_val:
                    pass
                elif split_tag=='test' and not eval_bool_test:
                    pass
                else:
                    eval_ids = cases_dict[split_tag]

                    if eval_bool_train and eval_bool_val and split_tag in ['train','val']:
                        dev_data2calMetrics['pat_id'].extend(eval_ids)

                    np.random.seed(epoch)
                    np.random.shuffle(eval_ids)

                    eval_out_dir = os.path.join(config.eval_out_dir, 'epoch_{}'.format(epoch), split_tag)
                    try:
                        pred_out = predict.predict(expe_config, eval_ids, model, model_config, eval_out_dir, mode='test_online',surv_endpoint_fname=surv_endpoint_fname)
                    except Exception as e:
                        logger.exception("Unexpected exception! %s",e)
                        raise ValueError('eval failed')

                    # for surv
                    if expe_config.addSurvTask:
                        recur_cindex = pred_out['surv']['recur']['cindex']
                        death_cindex = pred_out['surv']['death']['cindex']

                        for task in ['recur','death']:
                            if task in model_config.task_names:
                                config.writer.add_scalar('{}/c_index/{}'.format(task, split_tag), pred_out['surv'][task]['cindex'], epoch)

                                if split_tag=='val':
                                    val_loss_tmp += -pred_out['surv'][task]['cindex']

                                if eval_bool_train and eval_bool_val and split_tag in ['train', 'val']: # prep input for computing dev cindex
                                    dev_data2calMetrics['surv'][task]['time'].extend([pred_out['surv'][task]['pat_res'][pat_id]['time'] for pat_id in eval_ids])
                                    dev_data2calMetrics['surv'][task]['status'].extend([pred_out['surv'][task]['pat_res'][pat_id]['status'] for pat_id in eval_ids])
                                    dev_data2calMetrics['surv'][task]['logits'].extend([pred_out['surv'][task]['pat_res'][pat_id]['logits'] for pat_id in eval_ids])

                                else:
                                    pass
                            else:
                                pass
                    else:
                        recur_cindex = None
                        death_cindex = None
                    
                    # for seg
                    if not expe_config.addSegTask:
                        mean_dice_art_liver = None
                        mean_dice_art_tumor = None
                        mean_dice_pv_liver = None
                        mean_dice_pv_tumor = None
                    else:
                        mean_dice_art_liver = round(pred_out['seg']['mean_dice_art_liver'],3)
                        mean_dice_art_tumor = round(pred_out['seg']['mean_dice_art_tumor'],3)
                        mean_dice_pv_liver = round(pred_out['seg']['mean_dice_pv_liver'],3)
                        mean_dice_pv_tumor = round(pred_out['seg']['mean_dice_pv_tumor'],3)

                        # collect for cal metrics for dev
                        if eval_bool_train and eval_bool_val and split_tag in ['train','val']:
                            dev_data2calMetrics['seg']['dice_art_liver'].extend(pred_out['seg']['dice_art_liver'])
                            dev_data2calMetrics['seg']['dice_art_tumor'].extend(pred_out['seg']['dice_art_tumor'])
                            dev_data2calMetrics['seg']['dice_pv_liver'].extend(pred_out['seg']['dice_pv_liver'])
                            dev_data2calMetrics['seg']['dice_pv_tumor'].extend(pred_out['seg']['dice_pv_tumor'])
                        else:
                            pass

                    val_wo.writerow([epoch,split_tag, recur_cindex,death_cindex,mean_dice_art_liver, mean_dice_art_tumor, mean_dice_pv_liver,mean_dice_pv_tumor])
                    val_fo.flush()

                    logger.info('epoch {}: {}, sample_size={}, recur_cindex={}, death_cindex={}, dice_art_liver={}, dice_art_tumor={}, dice_pv_liver={}, dice_pv_tumor={}'.format(epoch, split_tag, len(eval_ids), str(recur_cindex), str(death_cindex), str(mean_dice_art_liver), str(mean_dice_art_tumor), str(mean_dice_pv_liver), str(mean_dice_pv_tumor)))
                
            # cal val loss
            val_loss = val_loss_tmp/len(model_config.task_names) # use -c-index as val loss

            ## compute dev cindex: use out from val and train, to save time
            if eval_bool_train and eval_bool_val:
                dev_sample_size = len(dev_data2calMetrics['pat_id'])
                
                # for surv
                dev_cindices = dict()
                for task in ['recur','death']:
                    if expe_config.addSurvTask and task in model_config.task_names:
                        dev_cindices[task] = concordance_index(dev_data2calMetrics['surv'][task]['time'], [-i for i in dev_data2calMetrics['surv'][task]['logits']], dev_data2calMetrics['surv'][task]['status'])
                        config.writer.add_scalar('{}/c_index/dev'.format(task), dev_cindices[task], epoch)
                    else:
                        dev_cindices[task] = None
                
                # for seg
                dev_seg_metrics = dict()
                dev_seg_metrics['dice'] = dict()
                for ph in ['art','pv']:
                    dev_seg_metrics['dice'][ph] = dict()
                    if expe_config.addSegTask:
                        dev_seg_metrics['dice'][ph]['liver'] = round(np.mean(dev_data2calMetrics['seg']['dice_{}_liver'.format(ph)]),3)
                        dev_seg_metrics['dice'][ph]['tumor'] = round(np.mean(dev_data2calMetrics['seg']['dice_{}_tumor'.format(ph)]),3)
                    else:
                        dev_seg_metrics['dice'][ph]['liver'] = None
                        dev_seg_metrics['dice'][ph]['tumor'] = None

                val_wo.writerow([epoch,'dev', dev_cindices['recur'],dev_cindices['death'], dev_seg_metrics['dice']['art']['liver'], dev_seg_metrics['dice']['art']['tumor'], dev_seg_metrics['dice']['pv']['liver'], dev_seg_metrics['dice']['pv']['tumor']])
                val_fo.flush()

                logger.info('epoch {}: dev, sample_size={}, recur_cindex={}, death_cindex={}, dice_art_liver={}, dice_art_tumor={}, dice_pv_liver={}, dice_pv_tumor={}'.format(epoch, dev_sample_size, str(dev_cindices['recur']), str(dev_cindices['death']), str(dev_seg_metrics['dice']['art']['liver']), str(dev_seg_metrics['dice']['art']['tumor']), str(dev_seg_metrics['dice']['pv']['liver']), str(dev_seg_metrics['dice']['pv']['tumor'])))


            ###### lr decay
            if if_lr_decay:
                lr_old = optimizer.param_groups[0]['lr']

                if train_config.lrScheduler == 'ReduceLROnPlateau':

                    lrScheduler.step(train_epoch_loss)

                    # adjust lr based on validation loss. below partly refer to: Jiawen yao, 2023, Deep learning for fully automated prediction of overlall survival in patients undergoing resection for pancreatic cancer: a retrospective multicenter study. Annals of Surgery.
                    # lrScheduler.step(val_loss)

                elif train_config.lrScheduler == 'CyclicLR':
                    lrScheduler.step()
                lr = optimizer.param_groups[0]['lr']

                if lr<lr_old:
                    logger.info('epoch:{}. lr is reduced from {} to {}. continue training'.format(epoch, lr_old, lr))
            

                    
        else:
            break

                    
