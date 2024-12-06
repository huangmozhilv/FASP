
import os
import sys
sys.path.append(os.getcwd())
import copy
import csv
import time

import SimpleITK as sitk
import pandas as pd
import numpy as np
import tqdm
from lifelines.utils import concordance_index
# from sksurv.metrics import concordance_index_censored
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

import torch
import torch.nn as nn
import torch.nn.functional as F

from survPred.training import train_utils
import survPred.config as config
from survPred.surv_dataloader import survDataLoader
# from survPred.tumorSurv_dataloader import tumorSurvDataLoader

import ccToolkits.tinies as tinies
import ccToolkits.logger as logger
from ccToolkits import utils_img
from ccToolkits.eval_metrics import seg_metrics

from tumorSurvPred.model_interpretability import get_cam,get_tsneplot,get_KMplot


# def predict(data_splits, model, model_config, mode='test_online'):
def predict(expe_config, cases, model, model_config, pred_out_dir, mode='test_online',surv_endpoint_fname=None):
    # mode: 
    # --- test_online, test during training or right after training, so imediately use the trained model;
    # --- test_offline， test with saved checkpoint.

    if_prep_tumorMask=expe_config.addSegTask or expe_config.addTumorMask or expe_config.addClsTask
    if_prep_liverMsak=expe_config.addSegTask or expe_config.addLiverMask or expe_config.addClsTask

    # ## prep gt: survival
    # if mode != 'infer':
    #     surv_dir = os.path.join(config.data_root_dict['nonImg'],'survival')

    #     # recur_surv_f = os.path.join(surv_dir, 'surv_recurrence.csv')
    #     # death_surv_f = os.path.join(surv_dir, 'surv_death.csv')
    #     if not surv_endpoint_fname:
    #         surv_endpoint_f = os.path.join(surv_dir, 'surv_endpoint.csv')
    #     else:
    #         surv_endpoint_f = os.path.join(surv_dir, surv_endpoint_fname)

    #     surv_endpoint_df = pd.read_csv(surv_endpoint_f)

    #     all_surv_dict = dict()
    #     for task in ['recur','death']:
    #         if task in model_config.task_names:
    #             all_surv_dict[task] = dict()

    #             for pat_id in cases:
    #                 # pat_id = cases[0]
    #                 if "TCGA" in pat_id:
    #                     pat_id_=f"TCGA-{pat_id[-6:-4]}-{pat_id[-4:]}"
    #                 else:
    #                     pat_id_=pat_id
    #                 all_surv_dict[task][pat_id] = dict()
    #                 all_surv_dict[task][pat_id]['status'] = surv_endpoint_df['{}_status'.format('recur')][surv_endpoint_df['pat_id']==pat_id_].values[0]
    #                 all_surv_dict[task][pat_id]['time'] = surv_endpoint_df['{}_time'.format('recur')][surv_endpoint_df['pat_id']==pat_id_].values[0]
    #         else:
    #             pass
    # else:
    #     pass

    # outputs
    test_result_dir = pred_out_dir + '/testResult'
    tinies.newDir(test_result_dir)

    if mode=='test_online':
        pass
    elif mode=='test_offline' or mode=='infer':
        ckpt_dir = model_config.model_loc
        ckpt = torch.load(ckpt_dir)

        model = nn.parallel.DataParallel(model) # essential
        # model.load_state_dict(ckpt['model_state_dict'])

        try:
          from collections import OrderedDict
          new_state_dict = OrderedDict()
          for k, v in ckpt['model_state_dict'].items():
            name = k.split('module.')[1] # remove 'module.' of dataparallel # don't use .strip as it will remove the leading or trailing characters same as in .strip()
            new_state_dict[name]=v
            model.load_state_dict(new_state_dict)
        except:
            model.load_state_dict(ckpt['model_state_dict'])

        #model_name = 'Resnet_2d-150-0000.pth'
        model_name = ckpt_dir.split('/')[-1]
        logger.info('current model: {}'.format(model_name))
    else:
        raise ValueError('Please specify the correct mode for testing')
    # load model
    model.cuda()
    model.eval()
    

    # print('evaluate on each case...')
    fo = open(os.path.join(pred_out_dir, 'eval_metrics_per_case.csv'), 'w')
    wo = csv.writer(fo, delimiter=',')
    wo.writerow(['pat_id', 'dice_art_liver', 'dice_art_tumor', 'dice_pv_liver', 'dice_pv_tumor'])
    fo.flush()

    # # init
    if mode != 'infer':
        res_surv_out = dict()
        res_seg_out = dict()

        # init for surv
        for task in ['recur', 'death']:
            res_surv_out[task] = dict()
            res_surv_out[task]['pat_res'] = dict()
            res_surv_out[task]['cindex'] = None

        # init for seg
        res_seg_out['pat_id'] = list()
        res_seg_out['dice_art_liver'] = list()
        res_seg_out['dice_art_tumor'] = list()
        res_seg_out['dice_pv_liver'] = list()
        res_seg_out['dice_pv_tumor'] = list()

        res_seg_out['mean_dice_art_liver'] = None
        res_seg_out['mean_dice_art_tumor'] = None
        res_seg_out['mean_dice_pv_liver'] = None
        res_seg_out['mean_dice_pv_tumor'] = None
    else:
        pass

    #     torch.cuda.empty_cache() # necessary. otherwise this code will result in increasing GPU occupation.
    if mode=='test_online':
        pred_batch_size = 3 # 6 # model_config.batch_size
        num_threads_in_multithreaded = 3
    elif mode=='test_offline' or mode=='infer':
        pred_batch_size = 6 # model_config.batch_size
        num_threads_in_multithreaded = 6
    
    # if model_config.model_name.startswith('Liver'):
    #     pred_loader = survDataLoader(model_config.task_names, config.data_root_dict, expe_config.imgs_dir, cases, batch_size=pred_batch_size, patch_size=expe_config.patch_size,
    #     mode='infer',clin=model_config.addClin,  num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=True,
    #                                  shuffle=False, infinite=False,if_prep_tumorMask=if_prep_tumorMask,surv_endpoint_fname=surv_endpoint_fname) # 'num_threads_in_multithreaded' here should be the same as that in 'MultiThreadedAugmenter'
    # elif model_config.model_name.startswith('Tumor'):
    #     pred_loader = tumorSurvDataLoader(model_config.task_names, config.data_root_dict, expe_config.imgs_dir, cases, batch_size=pred_batch_size, patch_size=expe_config.patch_size,
    #     mode='infer',clin=model_config.addClin,  num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=True, shuffle=False, infinite=False,if_prep_tumorMask=True) # 'num_threads_in_multithreaded' here should be the same as that in 'MultiThreadedAugmenter'
    # else:
    #     raise ValueError('model name not in correct format')

    pred_loader = survDataLoader(model_config.task_names, config.data_root_dict, expe_config.imgs_dir, cases, batch_size=pred_batch_size, patch_size=expe_config.patch_size,
    mode='infer',clin=model_config.addClin,  num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=True,
                                 shuffle=False, infinite=False,if_prep_tumorMask=if_prep_tumorMask,surv_endpoint_fname=surv_endpoint_fname) # 'num_threads_in_multithreaded' here should be the same as that in 'MultiThreadedAugmenter'

    pred_gen = MultiThreadedAugmenter(pred_loader, None, num_processes=num_threads_in_multithreaded, num_cached_per_queue=2, pin_memory=False)
    
    # time_pred_dataloader = time.time()
    cases_done = []
    batch_num = 0
    dice_list = []
    for bi, pred_batch in enumerate(pred_gen):
        # logger.info('\nelapsed time_pred_dataloader:{}s'.format(time.time()-time_pred_dataloader)) # ?s for ? workers
        # bi = 0 # batch index
        # pred_batch = next(pred_gen)
        batch_cases = pred_batch['names']
        cases_done.extend(batch_cases)

        batch_num += 1

        images_all = pred_batch['data']
        images_all = torch.tensor(images_all, dtype=torch.float32).cuda()

        if model_config.addClin:
            clin_data = torch.tensor(pred_batch['clin_data']).cuda()
        else:
            clin_data = None

        # prep surv endpoints
        batch_surv_labs = pred_batch['surv']
        
        # prep gt: seg
        allSeg_all = pred_batch['seg']
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
            
        ## to determine what inputs to the model are, and run modelling
        if 'mmtm' in model_config.model_name.lower():
            images_ART = images_all[:,0,:,:,:].unsqueeze(1)
            images_PV = images_all[:,1,:,:,:].unsqueeze(1)
            if expe_config.addTumorMask and not expe_config.addLiverMask:
                images_ART = torch.cat([images_ART, tumorMasks_all[:, 0, :, :, :].unsqueeze(1)], dim=1)
                images_PV = torch.cat([images_PV, tumorMasks_all[:, 1, :, :, :].unsqueeze(1)], dim=1)
            elif expe_config.addLiverMask and not expe_config.addTumorMask:  
                images_ART = torch.cat([images_ART, LiverMasks_all[:, 0, :, :, :].unsqueeze(1)], dim=1)
                images_PV = torch.cat([images_PV,LiverMasks_all[:, 1, :, :, :].unsqueeze(1)], dim=1)
            elif expe_config.addLiverMask and expe_config.addTumorMask: 
                try: 
                    images_ART_ = torch.cat([images_ART, LiverTumorMasks_all[:, 0, :, :, :].unsqueeze(1)], dim=1)
                    images_PV_ = torch.cat([images_PV,LiverTumorMasks_all[:, 1, :, :, :].unsqueeze(1)], dim=1)
                except:
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
            model_res = model(images_ART, images_PV,clin_data)
        elif len(model_config.modality)==1 and 'ART' in model_config.modality:
            images_ART = images_all[:,0,:,:,:].unsqueeze(1)
            raise ValueError('if expe_config.addTumorMask: TBD')
            model_res = model(images_ART,None,clin_data)

        elif len(model_config.modality)==1 and 'PV' in model_config.modality:
            images_PV = images_all[:,1,:,:,:].unsqueeze(1)
            raise ValueError('if expe_config.addTumorMask: TBD')
            model_res = model(images_PV, None,clin_data)
            
        else:
            model_res = model(images_all,clin_data) # non_blocking=True # # 对于不含MMTM的模型，目前forward中还没添加clin_data=None
        # logger.info('\nelapsed time_pred_forward:{}s'.format(time.time()-time_pred_forward)) # ?s for ? workers


        #### extract logits for each case for computing c-index
        if expe_config.addSurvTask:
            logits_dict = model_res['logits']
            t_sne_dict = model_res['t_sne'] # keys: 'recur', 'death'. t_sne_dict['recur'] is a torch.tensor of shape (batch_size, 64) (64 is the second last fc layer neurons number)
            for task in ['recur', 'death']:
                if task in model_config.task_names:
                    batch_logits_list = [float(i) for i in logits_dict[task].detach().cpu().squeeze(1)]
                    batch_t_sne_list = [t_sne_dict[task].detach().cpu()[i].tolist() for i in range(len(batch_cases))] # [np.asarray(i) for i in t_sne_dict[task].detach().cpu().squeeze(1)]

                    for bci in range(len(batch_cases)): # batch case idx
                        pat_id = batch_cases[bci]
                        res_surv_out[task]['pat_res'][pat_id] = dict()

                        logits = batch_logits_list[bci]
                        res_surv_out[task]['pat_res'][pat_id]['logits'] = logits
                        
                        t_sne_ = batch_t_sne_list[bci]
                        res_surv_out[task]['pat_res'][pat_id]['t_sne'] = t_sne_

                        if mode!='infer':
                            surv_status = int(batch_surv_labs[bci]['{}_surv'.format(task)]['status']) # all_surv_dict[task][pat_id]['status']
                            surv_time = float(batch_surv_labs[bci]['{}_surv'.format(task)]['time'])  # all_surv_dict[task][pat_id]['time']
                            
                            res_surv_out[task]['pat_res'][pat_id]['status'] = surv_status
                            res_surv_out[task]['pat_res'][pat_id]['time'] = surv_time
        else:
            pass

        #### cal dice for each case
        if expe_config.addSegTask:
            # res_level = 1 # use -1 to select the seg pred of the second last resolution level

            seg_pred_list_art = model_res['seg_pred_art']
            seg_pred_list_pv = model_res['seg_pred_pv']

            featMapSize_list = model_res['featMapSize']
            assert type(seg_pred_list_art) is list, 'seg_pred_list_art should be a list'

            if not expe_config.addClsTask:
                batch_seg_pred_probs_art = F.softmax(seg_pred_list_art[-1], dim=1) #res_level
                batch_seg_pred_probs_pv = F.softmax(seg_pred_list_pv[-1], dim=1) #res_level# use -1 to select the seg pred of the second last resolution level

                featMapSize = featMapSize_list[1] #res_level
            else:
                batch_seg_pred_probs_art = F.softmax(seg_pred_list_art[0], dim=1) 
                batch_seg_pred_probs_pv = F.softmax(seg_pred_list_pv[0], dim=1) # use -1 to select the seg pred of the second last resolution level

                featMapSize = featMapSize_list[2] #res_level
            
            batch_seg_gt = F.interpolate(LiverTumorMasks_all, tuple(featMapSize[-3::]), mode='nearest') # if sizes same, no interpolation done.
            batch_seg_gt_art = batch_seg_gt[:,0,:,:,:].contiguous()
            batch_seg_gt_pv = batch_seg_gt[:,1,:,:,:].contiguous()
            for bci in range(len(batch_cases)): # batch case idx
                # bci=0
                pat_id = batch_cases[bci]
                # if len(model_config.modality) == 1:
                seg_gt_art = batch_seg_gt_art[bci,:,:,:]
                seg_pred_art = torch.argmax(batch_seg_pred_probs_art[bci,:,:,:,:], dim=0)
                seg_dice_art = seg_metrics.dice_coef_torch(seg_pred_art, seg_gt_art, c_list=[0,1,2])

                seg_gt_pv = batch_seg_gt_pv[bci,:,:,:]
                seg_pred_pv = torch.argmax(batch_seg_pred_probs_pv[bci,:,:,:,:], dim=0)  # out: [d, h, w]
                seg_dice_pv = seg_metrics.dice_coef_torch(seg_pred_pv, seg_gt_pv, c_list=[0,1,2])

                # collect res
                dice_art_liver = float(seg_dice_art[0].detach().cpu())
                dice_art_tumor = float(seg_dice_art[1].detach().cpu())

                dice_pv_liver = float(seg_dice_pv[0].detach().cpu())
                dice_pv_tumor = float(seg_dice_pv[1].detach().cpu())

                res_seg_out['pat_id'].append(pat_id)
                res_seg_out['dice_art_liver'].append(dice_art_liver)
                res_seg_out['dice_art_tumor'].append(dice_art_tumor)

                res_seg_out['dice_pv_liver'].append(dice_pv_liver)
                res_seg_out['dice_pv_tumor'].append(dice_pv_tumor)

                # ['dice_art_liver', 'dice_art_tumor', 'dice_pv_liver', 'dice_pv_tumor']
                wo.writerow([pat_id, round(dice_art_liver,3), round(dice_art_tumor,3), round(dice_pv_liver,3), round(dice_pv_tumor,3)])
                fo.flush()
        else:
            pass        

    # end of loop over batches
    # should be used after 'del ....', e.g. del model . then the GPU memory occupied by model will be released after running the below code line.
    with torch.no_grad():
        torch.cuda.empty_cache() # necessary to free GPU memory which are currently occuppied by cache like the objects currently deleted.
    
        
    #
    print('predicted: batch_num={}, patients_num={}'.format(batch_num, len(cases_done))) 

    ## cal metrics: surv ##   
    #  compute concordance index
    if mode!='infer':
        if expe_config.addSurvTask:
            for task in ['recur', 'death']:
                if task in model_config.task_names:
                    # recur_cindex = concordance_index(recur_time_list, recur_model_logits_list, recur_status_list) # this is wrong. negative of CNN output should be used. refer:https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
                    
                    # refer2:https://codeocean.com/capsule/5978670/tree/v1
                    time_list = [res_surv_out[task]['pat_res'][pat_id]['time'] for pat_id in cases_done]
                    negLogits_list = [-res_surv_out[task]['pat_res'][pat_id]['logits'] for pat_id in cases_done]
                    status_list = [res_surv_out[task]['pat_res'][pat_id]['status'] for pat_id in cases_done]
                    res_surv_out[task]['cindex'] = concordance_index(time_list, negLogits_list, status_list)
                else:
                    pass

        else:
            pass

        ### cal metrics: seg ##
        if expe_config.addSegTask:
            res_seg_out['mean_dice_art_liver'] = np.mean(res_seg_out['dice_art_liver'])
            res_seg_out['mean_dice_art_tumor'] = np.mean(res_seg_out['dice_art_tumor'])
            res_seg_out['mean_dice_pv_liver'] = np.mean(res_seg_out['dice_pv_liver'])
            res_seg_out['mean_dice_pv_tumor'] = np.mean(res_seg_out['dice_pv_tumor'])
        else:
            pass

    # close tmp I/O
    fo.close()

    # for return
    pred_out_final = dict()
    pred_out_final['surv'] = res_surv_out
    pred_out_final['seg'] = res_seg_out

    # return recur_cindex, death_cindex, recur_cindex_input, death_cindex_input
    return pred_out_final


