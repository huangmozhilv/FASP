import os
import sys

sys.path.append(os.getcwd())
import csv

import SimpleITK as sitk
import pandas as pd
import numpy as np
import tqdm
from lifelines.utils import concordance_index
# from sksurv.metrics import concordance_index_censored
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter

import torch

from survPred.training import train_utils
import survPred.config as config
from survPred.tumorsurv_dataloader import tumorsurvDataLoader as survDataLoader
import ccToolkits.tinies as tinies
import ccToolkits.logger as logger
from ccToolkits import utils_img
from tumorSurvPred.model_interpretability import get_cam, get_tsneplot, get_KMplot


# def predict(data_splits, model, model_config, mode='test_online'):
def predict(expe_config, cases, model, model_config, pred_out_dir, mode='test_online'):
    # mode:
    # --- test_online, test during training or right after training, so imediately use the trained model;
    # --- test_offline， test with saved checkpoint.

    # inputs
    # pv_imgROI_dir = os.path.join(expe_config.imgs_dir, 'PV', 'imgROI')
    # art_imgROI_dir = os.path.join(expe_config.imgs_dir, 'ART', 'imgROI')

    # pv_liverMask_dir = os.path.join(expe_config.imgs_dir, 'PV', 'liverMaskROI')
    # art_liverMask_dir = os.path.join(expe_config.imgs_dir, 'ART', 'liverMaskROI')

    if mode != 'infer':
        surv_dir = os.path.join(config.data_root, 'survival')

        recur_surv_f = os.path.join(surv_dir, 'surv_recurrence.csv')
        death_surv_f = os.path.join(surv_dir, 'surv_death.csv')

        # recur
        recur_surv_df = pd.read_csv(recur_surv_f)
        recur_surv_dict = dict()

        for case in cases:
            # case = cases[0]
            recur_surv_dict[case] = dict()
            recur_surv_dict[case]['status'] = recur_surv_df['status'][recur_surv_df['pat_id'] == case].values[0]
            recur_surv_dict[case]['time'] = recur_surv_df['time'][recur_surv_df['pat_id'] == case].values[0]

        # death
        death_surv_df = pd.read_csv(death_surv_f)
        death_surv_dict = dict()

        for case in cases:
            # case = cases[0]
            death_surv_dict[case] = dict()
            death_surv_dict[case]['status'] = death_surv_df['status'][death_surv_df['pat_id'] == case].values[0]
            death_surv_dict[case]['time'] = death_surv_df['time'][death_surv_df['pat_id'] == case].values[0]
    else:
        pass

    # outputs
    test_result_dir = pred_out_dir + '/testResult'
    tinies.newDir(test_result_dir)

    if mode == 'test_online':
        pass
    elif mode == 'test_offline' or mode == 'infer':
        ckpt_dir = model_config.model_loc
        ckpt = torch.load(ckpt_dir)
        # model.load_state_dict(torch.load(ckpt_dir))
        model = ckpt['model']
        model.load_state_dict(ckpt['model_state_dict'])
        # model_name = 'Resnet_2d-150-0000.pth'
        model_name = ckpt_dir.split('/')[-1]
        logger.info('current model: {}'.format(model_name))
    else:
        raise ValueError('Please specify the correct mode for testing')
    # load model
    model.cuda()
    model.eval()

    # print('evaluate on test images...')
    fo = open(os.path.join(pred_out_dir, 'test_cindex.csv'), 'w')
    wo = csv.writer(fo, delimiter=',')
    wo.writerow(['recur_cindex', 'death_cindex'])
    fo.flush()

    # init
    if mode != 'infer':
        recur_status_list = list()
        recur_time_list = list()

        death_status_list = list()
        death_time_list = list()
    else:
        pass
    if 'recur' in model_config.task_names:
        recur_model_logits_list = list()
    if 'death' in model_config.task_names:
        death_model_logits_list = list()

    # for case in tqdm.tqdm(cases):
    #     # case = cases[0]
    #     # case = 'BA_000412823'
    #     # logger.info('predicting {}'.format(case))

    #     # load img
    #     pv_imgROI_f = os.path.join(pv_imgROI_dir, case + '.nii.gz')
    #     art_imgROI_f = os.path.join(art_imgROI_dir, case + '.nii.gz')

    #     pv_imgROI = sitk.GetArrayFromImage(sitk.ReadImage(pv_imgROI_f))
    #     art_imgROI = sitk.GetArrayFromImage(sitk.ReadImage(art_imgROI_f))

    #     # load livermask
    #     pv_liverMask_f = os.path.join(pv_liverMask_dir, case + '.nii.gz')
    #     art_liverMask_f = os.path.join(art_liverMask_dir, case + '.nii.gz')

    #     pv_liverMask = sitk.GetArrayFromImage(sitk.ReadImage(pv_liverMask_f))
    #     art_liverMask = sitk.GetArrayFromImage(sitk.ReadImage(art_liverMask_f))

    #     # # load img
    #     # pv_imgROI_f = os.path.join(pv_imgROI_dir, case + '.npy')
    #     # art_imgROI_f = os.path.join(art_imgROI_dir, case + '.npy')

    #     # pv_imgROI = np.load(pv_imgROI_f)
    #     # art_imgROI = np.load(art_imgROI_f)

    #     # # load livermask
    #     # pv_liverMask_f = os.path.join(pv_liverMask_dir, case + '.npy')
    #     # art_liverMask_f = os.path.join(art_liverMask_dir, case + '.npy')

    #     # pv_liverMask = np.load(pv_liverMask_f)
    #     # art_liverMask = np.load(art_liverMask_f)

    #     # # crop imgROI
    #     # pv_imgROI = pv_imgROI * pv_liverMask.astype(pv_imgROI.dtype)
    #     # art_imgROI = art_imgROI * art_liverMask.astype(art_imgROI.dtype)

    #     # apply model to infer
    #     # images_all_tmp = torch.cat((torch.unsqueeze(torch.tensor(pv_imgROI), 0), torch.unsqueeze(torch.tensor(art_imgROI), 0)), dim=0)
    #     images_all_tmp = np.concatenate((np.expand_dims(pv_imgROI, 0), np.expand_dims(art_imgROI, 0)), axis=0)
    #     liverMask_all_tmp = np.concatenate((np.expand_dims(pv_liverMask, 0), np.expand_dims(art_liverMask, 0)), axis=0)

    #     images_all = np.zeros([model_config.numChannels] + expe_config.patch_size, dtype=np.float32)
    #     liverMask_all = np.zeros([model_config.numChannels] + expe_config.patch_size, dtype=np.uint8)
    #     for chan in range(model_config.numChannels):
    #         images_all[chan] = utils_img.pad_to_shape(images_all_tmp[chan], expe_config.patch_size)
    #         liverMask_all[chan] = utils_img.pad_to_shape(liverMask_all_tmp[chan], expe_config.patch_size)
    #     images_all = utils_img.vol_intensity_normTo0to1(images_all, liverMask_all, tgt_range=config.liver_HU_range)
    #     # images_all = utils_img.vol_intensity_normTo0to1_with_specifiedMax(images_all,config.liver_HU_range[1])

    #     images_all = torch.tensor(images_all, dtype=torch.float32).unsqueeze(0).cuda()
    #     logits_dict, model_res_other = model(images_all) # non_blocking=True
    #     if mode!='test_online':
    #         train_utils.tb_images([images_all[0, 0, :, :,:], images_all[0, 1, :, :,:]], [False, False], ['pv', 'art'], 99999, tag='Test_{}'.format(case), img_is_RGB=False)
    #     # model_out_logits = float(model_out_logits.detach().cpu()[0])

    #     if mode!='infer':
    #         if 'recur' in model_config.task_names:
    #             recur_logits = float(logits_dict['recur'].detach().cpu()[0])

    #             # recur_status, recur_time = recur_surv_dict[case]
    #             recur_status = recur_surv_dict[case]['status']
    #             recur_time = recur_surv_dict[case]['time']
    #             recur_status_list.append(recur_status)
    #             recur_time_list.append(recur_time)
    #             # recur_model_logits_list.append(-recur_logits) # '-' should be added. since the risk is negatively associated with survival.
    #             recur_model_logits_list.append(recur_logits)
    #         else:
    #             pass
    #         if 'death' in model_config.task_names:
    #             death_logits = float(logits_dict['death'].detach().cpu()[0])

    #             # death_status, death_time = death_surv_dict[case]
    #             death_status = death_surv_dict[case]['status']
    #             death_time = death_surv_dict[case]['time']
    #             death_status_list.append(death_status)
    #             death_time_list.append(death_time)
    #             # death_model_logits_list.append(-death_logits)# '-' should be added. since the risk is negatively associated with survival.
    #             death_model_logits_list.append(death_logits)
    #         else:
    #             pass
    #     else:
    #         pass

    #     # logger.info('Case {} test done'.format(case))
    #     torch.cuda.empty_cache() # necessary. otherwise this code will result in increasing GPU occupation.
    pred_batch_size = 10  # model_config.batch_size
    num_threads_in_multithreaded = 8

    pred_loader = survDataLoader(model_config.task_names, config.data_root, expe_config.imgs_dir, cases,
                                 batch_size=pred_batch_size, patch_size=expe_config.patch_size,
                                 mode='infer', clin=model_config.addClin,
                                 num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=True,
                                 shuffle=False, infinite=False,
                                 if_prep_tumorMask=False)  # 'num_threads_in_multithreaded' here should be the same as that in 'MultiThreadedAugmenter'

    pred_gen = MultiThreadedAugmenter(pred_loader, None, num_processes=num_threads_in_multithreaded,
                                      num_cached_per_queue=2, pin_memory=False)

    for bi, pred_batch in enumerate(pred_gen):
        batch_cases = pred_batch['names']
        images_all = pred_batch['data']
        # images_all = torch.tensor(images_all, dtype=torch.float32).unsqueeze(0).cuda()
        images_all = torch.tensor(images_all, dtype=torch.float32).cuda()

        if model_config.addClin:
            clin_data = torch.tensor(pred_batch['clin_data']).cuda()
        else:
            clin_data = None

        if 'MMTM' in model_config.model_name or 'Mmtm' in model_config.model_name:
            images_PV = images_all[:, 1, :, :, :].unsqueeze(1)
            images_ART = images_all[:, 0, :, :, :].unsqueeze(1)
            logits_dict, model_res_other = model(images_ART, images_PV, clin_data)
        else:
            logits_dict, model_res_other = model(images_all,
                                                 clin_data)  # non_blocking=True # # 对于不含MMTM的模型，目前forward中还没添加clin_data=None

        if mode != 'test_online':
            train_utils.tb_images([images_all[0, 0, :, :, :], images_all[0, 1, :, :, :]], [False, False], ['pv', 'art'],
                                  99999, tag='Test_{}'.format(case), img_is_RGB=False)
        # model_out_logits = float(model_out_logits.detach().cpu()[0])
        if 'recur' in model_config.task_names:
            recur_logits_list = [float(i) for i in logits_dict['recur'].detach().cpu().squeeze(1)]
        else:
            pass
        if 'death' in model_config.task_names:
            death_logits_list = [float(i) for i in logits_dict['death'].detach().cpu().squeeze(1)]
        else:
            pass
        for bci in range(len(batch_cases)):  # batch case idx
            case = batch_cases[bci]
            if mode != 'infer':
                if 'recur' in model_config.task_names:
                    recur_logits = recur_logits_list[bci]

                    # recur_status, recur_time = recur_surv_dict[case]
                    recur_status = recur_surv_dict[case]['status']
                    recur_time = recur_surv_dict[case]['time']
                    recur_status_list.append(recur_status)
                    recur_time_list.append(recur_time)
                    # recur_model_logits_list.append(-recur_logits) # '-' should be added. since the risk is negatively associated with survival.
                    recur_model_logits_list.append(recur_logits)
                else:
                    pass
                if 'death' in model_config.task_names:
                    death_logits = death_logits_list[bci]

                    # death_status, death_time = death_surv_dict[case]
                    death_status = death_surv_dict[case]['status']
                    death_time = death_surv_dict[case]['time']
                    death_status_list.append(death_status)
                    death_time_list.append(death_time)
                    # death_model_logits_list.append(-death_logits)# '-' should be added. since the risk is negatively associated with survival.
                    death_model_logits_list.append(death_logits)
                else:
                    pass
            else:
                pass

        torch.cuda.empty_cache()  # necessary. otherwise this code will result in increasing GPU occupation.
    #
    print('predicted {} patients'.format(len(cases)))

    ## compute concordance index ##
    if mode != 'infer':
        if 'recur' in model_config.task_names:
            # recur_cindex = concordance_index(recur_time_list, recur_model_logits_list, recur_status_list) # this is wrong. negative of CNN output should be used. refer:https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
            recur_cindex = concordance_index(recur_time_list, [-i for i in recur_model_logits_list], recur_status_list)
            # recur_cindex = concordance_index_censored(np.array(recur_status_list).astype(bool),np.array(recur_time_list), np.array(recur_model_logits_list))[0]
            recur_cindex_input = {
                'time': recur_time_list,
                'status': recur_status_list,
                'model_logits': recur_model_logits_list
            }
        else:
            recur_cindex = None
            recur_cindex_input = None
        if 'death' in model_config.task_names:
            # death_cindex = concordance_index(death_time_list, death_model_logits_list, death_status_list) # this is wrong. negative of CNN output should be used. refer:https://stats.stackexchange.com/questions/352183/use-median-survival-time-to-calculate-cph-c-statistic/352435#352435
            death_cindex = concordance_index(death_time_list, [-i for i in death_model_logits_list], death_status_list)
            # death_cindex = concordance_index_censored(np.array(death_status_list).astype(bool),np.array(death_time_list), np.array(death_model_logits_list))[0]
            death_cindex_input = {
                'time': death_time_list,
                'status': death_status_list,
                'model_logits': death_model_logits_list
            }
        else:
            death_cindex = None
            death_cindex_input = None
    else:
        recur_cindex, death_cindex = None, None

    wo.writerow([recur_cindex, death_cindex])
    # logger.info('predict.py cindex ----- recur={}; death={}'.format(recur_cindex, death_cindex))
    fo.flush()
    fo.close()

    out = {
        'recur': {
            'cindex': recur_cindex,
            'cindex_input': recur_cindex_input,
        },
        'death': {
            'cindex': death_cindex,
            'cindex_input': death_cindex_input
        }
    }

    # return recur_cindex, death_cindex, recur_cindex_input, death_cindex_input
    return out


