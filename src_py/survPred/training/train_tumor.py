# for survival prediction
import os
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
from survPred.tumorsurv_dataloader import tumorsurvDataLoader as survDataLoader, get_train_transform
from loss.surv_loss import cox_loss_cox_nnet
# from loss.surv_loss import CoxPHLoss
# from loss.surv_loss import NegativeLogLikelihood
# from loss.surv_loss import cox_loss_Olivier
# from loss.surv_loss import NegLogPartialLikelihood
from loss.lovasz_loss import lovasz_softmax
from loss.focal_loss import FocalLoss

# from loss.multiTaskLoss import multiTaskUncertaintySurvLoss
from loss.multiTaskLoss import multiTaskUncertaintySurvLoss_sameAsPaper


def train(expe_config, data_splits, model, model_config, train_config):
    torch.backends.cudnn.benchmark = True

    if expe_config.resume_ckp != '':
        logger.info('==> loading checkpoint: {}'.format(expe_config.resume_ckp))
        checkpoint = torch.load(expe_config.resume_ckp)

    model = nn.parallel.DataParallel(model)

    if train_config.use_gpu:
        model.cuda()  # required bofore optimizer?

    print(model)  # especially useful for debugging model structure.

    # lr
    lr = train_config.base_lr
    if_lr_decay = True if train_config.final_lr is not None else False
    if expe_config.resume_ckp != '':
        optimizer = checkpoint['optimizer']
    else:
        if train_config.multiTaskUncertainty == 'Kendall' and len(model_config.task_names) > 1:
            # MTU_module = multiTaskUncertaintySurvLoss(len(model_config.task_names))
            MTU_module = multiTaskUncertaintySurvLoss_sameAsPaper(len(model_config.task_names))
            optimizer = torch.optim.Adam([
                {'params': model.parameters()},
                {'params': MTU_module.parameters(), 'weight_decay': 0}
            ], lr=lr, weight_decay=train_config.weight_decay)  #
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=train_config.weight_decay)
    if train_config.lrScheduler == 'ReduceLROnPlateau':
        lrScheduler = ReduceLROnPlateau(optimizer, 'min', min_lr=0, patience=train_config.lrPatience, factor=0.5,
                                        verbose=True)
    elif train_config.lrScheduler == 'CyclicLR':
        lrScheduler = CyclicLR(optimizer, base_lr=train_config.base_lr, max_lr=0.01, step_size_down=100,
                               step_size_up=100, mode='triangular2', cycle_momentum=False)

    # loss_fn = cox_loss_cox_nnet()
    # loss_fn = CoxPHLoss()
    # loss_fn = NegativeLogLikelihood()
    # loss_fn = cox_loss_Olivier()
    if train_config.loss_type == 'Cox':
        # loss_fn = NegLogPartialLikelihood()
        loss_fn = cox_loss_cox_nnet()
    elif train_config.loss_type == 'CE':
        loss_fn = nn.CrossEntropyLoss()
    focal_loss = FocalLoss(gamma=2)

    ### prep train data
    train_ids = data_splits['dev'][expe_config.split]['train']
    # train_ids = train_ids[:5] # for debug

    num_threads_in_multithreaded = config.num_workers
    # num_threads_in_multithreaded = 1
    train_transforms = get_train_transform(expe_config.patch_size)

    seeds_candidates = list(range(0, 100))

    if train_config.epoch_method == 'infinite':
        # separate recur and death sampling as it is rather difficult to obtain a batch containing both negative and positive cases in respect to both recur and death since the positive cases are rare in the datasets. however, this is required to compute cox loss. as a result Chao adopt the round-robin method to update the model by updating the model with the loss of recur and loss of death in turn.
        train_loader = survDataLoader(model_config.task_names, config.data_root, expe_config.imgs_dir, train_ids,
                                      batch_size=model_config.batch_size, patch_size=expe_config.patch_size,
                                      mode='train', clin=model_config.addClin,
                                      num_threads_in_multithreaded=num_threads_in_multithreaded,
                                      return_incomplete=False, shuffle=True, infinite=True,
                                      if_prep_tumorMask=True)

        # train_gen = MultiThreadedAugmenter(train_loader, train_transforms, num_processes=num_threads_in_multithreaded, num_cached_per_queue=2, seeds=seeds_candidates[:num_threads_in_multithreaded], pin_memory=False)
        train_gen = MultiThreadedAugmenter(train_loader, train_transforms, num_processes=num_threads_in_multithreaded,
                                           num_cached_per_queue=2, pin_memory=False)
    elif train_config.epoch_method == 'finite':
        if train_config.return_incomplete:
            train_config.step_per_epoch = math.ceil(len(train_ids) / model_config.batch_size)
        else:
            train_config.step_per_epoch = math.floor(len(train_ids) / model_config.batch_size)

    ## TBD: code to be added for resuming training. refer to LELC_src/training/train.py by Chao.
    # if expe_config.resume_epoch > 0:
    #     start_epoch = expe_config.resume_epoch + 1
    #     iter_count = expe_config.resume_epoch * train_config.step_per_epoch + 1
    # else:
    #     start_epoch = 1
    #     iter_count = 1
    # logger.info('start epoch: {}'.format(start_epoch))
    # for epoch in range(start_epoch, train_config.max_epoch + 1):

    steps_per_epoch = train_config.step_per_epoch  # len(train_loader)
    print('steps per epoch:', steps_per_epoch)

    iter_count = 1
    train_stop = False
    epoch_loss = 0

    val_f = os.path.join(config.result_dir, 'epochs_val_cindices.csv')
    val_fo = open(val_f, 'w')
    val_wo = csv.writer(val_fo, delimiter=',')
    val_wo.writerow(['epoch', 'recur_cindex', 'death_cindex', 'split_tag'])
    val_fo.flush()

    tr_loss_win = 4  # 20
    tr_loss_queue = deque(maxlen=tr_loss_win)  # 20 epochs in queue for computing moving average of train loss
    for epoch in tqdm.tqdm(range(train_config.max_epoch)):
        # epoch = 0
        if not train_stop:
            # logger.info('    ----- training epoch {} -----'.format(epoch))

            if train_config.epoch_method == 'finite':
                train_loader = survDataLoader(model_config.task_names, config.data_root, expe_config.imgs_dir,
                                              train_ids, batch_size=model_config.batch_size,
                                              patch_size=expe_config.patch_size, mode='train',
                                              clin=model_config.addClin,
                                              num_threads_in_multithreaded=num_threads_in_multithreaded,
                                              return_incomplete=train_config.return_incomplete, shuffle=True,
                                              infinite=False, if_prep_tumorMask=expe_config.addSegTask)

                # train_gen = MultiThreadedAugmenter(train_loader, train_transforms, num_processes=num_threads_in_multithreaded, num_cached_per_queue=2, seeds=seeds_candidates[:num_threads_in_multithreaded], pin_memory=False)
                train_gen = MultiThreadedAugmenter(train_loader, train_transforms,
                                                   num_processes=num_threads_in_multithreaded, num_cached_per_queue=2,
                                                   pin_memory=False)
            else:
                pass

            train_progressor = ProgressBar(mode='Train_surv', epoch=epoch, total_epoch=train_config.max_epoch,
                                           model_name=model_config.model_name, total=steps_per_epoch)
            model.train()
            # train dataloader

            iter_loss_sum = 0
            pv_iter_cindex = 0
            art_iter_cindex = 0
            # pv_val_iter_loss_sum = 0

            # epoch_cindex = 0
            # for step, (fnames, images_pv, images_art,recur_surv, death_surv) in enumerate(train_loader):
            for step in tqdm.tqdm(range(steps_per_epoch)):
                # for step, train_batch in enumerate(train_gen):
                # step = 0
                # fnames, images_pv, images_art,recur_surv, death_surv = next(enumerate(train_loader))[1] # for debug
                train_progressor.current = step

                loss = torch.tensor(0, dtype=torch.float32)
                loss = loss.cuda()  # non_blocking=True

                # import ipdb; ipdb.set_trace()
                # time_dataloader = time.time()
                train_batch = next(train_gen)

                # dataloader_time = time.time()-time_dataloader
                # logger.info('\nelapsed time for dataloader:{}s'.format(dataloader_time)) # 0.02~0.5s for 7 workers

                sample_ids = train_batch['names']
                images_all = train_batch['data']  # b, c, z,y,x

                lab = train_batch['surv']
                if model_config.addClin:
                    clin_data = torch.tensor(train_batch['clin_data']).cuda()
                else:
                    clin_data = None

                if train_config.use_gpu:
                    images_all = torch.tensor(images_all).cuda()  # inputs to GPU # non_blocking=True
                    if expe_config.addSegTask:
                        tumorMasks_all = train_batch['seg']
                        tumorMasks_all = torch.tensor(tumorMasks_all).cuda()

                # model_out_logits = nn.LeakyReLU(-1)(model_out_logits) # why Olivier nature machine intelligence paper apply LeakyReLU like this?? TBD
                # time_iter = time.time()
                # import ipdb; ipdb.set_trace()
                if 'MMTM' in model_config.model_name or 'Mmtm' in model_config.model_name:
                    images_PV = images_all[:, 1, :, :, :].unsqueeze(1)
                    images_ART = images_all[:, 0, :, :, :].unsqueeze(1)
                    logits_dict, model_res_other = model(images_ART, images_PV, clin_data)
                else:
                    logits_dict, model_res_other = model(images_all,
                                                         clin_data)  # 对于不含MMTM的模型，目前forward中还没添加clin_data=None
                # forward_time = time.time()-time_iter
                # logger.info('\nelapsed time for forward pass:{}s'.format(forward_time))
                # logger.info('\nforward_time - dataloader_time:{}s'.format(forward_time-dataloader_time))

                if iter_count % 10 == 0:
                    # # for debug
                    # import SimpleITK as sitk
                    # for k in range(2):
                    #     sitk_img = sitk.GetImageFromArray(images_all.detach().cpu().numpy()[0,k,:])
                    #     sitk.WriteImage(sitk_img, '/data/cHuang/HCC_proj/results/{}_{}.nii.gz'.format(sample_ids[0],k))
                    train_utils.tb_images([images_all[0, 0, :, :, :], images_all[0, 1, :, :, :]], [False, False],
                                          ['pv', 'art'], iter_count,
                                          tag='Train_{}_epoch{}_step{}'.format(sample_ids[0], str(epoch).zfill(3),
                                                                               str(step).zfill(3)), img_is_RGB=False)

                #
                # if step>32:
                # import ipdb; ipdb.set_trace()

                if train_config.L1_reg_lambda is not None:
                    loss += train_config.L1_reg_lambda * l_reg(model, 'L1')
                if train_config.L2_reg_lambda is not None:
                    loss += train_config.L2_reg_lambda * l_reg(model, 'L2')

                # logger.info('{}'.format(str(sample_ids)))

                loss_tmp_list = list()

                # num_positives = 0
                for task_name in model_config.task_names:
                    # task_name = model_config.task_names[0]
                    surv = np.array([i['{}_surv'.format(task_name)] for i in lab])
                    surv_status_tensor = torch.tensor([i['status'] for i in surv]).cuda()
                    surv_time_tensor = torch.tensor([i['time'] for i in surv]).cuda()
                    # logger.info('\n--task:{}--'.format(task_name)) #detach().cpu().
                    # logger.info('\nlogits:{}'.format(str(logits_dict[task_name][:,0].tolist())))

                    # logger.info('status:{}'.format(str(surv_status_tensor.tolist()))) #detach().cpu().

                    logits_to_print = logits_dict[task_name][:, 0].tolist()
                    for i in range(len(logits_to_print)):
                        if abs(logits_to_print[i]) > 50:
                            logger.info('logits---{}: {}'.format(sample_ids[i], str(logits_to_print[i])))

                    if train_config.loss_type == 'Cox':
                        loss_tmp = loss_fn(logits_dict[task_name][:, 0], surv_time_tensor, surv_status_tensor)
                    elif train_config.loss_type == 'CE':
                        loss_tmp = loss_fn(torch.sigmoid(logits_dict[task_name][:, 0]), surv_status_tensor)
                    loss_tmp_list.append(loss_tmp)

                    # logger.info('\n{}_loss_tmp:{}'.format(task_name, str(loss_tmp.tolist())))
                    ystatus = surv_status_tensor.reshape(-1, 1)
                    # num_positives += torch.sum(ystatus)

                # if num_positives==0 and not expe_config.addSegTask:
                #     pass
                # else:
                current_loss_by_tasks = ['{}:{}'.format(model_config.task_names[i], round(loss_tmp_list[i].tolist(), 4))
                                         for i in range(len(loss_tmp_list))]

                if train_config.multiTaskUncertainty == 'Kendall' and len(model_config.task_names) > 1:
                    loss += MTU_module(loss_tmp_list).cuda()
                    for name, param in MTU_module.named_parameters():
                        logger.info('\nMTU params: {}'.format(str(param.data)))
                else:
                    multitask_loss = torch.tensor(0, dtype=torch.float32)
                    multitask_loss = multitask_loss.cuda()
                    for loss_tmp in loss_tmp_list:
                        multitask_loss += loss_tmp
                    multitask_loss /= len(model_config.task_names)
                    loss += multitask_loss

                if expe_config.addSegTask:
                    pv_tumorMask = tumorMasks_all[0]
                    art_tumorMask = tumorMasks_all[1]

                    pv_outputs = model_res_other['tumor_seg_out_PV']
                    art_outputs = model_res_other['tumor_seg_out_ART']

                    tumorSeg_loss_pv = lovasz_softmax(F.softmax(pv_outputs, dim=1), pv_tumorMask,
                                                      ignore=10) + focal_loss(pv_outputs, pv_tumorMask)
                    tumorSeg_loss_art = lovasz_softmax(F.softmax(art_outputs, dim=1), art_tumorMask,
                                                       ignore=10) + focal_loss(art_outputs, art_tumorMask)

                    config.writer.add_scalar('train/loss_step_tumorSeg_pv', float(tumorSeg_loss_pv), iter_count)
                    config.writer.add_scalar('train/loss_step_tumorSeg_art', float(tumorSeg_loss_art), iter_count)

                    tumorSeg_lambda = 0.25  # 0.5
                    loss += tumorSeg_lambda * tumorSeg_loss_pv
                    loss += tumorSeg_lambda * tumorSeg_loss_art

                # loss /= len(model_config.task_names) # multitask loss function will do the average computation.

                # # for debug
                # import ipdb; ipdb.set_trace()

                iter_loss_sum += float(
                    loss.item())  # .item() Returns the value of this tensor as a standard Python number
                optimizer.zero_grad()  # zero the parameter gradients
                loss.backward()  # backward
                optimizer.step()  # optimize

                ##########需要打印每个iter的，否则数据更新不及时，无法反映报错的iter的情况#########
                # this op costs much time.
                if expe_config.debug:
                    # 这部分代码在lungnetOlivier17，batchsize=10时会占用3GB显存
                    for name, param in model.named_parameters():  # 返回模型的参数
                        print(name + '_data')

                        print(param)
                        config.writer.add_histogram(name + '_data', param.data, iter_count)  # 参数的权值
                        print(name + '_grad')
                        print(param.grad)
                        config.writer.add_histogram(name + '_grad', param.grad,
                                                    iter_count)  # 参数的梯度，need to be after loss.backward to get the grads #如果返回'The histogram is empty, please file a bug report.'，根据报错信息回溯到tensorboardX的一段代码，发现是由于 if counts.size == 0 or limits.size == 0: raise ValueError('The histogram is empty, please file a bug report.')

                config.writer.add_scalar('train/loss_step', loss.item(), iter_count)

                # train_progressor.current_loss = loss.item()
                if expe_config.addSegTask:
                    train_progressor.current_loss = str(
                        current_loss_by_tasks + ['total:{}'.format(round(float(loss.item()), 4))] + [
                            'PV diceLoss:{}'.format(round(float(tumorSeg_loss_pv), 4))] + [
                            'ART diceLoss:{}'.format(round(float(tumorSeg_loss_art), 4))])
                else:
                    train_progressor.current_loss = str(
                        current_loss_by_tasks + ['total:{}'.format(round(float(loss.item()), 4))])

                train_progressor()

                config.writer.add_histogram('logits_' + task_name, logits_dict[task_name][:, 0], iter_count)
                iter_count += 1

            train_progressor.done(os.path.join(config.log_dir, 'train_progress_bar.txt'))

            # epoch_loss = sum(iter_loss_sum)/steps_per_epoch
            # epoch_cindex = sum(iter_cindex)/steps_per_epoch
            epoch_loss = iter_loss_sum / steps_per_epoch
            tr_loss_queue.append(epoch_loss)
            # epoch_cindex = iter_cindex/steps_per_epoch
            # pv_epoch_cindex = pv_iter_cindex/steps_per_epoch
            # art_epoch_cindex = art_iter_cindex/steps_per_epoch
            config.writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            if expe_config.addSegTask:
                logger.info(
                    'epoch {}: PV dice loss={}, ART dice loss={}'.format(epoch, round(float(tumorSeg_loss_pv), 4),
                                                                         round(float(tumorSeg_loss_art), 4)))
            else:
                pass
            config.writer.add_scalar('train/lr', lr, iter_count - 1)

            # inits
            Eval_bool = False
            Test_bool = False
            train_stop = False

            ###### lr decay
            if if_lr_decay:
                # lr = train_config.base_lr/(10**(epoch//5)) # train_config.base_lr/(10**(epoch//5))
                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr
                lr_old = optimizer.param_groups[0]['lr']

                if train_config.lrScheduler == 'ReduceLROnPlateau':
                    # tr_loss_ma = np.asarray(tr_loss_queue).mean()
                    # if epoch > tr_loss_win +1:
                    #     lrScheduler.step(tr_loss_ma)
                    # else:
                    #     pass
                    lrScheduler.step(epoch_loss)
                elif train_config.lrScheduler == 'CyclicLR':
                    lrScheduler.step()
                lr = optimizer.param_groups[0]['lr']

                if lr < lr_old:
                    logger.info('epoch:{}. lr is reduced from {} to {}. continue training'.format(epoch, lr_old, lr))

            if (lr < train_config.final_lr):
                Eval_bool = True
                Test_bool = True
                train_stop = True
                logger.info('epoch:{}. lr is reduced to {}. Will do the last evaluation'.format(epoch, lr))
            elif epoch == train_config.max_epoch - 1:
                Eval_bool = True
                Test_bool = True
                train_stop = True
                logger.info('reach to max epoch {}. Will do the last evaluation'.format(train_config.max_epoch))
            else:
                pass

            ### validation
            if expe_config.val and epoch >= train_config.start_val_epoch and (
                    epoch - train_config.start_val_epoch) % train_config.val_epoch_interval == 0:
                Eval_bool = True
            else:
                pass
            #
            if Eval_bool:
                # if epoch % 2 ==0: # train prediction is time-costly.
                #     train_eval_bool = True
                # else:
                #     train_eval_bool = False
                train_eval_bool = True

                dev_cindex_input = {
                    'recur': {
                        'time': list(),
                        'status': list(),
                        'model_logits': list()
                    },
                    'death': {
                        'time': list(),
                        'status': list(),
                        'model_logits': list()
                    }
                }

                ### prep val data
                # val_ids have to include both positive and negative cases.
                logger.info('start to validate after epoch {}'.format(epoch))

                # save best model in terms of validation cindex---------
                ## evaluate 'train ids'
                cases_to_predict_dict = {
                    # 'dev':data_splits['dev'][expe_config.split]['train'] + data_splits['dev'][expe_config.split]['val'],
                    'train': data_splits['dev'][expe_config.split]['train'],
                    'val': data_splits['dev'][expe_config.split]['val'],
                    'test': data_splits['test']
                }

                for split_tag, val_ids in cases_to_predict_dict.items():
                    if split_tag == 'train' and not train_eval_bool:
                        pass
                    else:
                        np.random.seed(epoch)
                        np.random.shuffle(val_ids)

                        val_out_dir = os.path.join(config.val_out_dir, 'epoch_{}'.format(epoch), split_tag)
                        # config.pred_type == 'val'
                        # recur_cindex, death_cindex = predict.predict(expe_config, val_ids, model, model_config, val_out_dir, mode='test_online')
                        pred_out = predict.predict(expe_config, val_ids, model, model_config, val_out_dir,
                                                   mode='test_online')

                        recur_cindex = pred_out['recur']['cindex']
                        death_cindex = pred_out['death']['cindex']
                        for task in ['recur', 'death']:
                            if task in model_config.task_names:
                                config.writer.add_scalar('{}/c_index/{}'.format(task, split_tag),
                                                         pred_out[task]['cindex'], epoch)

                                if train_eval_bool and split_tag in ['train',
                                                                     'val']:  # prep input for computing dev cindex
                                    cindex_input = pred_out[task]['cindex_input']

                                    dev_cindex_input[task]['time'].extend(cindex_input['time'])
                                    dev_cindex_input[task]['status'].extend(cindex_input['status'])
                                    dev_cindex_input[task]['model_logits'].extend(cindex_input['model_logits'])
                                else:
                                    pass
                            else:
                                pass

                        val_wo.writerow([epoch, recur_cindex, death_cindex, split_tag])
                        val_fo.flush()
                        logger.info(
                            'epoch {}: {}, sample_size={}, recur_cindex={}, death_cindex={}'.format(epoch, split_tag,
                                                                                                    len(val_ids),
                                                                                                    str(recur_cindex),
                                                                                                    str(death_cindex)))
                ## compute dev cindex: use out from val and train, to save time
                if train_eval_bool:
                    dev_cindices = dict()
                    for task in ['recur', 'death']:
                        if task in model_config.task_names:
                            dev_cindices[task] = concordance_index(dev_cindex_input[task]['time'],
                                                                   [-i for i in dev_cindex_input[task]['model_logits']],
                                                                   dev_cindex_input[task]['status'])
                            dev_sample_size = len(dev_cindex_input[task]['time'])
                            config.writer.add_scalar('{}/c_index/dev'.format(task), dev_cindices[task], epoch)
                        else:
                            dev_cindices[task] = None

                    val_wo.writerow([epoch, dev_cindices['recur'], dev_cindices['death'], 'dev'])
                    val_fo.flush()
                    logger.info(
                        'epoch {}: dev, sample_size={}, recur_cindex={}, death_cindex={}'.format(epoch, dev_sample_size,
                                                                                                 str(dev_cindices[
                                                                                                         'recur']), str(
                                dev_cindices['death'])))

                # if_SAVE = if_best or pv_valid_result[0] >= 0.6 or art_valid_result[0] >= 0.6
                if_SAVE = True  # save checkpoint
            else:
                if_SAVE = False

            # save checkpoint
            if if_SAVE or (epoch > 100 and epoch % train_config.save_epoch == 0) or epoch == train_config.max_epoch - 1:
                # ckp_path = os.path.join(config.ckpt_dir, 'epoch{}_{}_pv{}_art{}.pth.tar'.format(epoch, tinies.datestr(),pv_epoch_cindex, art_epoch_cindex))
                ckp_path = os.path.join(config.ckpt_dir, 'epoch{}.pth.tar'.format(epoch))
                torch.save({
                    'epoch': epoch,
                    'model': model,
                    'model_state_dict': model.state_dict(),
                    'optimizer': optimizer,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss
                }, ckp_path)
            else:
                pass

            ## testing with the final model
            # model_config.model_loc = os.path.join(config.ckpt_dir, 'epoch{}.pth.tar'.format(94))
            if expe_config.test and Test_bool:
                cases_to_predict_dict = {
                    'dev': data_splits['dev'][expe_config.split]['train'] + data_splits['dev'][expe_config.split][
                        'val'],
                    'train': data_splits['dev'][expe_config.split]['train'],
                    'val': data_splits['dev'][expe_config.split]['val'],
                    'test': data_splits['test']
                }
                for k, test_cases in cases_to_predict_dict.items():
                    # test_cases = cases_to_predict_dict['test']
                    pred_out_dir = os.path.join(config.test_out_dir, 'final_model_{}'.format(k))
                    tinies.sureDir(pred_out_dir)
                    config.pred_type = 'test'
                    # predict.predict(data_splits, model, model_config, mode='test_online')
                    recur_cindex, death_cindex = predict.predict(expe_config, test_cases, model, model_config,
                                                                 pred_out_dir, mode='test_online')
                    logger.info('--- predict for [final] model at epoch {} ----\n'.format(epoch))
                    logger.info(
                        '{}: recur_cindex={}, death_cindex={} \n'.format(k, str(recur_cindex), str(death_cindex)))

        else:
            ## testing with the best model
            # predict
            if expe_config.test and Test_bool:
                val_df = pd.read_csv(val_f)
                val_epochs = val_df['epoch'].values.tolist()

                recur_cindex_list = []
                for i in val_df['recur_cindex'].values.tolist():
                    if not np.isnan(val_df['recur_cindex'].values.tolist()[0]):
                        recur_cindex_list.append(i)
                    else:
                        recur_cindex_list.append(0)

                death_cindex_list = []
                for i in val_df['death_cindex'].values.tolist():
                    if not np.isnan(val_df['death_cindex'].values.tolist()[0]):
                        death_cindex_list.append(i)
                    else:
                        death_cindex_list.append(0)

                comb_cindex_list = [recur_cindex_list[i] + death_cindex_list[i] for i in range(len(recur_cindex_list))]

                # find the best model location
                best_epoch = val_epochs[comb_cindex_list.index(max(comb_cindex_list))]
                model_config.model_loc = os.path.join(config.ckpt_dir, 'epoch{}.pth.tar'.format(best_epoch))

                cases_to_predict_dict = {
                    'train': data_splits['dev'][expe_config.split]['train'],
                    'val': data_splits['dev'][expe_config.split]['val'],
                    'test': data_splits['test']
                }
                for k, test_cases in cases_to_predict_dict.items():
                    # test_cases = cases_to_predict_dict['test']
                    pred_out_dir = os.path.join(config.test_out_dir, 'best_model_{}'.format(k))
                    tinies.sureDir(pred_out_dir)
                    config.pred_type = 'test'
                    # predict.predict(data_splits, model, model_config, mode='test_online')
                    recur_cindex, death_cindex = predict.predict(expe_config, test_cases, model, model_config,
                                                                 pred_out_dir, mode='test_offline')
                    logger.info('--- predict for [best] model at epoch {} ----\n'.format(best_epoch))
                    logger.info(
                        '{}: recur_cindex={}, death_cindex={} \n'.format(k, str(recur_cindex), str(death_cindex)))

            val_fo.close()
            break