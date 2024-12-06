import os
import time
import math
import csv
from multiprocessing import Process, Queue
from collections import deque

import numpy as np
from PIL import Image
from medpy.io import load

import torch
import torch.nn as nn
import torch.nn.functional as F

import ccToolkits.logger as logger
from ccToolkits import tinies

import survPred.config as config

def tb_images(array_list, is_label_list, title_list, n_iter, tag='', img_is_RGB=False):
    # tensorboard batch images
    # image: d, h, w
    # pred: d, h, w
    # gt: d, h, w

    # import ipdb; ipdb.set_trace()

    colorsList = config.colorsList

    slice_indices = config.writer.chooseSlices(array_list[-1], is_label_list[-1])  # arrange the arrays as image1, image2,.., label1, label2,...

    figs = list()
    for i in range(len(array_list)):
        fig = config.writer.tensor2figure(array_list[i], slice_indices, colorsList=colorsList,is_label=is_label_list[i],  fig_title=title_list[i], img_is_RGB=img_is_RGB)
        figs.append(fig)

    config.writer.add_figure('figure/{}'.format(tag), figs, n_iter)


def l1_reg_all(model, reg_type=None):
    l1_reg = None

    for W in model.parameters():
        if l1_reg is None:
            l1_reg = torch.abs(W).sum()
        else:
            l1_reg = l1_reg + torch.abs(W).sum() # torch.abs(W).sum() is equivalent to W.norm(1)
    return l1_reg

def l1_reg_modules(model, reg_type=None):
    l1_reg = 0

    l1_reg += l1_reg_all(model.fc_omic)
    l1_reg += l1_reg_all(model.mm)

    return l1_reg