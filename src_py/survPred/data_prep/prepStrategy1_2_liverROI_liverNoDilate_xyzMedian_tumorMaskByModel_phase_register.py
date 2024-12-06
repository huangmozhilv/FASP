from multiprocessing.dummy import current_process
import os
from sre_constants import AT_LOC_BOUNDARY
import sys
sys.path.append(os.getcwd())
import warnings
warnings.filterwarnings("ignore")
import csv
from multiprocessing import Pool

import tqdm
import numpy as np
import pandas as pd
from medpy.io import load
from skimage.io import imsave
import cv2
from PIL import Image
import SimpleITK as sitk
from scipy import ndimage
from skimage.transform import resize

from ccToolkits import tinies
from ccToolkits import utils_img
import survPred.config as config
from importlib import reload
reload(config)


def find_box_coords(case_mask, axes=None):
    # if input x,y,z, the output will be [xmin, xmax, ymin,ymax,zmin,zmax]
    index = np.where(case_mask != 0)  # segmentation中 liver(不包括tumor)部分的index（coordinate）
    out = list()
    axes_to_output = list(range(len(case_mask.shape))) if axes is None else axes
    for idx, i in enumerate(index):
        if idx in axes_to_output:
            i_min = i.min()
            i_max = i.max()
            out.extend([i_min, i_max])
        else:
            pass
    return out

def phases_register(caseName):
    # pad the image ROIs to the max shape. padding = 0.??
    # resize?: not appropriate. as liver size will change for cirrhosis, et al. here will use crop then resize.
    
    # will crop to 48,321,449(z,y,x) to input to the 3Ddensenet. this size was determined by the densenet_3d and was debugged by Chao.

    # caseName = cases[0]
    # caseName = 'BA_001024450'
    # img
    sitk_pv = sitk.ReadImage(os.path.join(src_pv_imgROI_dir, '{}.nii.gz'.format(caseName)))
    sitk_art = sitk.ReadImage(os.path.join(src_art_imgROI_dir, '{}.nii.gz'.format(caseName)))

    pv_imgROI = sitk.GetArrayFromImage(sitk_pv)
    art_imgROI = sitk.GetArrayFromImage(sitk_art)

    # liver mask
    sitk_pv_liverMaskROI = sitk.ReadImage(os.path.join(src_pv_liverMaskROI_dir, '{}.nii.gz'.format(caseName)))
    sitk_art_liverMaskROI = sitk.ReadImage(os.path.join(src_art_liverMaskROI_dir, '{}.nii.gz'.format(caseName)))

    pv_liverMaskROI = sitk.GetArrayFromImage(sitk_pv_liverMaskROI)
    art_liverMaskROI = sitk.GetArrayFromImage(sitk_art_liverMaskROI)

    # tumor mask
    if src_pv_tumorMaskROI_dir is None:
        pass
    else:
        sitk_pv_tumorMaskROI = sitk.ReadImage(os.path.join(src_pv_tumorMaskROI_dir, '{}.nii.gz'.format(caseName)))
        sitk_art_tumorMaskROI = sitk.ReadImage(os.path.join(src_art_tumorMaskROI_dir, '{}.nii.gz'.format(caseName)))

        pv_tumorMaskROI = sitk.GetArrayFromImage(sitk_pv_tumorMaskROI)
        art_tumorMaskROI = sitk.GetArrayFromImage(sitk_art_tumorMaskROI)
        
    print('case:{}, before resize: pv_shape:{}, art_shape:{}'.format(caseName, pv_imgROI.shape, art_imgROI.shape)) # z,y,x

    # #### resize ####
    # # # pv
    # # pv_expecScale_z, pv_expecScale_y, pv_expecScale_x = resize_largest_shape[0]/pv_imgROI.shape[0], resize_largest_shape[1]/pv_imgROI.shape[1], resize_largest_shape[2]/pv_imgROI.shape[2]
    # # pv_targetScale = min([pv_expecScale_z, pv_expecScale_y, pv_expecScale_x])

    # # pv_resize_target_shape = [int(pv_targetScale*pv_imgROI.shape[0]), int(pv_targetScale*pv_imgROI.shape[1]), int(pv_targetScale*pv_imgROI.shape[2])]
    
    # # # art
    # # art_expecScale_z, art_expecScale_y, art_expecScale_x = resize_largest_shape[0]/art_imgROI.shape[0], resize_largest_shape[1]/art_imgROI.shape[1], resize_largest_shape[2]/art_imgROI.shape[2]
    # # art_targetScale = min([art_expecScale_z, art_expecScale_y, art_expecScale_x])

    # # art_resize_target_shape = [int(art_targetScale*art_imgROI.shape[0]), int(art_targetScale*art_imgROI.shape[1]), int(art_targetScale*art_imgROI.shape[2])]

    # # pv
    # pv_expecScale_z, pv_expecScale_y, pv_expecScale_x = resize_largest_shape[0]/pv_imgROI.shape[0], resize_largest_shape[1]/pv_imgROI.shape[1], resize_largest_shape[2]/pv_imgROI.shape[2]
    
    # # art
    # art_expecScale_z, art_expecScale_y, art_expecScale_x = resize_largest_shape[0]/art_imgROI.shape[0], resize_largest_shape[1]/art_imgROI.shape[1], resize_largest_shape[2]/art_imgROI.shape[2]

    # # target scale
    # targetScale = min([pv_expecScale_z, pv_expecScale_y, pv_expecScale_x] + [art_expecScale_z, art_expecScale_y, art_expecScale_x])

    # # resize
    # pv_resize_target_shape = [int(targetScale*pv_imgROI.shape[0]), int(targetScale*pv_imgROI.shape[1]), int(targetScale*pv_imgROI.shape[2])]

    # art_resize_target_shape = [int(targetScale*art_imgROI.shape[0]), int(targetScale*art_imgROI.shape[1]), int(targetScale*art_imgROI.shape[2])]

    # #
    # print('case:{}, after resize: pv_shape:{}, art_shape:{}'.format(caseName, pv_resize_target_shape, art_resize_target_shape)) # z,y,x

    # pv_imgROI = resize(pv_imgROI.astype(float),pv_resize_target_shape,order=3, mode="edge", anti_aliasing=False).astype(pv_imgROI.dtype)
    # art_imgROI = resize(art_imgROI.astype(float),art_resize_target_shape,order=3, mode="edge", anti_aliasing=False).astype(art_imgROI.dtype)

    # pv_liverMaskROI = resize(pv_liverMaskROI.astype(float), pv_resize_target_shape, order=0,anti_aliasing=False).astype(pv_liverMaskROI.dtype) #语义分割标签图像的数值发生了改变，数据类型也发生了改变，最关键的是数值也发生了改变。仔细查阅官方文档，添加anti_aliasing=False选项即可，因为默认是进行高斯滤波的；
    # art_liverMaskROI = resize(art_liverMaskROI.astype(float), art_resize_target_shape, order=0,anti_aliasing=False).astype(art_liverMaskROI.dtype) #语义分割标签图像的数值发生了改变，数据类型也发生了改变，最关键的是数值也发生了改变。仔细查阅官方文档，添加anti_aliasing=False选项即可，因为默认是进行高斯滤波的；
    # if src_pv_tumorMaskROI_dir is not None:
    #     pv_tumorMaskROI = resize(pv_tumorMaskROI.astype(float), pv_resize_target_shape, order=0,anti_aliasing=False).astype(pv_tumorMaskROI.dtype)
    #     art_tumorMaskROI = resize(art_tumorMaskROI.astype(float), art_resize_target_shape, order=0,anti_aliasing=False).astype(art_tumorMaskROI.dtype)
    # else:
    #     pass

    #### registration ####
    #  allign pv and art based on the centroid. according to 'liver_box_shapes.csv', the max difference between art and pv in x, y, z are 66, 40+, 11, respectively. The spacings are ~0.8mm, ~0.8mm, 5mm. then the size differences are 52.8mm, 32mm, 55mm, which are huge.

    pv_padded = utils_img.pad_to_shape(pv_imgROI, target_shape) # pad on the 6 ends. In this way, the pv and art are all centered to the same centroid.
    # np.sum(pv_padded[8:40, 53:267, 79:369]!=pv_imgROI) # test for caseName: 'BA_000493756'
    art_padded = utils_img.pad_to_shape(art_imgROI, target_shape)

    pv_liverMask_padded = utils_img.pad_to_shape(pv_liverMaskROI, target_shape)
    art_liverMask_padded = utils_img.pad_to_shape(art_liverMaskROI, target_shape)

    if src_pv_tumorMaskROI_dir is None:
        pass
    else:
        pv_tumorMask_padded = utils_img.pad_to_shape(pv_tumorMaskROI, target_shape)
        art_tumorMask_padded = utils_img.pad_to_shape(art_tumorMaskROI, target_shape)

    pv_bbox = find_box_coords(pv_liverMask_padded)
    art_bbox = find_box_coords(art_liverMask_padded)
    max_bbox = [min(pv_bbox[0], art_bbox[0]),max(pv_bbox[1], art_bbox[1]),min(pv_bbox[2], art_bbox[2]),max(pv_bbox[3], art_bbox[3]),min(pv_bbox[4], art_bbox[4]),max(pv_bbox[5], art_bbox[5])]

    # crop the registered ROIs
    pv_out = pv_padded[max_bbox[0]:(max_bbox[1]+1), max_bbox[2]:(max_bbox[3]+1), max_bbox[4]:(max_bbox[5]+1)]
    art_out = art_padded[max_bbox[0]:(max_bbox[1]+1), max_bbox[2]:(max_bbox[3]+1), max_bbox[4]:(max_bbox[5]+1)]

    pv_liverMask_out = pv_liverMask_padded[max_bbox[0]:(max_bbox[1]+1), max_bbox[2]:(max_bbox[3]+1), max_bbox[4]:(max_bbox[5]+1)]
    art_liverMask_out = art_liverMask_padded[max_bbox[0]:(max_bbox[1]+1), max_bbox[2]:(max_bbox[3]+1), max_bbox[4]:(max_bbox[5]+1)]

    if src_pv_tumorMaskROI_dir is None:
        pass
    else:
        pv_tumorMask_out = pv_tumorMask_padded[max_bbox[0]:(max_bbox[1]+1), max_bbox[2]:(max_bbox[3]+1), max_bbox[4]:(max_bbox[5]+1)]
        art_tumorMask_out = art_tumorMask_padded[max_bbox[0]:(max_bbox[1]+1), max_bbox[2]:(max_bbox[3]+1), max_bbox[4]:(max_bbox[5]+1)]

    # export img
    np.save(os.path.join(out_pv_imgROI_dir, '{}.npy'.format(caseName)), pv_out)
    np.save(os.path.join(out_art_imgROI_dir, '{}.npy'.format(caseName)), art_out)

    sitk_pv_out = sitk.GetImageFromArray(pv_out)
    # sitk_pv_out.SetOrigin(sitk_pv.GetOrigin())
    sitk_pv_out.SetSpacing(sitk_pv.GetSpacing())
    sitk_pv_out.SetDirection(sitk_pv.GetDirection())
    sitk.WriteImage(sitk_pv_out, os.path.join(out_pv_imgROI_dir, '{}.nii.gz'.format(caseName)))

    sitk_art_out = sitk.GetImageFromArray(art_out)
    # sitk_art_out.SetOrigin(sitk_art.GetOrigin())
    sitk_art_out.SetSpacing(sitk_art.GetSpacing())
    sitk_art_out.SetDirection(sitk_art.GetDirection())
    sitk.WriteImage(sitk_art_out, os.path.join(out_art_imgROI_dir, '{}.nii.gz'.format(caseName)))

    # export liverMask
    np.save(os.path.join(out_pv_liverMaskROI_dir, '{}.npy'.format(caseName)), pv_liverMask_out)
    np.save(os.path.join(out_art_liverMaskROI_dir, '{}.npy'.format(caseName)), art_liverMask_out)

    sitk_pv_liverMask_out = sitk.GetImageFromArray(pv_liverMask_out)
    # sitk_pv_liverMask_out.SetOrigin(sitk_pv.GetOrigin())
    sitk_pv_liverMask_out.SetSpacing(sitk_pv.GetSpacing())
    sitk_pv_liverMask_out.SetDirection(sitk_pv.GetDirection())
    sitk.WriteImage(sitk_pv_liverMask_out, os.path.join(out_pv_liverMaskROI_dir, '{}.nii.gz'.format(caseName)))

    sitk_art_liverMask_out = sitk.GetImageFromArray(art_liverMask_out)
    # sitk_art_liverMask_out.SetOrigin(sitk_art.GetOrigin())
    sitk_art_liverMask_out.SetSpacing(sitk_art.GetSpacing())
    sitk_art_liverMask_out.SetDirection(sitk_art.GetDirection())
    sitk.WriteImage(sitk_art_liverMask_out, os.path.join(out_art_liverMaskROI_dir, '{}.nii.gz'.format(caseName)))
    # print('phases registered for {}'.format(caseName))

    # export tumorMask
    if src_pv_tumorMaskROI_dir is None:
        pass
    else:
        np.save(os.path.join(out_pv_tumorMaskROI_dir, '{}.npy'.format(caseName)), pv_tumorMask_out)
        np.save(os.path.join(out_art_tumorMaskROI_dir, '{}.npy'.format(caseName)), art_tumorMask_out)

        sitk_pv_tumorMask_out = sitk.GetImageFromArray(pv_tumorMask_out)
        # sitk_pv_tumorMask_out.SetOrigin(sitk_pv.GetOrigin())
        sitk_pv_tumorMask_out.SetSpacing(sitk_pv.GetSpacing())
        sitk_pv_tumorMask_out.SetDirection(sitk_pv.GetDirection())
        sitk.WriteImage(sitk_pv_tumorMask_out, os.path.join(out_pv_tumorMaskROI_dir, '{}.nii.gz'.format(caseName)))

        sitk_art_tumorMask_out = sitk.GetImageFromArray(art_tumorMask_out)
        # sitk_art_tumorMask_out.SetOrigin(sitk_art.GetOrigin())
        sitk_art_tumorMask_out.SetSpacing(sitk_art.GetSpacing())
        sitk_art_tumorMask_out.SetDirection(sitk_art.GetDirection())
        sitk.WriteImage(sitk_art_tumorMask_out, os.path.join(out_art_tumorMaskROI_dir, '{}.nii.gz'.format(caseName)))
        # print('phases registered for {}'.format(caseName))

if __name__ == '__main__':
    NUM_OF_WORKERS = 100

    # ###
    print('----------step: register phases------------')
    # inputs
    resize_largest_shape=target_shape = [48,352,480] # [32, 248, 296] # [48,360,460] # [48,352,480] # [48,320,448] # z,y,x # devisible to 32; based on the 'target_shape' computed in survPred/data_prep/py1_data_prep.py
    # resize_largest_shape = [48,360,460]

    save_root = os.path.join(config.proj_root, 'data_cleaned_CECT_annotations/liverROI_3d_liverNoDilate_xyzMedian_no_zscore_tumorMaskByModel') # liverROI_3d_no_zscore, liverROI_3d_xyzSpacingMedian_no_zscore
    src_all_imgROI_dir = os.path.join(save_root, 'unalligned/') 
    src_pv_imgROI_dir = os.path.join(src_all_imgROI_dir, 'PV','imgROI')
    src_art_imgROI_dir = os.path.join(src_all_imgROI_dir, 'ART', 'imgROI')

    src_pv_liverMaskROI_dir = os.path.join(src_all_imgROI_dir, 'PV','liverMaskROI')
    src_art_liverMaskROI_dir = os.path.join(src_all_imgROI_dir, 'ART', 'liverMaskROI')

    src_pv_tumorMaskROI_dir = os.path.join(src_all_imgROI_dir, 'PV','tumorMaskROI')
    src_art_tumorMaskROI_dir = os.path.join(src_all_imgROI_dir, 'ART', 'tumorMaskROI')

    # outputs
    save_path = os.path.join(save_root, 'alligned/') 

    out_pv_imgROI_dir = os.path.join(save_path, 'PV','imgROI')
    tinies.sureDir(out_pv_imgROI_dir)
    out_art_imgROI_dir = os.path.join(save_path, 'ART', 'imgROI')
    tinies.sureDir(out_art_imgROI_dir)

    out_pv_liverMaskROI_dir = os.path.join(save_path, 'PV','liverMaskROI')
    tinies.sureDir(out_pv_liverMaskROI_dir)
    out_art_liverMaskROI_dir = os.path.join(save_path, 'ART', 'liverMaskROI')
    tinies.sureDir(out_art_liverMaskROI_dir)

    out_pv_tumorMaskROI_dir = os.path.join(save_path, 'PV','tumorMaskROI')
    tinies.sureDir(out_pv_tumorMaskROI_dir)
    out_art_tumorMaskROI_dir = os.path.join(save_path, 'ART', 'tumorMaskROI')
    tinies.sureDir(out_art_tumorMaskROI_dir)

    # 
    # cases = [i.strip('.nii.gz') for i in os.listdir(src_pv_imgROI_dir) if i.endswith('.nii.gz') and "BA" in i ]
    cases = [i.strip('.nii.gz') for i in os.listdir(src_pv_imgROI_dir) if i.endswith('.nii.gz')]
    # cases = ["BA_003176585"]
    # Running the Pool
    pool = Pool(NUM_OF_WORKERS)
    # caseName = cases[0]
    results = pool.map(phases_register, cases)