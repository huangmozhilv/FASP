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

from ccToolkits import tinies
from ccToolkits import utils_img
import survPred.config as config
from importlib import reload
reload(config)


dila_iters_3D = config.dila_iters_3D #1

# batch1&2 spacing: median_xyz [0.68359,0.68359,5]
# batch3 spacing: median_x or median_y=[0.68], range_x or range_y = [0.53,0.87]; median_z = 5, range_z = [-5,5]
# uni_spacing = [0.684, 0.684, 1] # set to retain details
liver_HU_range = config.liver_HU_range 
uni_spacing = [0.684,0.684,5] # uni_spacing = [None, None, 5] # [1,1,2.5] # [0.684,0.684,3] # [1,1,1] # [None, None, 5] only resample on z-axis to minimize computation cost, the rationale is x-y plane pixel spacing variations are small. In b1to2, x/y spacing range: (0.531 ~ 0.871), mean=0.689; median=0.684. in this way, the output x-y shape are all 512*512, no more need to apply cropping which could be complex. anyway, random scale is used to augment the training samples.


def find_centroid(image, flag):
    [x, y, z] = np.where(image == flag)
    # centroid_x = int(np.mean(x)) # from def find_centroid_z() of xyy
    # centroid_y = int(np.mean(y)) # from def find_centroid_z() of xyy
    # centroid_z = int(np.mean(z)) # from def find_centroid_z() of xyy

    centroid_x = int((np.min(x)+np.max(x))/2)
    centroid_y = int((np.min(y)+np.max(y))/2)
    centroid_z = int((np.min(z)+np.max(z))/2)

    return centroid_x,centroid_y,centroid_z


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

def crop_liverBbox_3d(caseName):
    # crop img and liver mask to bounding box of the livermask
    # caseName = caselist[0]
    # caseName = 'BA_001024450'
    try:
        # input files
        # print('cropping 3d ROI for {}'.format(caseName))
        img_pv_f = os.path.join(imgs_dir, '{}_{}_0000.nii.gz'.format(caseName, 'V'))
        img_art_f = os.path.join(imgs_dir, '{}_{}_0000.nii.gz'.format(caseName, 'A'))

        # load
        sitk_img_art = sitk.ReadImage(img_art_f) # x,y,z
        img_art = sitk.GetArrayFromImage(sitk_img_art) # z,y,x
        A_spacing = sitk_img_art.GetSpacing() # x,y,z

        sitk_img_pv = sitk.ReadImage(img_pv_f) # x,y,z
        img_pv = sitk.GetArrayFromImage(sitk_img_pv) # z,y,x
        V_spacing = sitk_img_pv.GetSpacing() # x,y,z

        # ## load image and tumor mask and check if to include this case

        # # these cases have unequal sizes of ART and PV. BA_003073088, BA_003009644, BA_003624270, BA_002996451, BA_003399078, BA_003901908, BA_002748296, BA_003504421, BA_003943200

        ## images resample to unified spacing
        uni_spacing_art = utils_img.interpolate_spacing(uni_spacing, A_spacing)
        img_art_resamp = utils_img.resample2fixedSpacing(img_art, A_spacing[::-1], uni_spacing_art[::-1], img_art_f, interpolate_method=sitk.sitkBSpline) # input: z,y,x; output: z,y,x
        img_art = np.transpose(img_art_resamp, [2,1,0]).astype(np.float32) # x,y,z

        uni_spacing_pv = utils_img.interpolate_spacing(uni_spacing, V_spacing)
        img_pv_resamp = utils_img.resample2fixedSpacing(img_pv, V_spacing[::-1], uni_spacing_pv[::-1], img_pv_f, interpolate_method=sitk.sitkBSpline) # output: z,y,x
        img_pv = np.transpose(img_pv_resamp, [2,1,0]).astype(np.float32) # x,y,z


        ## load liver mask
        liver_mask_pv_f = os.path.join(liver_mask_path, '{}_{}.nii.gz'.format(caseName,'V'))
        liver_mask_art_f = os.path.join(liver_mask_path,'{}_{}.nii.gz'.format(caseName, 'A'))

        liver_mask_pv, _ = load(liver_mask_pv_f)  # load nii mask data. x,y,z
        liver_mask_art, _ = load(liver_mask_art_f) # x,y,z

        liver_mask_pv[np.where(liver_mask_pv!=0)] = 1
        liver_mask_art[np.where(liver_mask_art!=0)] = 1

        # liver masks resample to unified spacing
        liver_mask_pv_resamp = utils_img.resample2fixedSpacing(np.transpose(liver_mask_pv, [2,1,0]), V_spacing[::-1], uni_spacing_pv[::-1], liver_mask_pv_f, interpolate_method=sitk.sitkNearestNeighbor) # output: z,y,x
        liver_mask_pv = np.transpose(liver_mask_pv_resamp, [2,1,0]) # out: x,y,z

        liver_mask_art_resamp = utils_img.resample2fixedSpacing(np.transpose(liver_mask_art, [2,1,0]), A_spacing[::-1], uni_spacing_art[::-1], liver_mask_art_f, interpolate_method=sitk.sitkNearestNeighbor) # output: z,y,x
        liver_mask_art = np.transpose(liver_mask_art_resamp, [2,1,0]) # out:x,y,z

        ## load tumor mask
        if tumor_mask_path is None:
            pass
        else:
            tumor_mask_pv_f = os.path.join(tumor_mask_path, caseName,'{}_{}_lab.nii.gz'.format(caseName,'V'))
            tumor_mask_art_f = os.path.join(tumor_mask_path,caseName,'{}_{}_lab.nii.gz'.format(caseName, 'A'))

            tumor_mask_pv, _ = load(tumor_mask_pv_f)  # load nii mask data. x,y,z
            tumor_mask_art, _ = load(tumor_mask_art_f) # x,y,z

            tumor_mask_pv[np.where(tumor_mask_pv!=0)] = 1
            tumor_mask_art[np.where(tumor_mask_art!=0)] = 1

            # tumor masks resample to unified spacing
            tumor_mask_pv_resamp = utils_img.resample2fixedSpacing(np.transpose(tumor_mask_pv, [2,1,0]), V_spacing[::-1], uni_spacing_pv[::-1], tumor_mask_pv_f, interpolate_method=sitk.sitkNearestNeighbor) # output: z,y,x
            tumor_mask_pv = np.transpose(tumor_mask_pv_resamp, [2,1,0]) # out: x,y,z

            tumor_mask_art_resamp = utils_img.resample2fixedSpacing(np.transpose(tumor_mask_art, [2,1,0]), A_spacing[::-1], uni_spacing_art[::-1], tumor_mask_art_f, interpolate_method=sitk.sitkNearestNeighbor) # output: z,y,x
            tumor_mask_art = np.transpose(tumor_mask_art_resamp, [2,1,0]) # out:x,y,z
        
            ## ensure all tumor voxels are also liver.
            liver_mask_pv[tumor_mask_pv==1] = 1
            liver_mask_art[tumor_mask_art==1] = 1

        # # dilate
        # struct = ndimage.generate_binary_structure(3,1)
        # liver_mask_pv = ndimage.binary_dilation(liver_mask_pv, structure=struct, iterations=dila_iters_3D).astype(np.int8) # liver_mask_pv.dtype
        # liver_mask_art = ndimage.binary_dilation(liver_mask_art, structure=struct, iterations=dila_iters_3D).astype(np.int8) # liver_mask_art.dtype

        ## intensity clip and norm: set pixels outside of liver to be 0; apply wl and ww to pixels in the liver
        img_art = np.clip(img_art, liver_HU_range[0], liver_HU_range[1]) # x,y,z

        img_pv = np.clip(img_pv, liver_HU_range[0], liver_HU_range[1]) # x,y,z

        ## find liver centroid
        liver_centroid_f = open(os.path.join(liver_centroid_path,caseName+'.txt'),'a')
        
        pv_liver_centroid = find_centroid(liver_mask_pv,1)  # find liver centroid
        np.savetxt(liver_centroid_f, pv_liver_centroid, fmt="%d")  # x,y,z
        pv_liver_cen_z = pv_liver_centroid[2]
        art_liver_centroid = find_centroid(liver_mask_art,1)  # find liver centroid
        np.savetxt(liver_centroid_f, art_liver_centroid, fmt="%d")  # x,y,z
        art_liver_cen_z = art_liver_centroid[2]

        liver_centroid_f.close() ## f look like: pv_x,pv_y,pv_z,art_x,art_y,art_z?

        pv_liver_coords = find_box_coords(liver_mask_pv) # x_min, x_max, y_min, y_max, z_min, z_max
        art_liver_coords = find_box_coords(liver_mask_art)

        ## extract ROI
        # img
        pv_imgROI = img_pv[pv_liver_coords[0]:pv_liver_coords[1]+1, pv_liver_coords[2]:pv_liver_coords[3]+1, pv_liver_coords[4]:pv_liver_coords[5]+1] # img_pv: x,y,z
        art_imgROI = img_art[art_liver_coords[0]:art_liver_coords[1]+1, art_liver_coords[2]:art_liver_coords[3]+1, art_liver_coords[4]:art_liver_coords[5]+1]

        # liver mask
        pv_liverMaskROI = liver_mask_pv[pv_liver_coords[0]:pv_liver_coords[1]+1, pv_liver_coords[2]:pv_liver_coords[3]+1, pv_liver_coords[4]:pv_liver_coords[5]+1] # img_pv: x,y,z
        art_liverMaskROI = liver_mask_art[art_liver_coords[0]:art_liver_coords[1]+1, art_liver_coords[2]:art_liver_coords[3]+1, art_liver_coords[4]:art_liver_coords[5]+1]

        # tumor mask
        if tumor_mask_path is None:
            pass
        else:
            pv_tumorMaskROI = tumor_mask_pv[pv_liver_coords[0]:pv_liver_coords[1]+1, pv_liver_coords[2]:pv_liver_coords[3]+1, pv_liver_coords[4]:pv_liver_coords[5]+1] # img_pv: x,y,z
            art_tumorMaskROI = tumor_mask_art[art_liver_coords[0]:art_liver_coords[1]+1, art_liver_coords[2]:art_liver_coords[3]+1, art_liver_coords[4]:art_liver_coords[5]+1]

        pv_bbox_shape = list(pv_imgROI.shape)
        art_bbox_shape = list(art_imgROI.shape)

        # export
        liver_box_fo = open(os.path.join(liver_box_path, caseName+'.txt'), 'w')
        np.savetxt(liver_box_fo, pv_bbox_shape + art_bbox_shape, fmt="%d") # caseName, pv_x, pv_y, pv_z, art_x, art_y, art_z
        liver_box_fo.close()

        pv_imgROI = np.transpose(pv_imgROI,axes=[2,1,0]) # to z,y,x
        art_imgROI = np.transpose(art_imgROI, axes=[2,1,0]) # to z,y,x

        pv_liverMaskROI = np.transpose(pv_liverMaskROI,axes=[2,1,0])
        art_liverMaskROI = np.transpose(art_liverMaskROI,axes=[2,1,0])

        pv_tumorMaskROI = np.transpose(pv_tumorMaskROI,axes=[2,1,0])
        art_tumorMaskROI = np.transpose(art_tumorMaskROI,axes=[2,1,0])

        # np.save(os.path.join(pv_imgROI_dir, '{}.npy'.format(caseName)), pv_imgROI)
        # np.save(os.path.join(art_imgROI_dir, '{}.npy'.format(caseName)), art_imgROI)
        # img
        sitk_pv_imgROI = sitk.GetImageFromArray(pv_imgROI)
        sitk_pv_imgROI.SetOrigin(sitk_img_pv.GetOrigin())
        sitk_pv_imgROI.SetSpacing(uni_spacing_pv)
        sitk_pv_imgROI.SetDirection(sitk_img_pv.GetDirection())
        sitk.WriteImage(sitk_pv_imgROI, os.path.join(pv_imgROI_dir, '{}.nii.gz'.format(caseName))) # out: x,y,z

        sitk_art_imgROI = sitk.GetImageFromArray(art_imgROI)
        sitk_art_imgROI.SetOrigin(sitk_img_art.GetOrigin())
        sitk_art_imgROI.SetSpacing(uni_spacing_art)
        sitk_art_imgROI.SetDirection(sitk_img_art.GetDirection())
        sitk.WriteImage(sitk_art_imgROI, os.path.join(art_imgROI_dir, '{}.nii.gz'.format(caseName)))

        # liver mask
        sitk_pv_liverMaskROI = sitk.GetImageFromArray(pv_liverMaskROI)
        sitk_pv_liverMaskROI.SetOrigin(sitk_img_pv.GetOrigin())
        sitk_pv_liverMaskROI.SetSpacing(uni_spacing_pv)
        sitk_pv_liverMaskROI.SetDirection(sitk_img_pv.GetDirection())
        sitk.WriteImage(sitk_pv_liverMaskROI, os.path.join(pv_liverMaskROI_dir, '{}.nii.gz'.format(caseName))) # out: x,y,z

        sitk_art_liverMaskROI = sitk.GetImageFromArray(art_liverMaskROI)
        sitk_art_liverMaskROI.SetOrigin(sitk_img_art.GetOrigin())
        sitk_art_liverMaskROI.SetSpacing(uni_spacing_art)
        sitk_art_liverMaskROI.SetDirection(sitk_img_art.GetDirection())
        sitk.WriteImage(sitk_art_liverMaskROI, os.path.join(art_liverMaskROI_dir, '{}.nii.gz'.format(caseName))) # out: x,y,z

        # tumor mask
        if tumor_mask_path is None:
            pass
        else:
            sitk_pv_tumorMaskROI = sitk.GetImageFromArray(pv_tumorMaskROI)
            sitk_pv_tumorMaskROI.SetOrigin(sitk_img_pv.GetOrigin())
            sitk_pv_tumorMaskROI.SetSpacing(uni_spacing_pv)
            sitk_pv_tumorMaskROI.SetDirection(sitk_img_pv.GetDirection())
            sitk.WriteImage(sitk_pv_tumorMaskROI, os.path.join(pv_tumorMaskROI_dir, '{}.nii.gz'.format(caseName))) # out: x,y,z

            sitk_art_tumorMaskROI = sitk.GetImageFromArray(art_tumorMaskROI)
            sitk_art_tumorMaskROI.SetOrigin(sitk_img_art.GetOrigin())
            sitk_art_tumorMaskROI.SetSpacing(uni_spacing_art)
            sitk_art_tumorMaskROI.SetDirection(sitk_img_art.GetDirection())
            sitk.WriteImage(sitk_art_tumorMaskROI, os.path.join(art_tumorMaskROI_dir, '{}.nii.gz'.format(caseName))) # out: x,y,z

        print('cropped 3D ROI for {}'.format(caseName))
    except Exception as e:
        print('error with {} when cropping the 3D ROI'.format(caseName))
        print(e)



if __name__ == '__main__':
    NUM_OF_WORKERS = 70

    #------ crop liver ROI 3d -------
    # inputs
    batch_tags = ['b1to2','b3','b4', 'HUZHOU',"TCGA_LIHC","SRRH"] #'b1to2','b3','b4', 'HUZHOU',"TCGA_LIHC","SRRH"
    case_choose = None# ["BA_000774856", "BA_000847313", "BA_000895519", "BA_001024450", "BA_001200058", "BA_001539781", "BA_002980736", "BA_002044583", "BA_003329343", "BA_004093829", "BA_005419751", "BA_003505205"]
    # outputs
    # proj_root = os.path.dirname(os.getcwd())/HCC_std1/HCC_proj
    proj_root = '/data/cHuang/HCC_proj/' # "/HCC_new_std1/HCC_proj"
    save_root = os.path.join(proj_root, 'data_cleaned_CECT_annotations/liverROI_3d_liverNoDilate_xyzMedian_no_zscore_tumorMaskByModel') # liverROI_3d_no_zscore
    save_path = os.path.join(save_root, 'unalligned') # for batch1&2
    liver_centroid_path = os.path.join(save_path, 'centroid_liver')
    tinies.sureDir(liver_centroid_path)
    liver_box_path = os.path.join(save_path, 'box_liver')
    tinies.sureDir(liver_box_path)

    pv_imgROI_dir = os.path.join(save_path, 'PV','imgROI')
    tinies.sureDir(pv_imgROI_dir) # crop liver Bbox on image
    art_imgROI_dir = os.path.join(save_path, 'ART', 'imgROI')
    tinies.sureDir(art_imgROI_dir)

    pv_liverMaskROI_dir = os.path.join(save_path, 'PV','liverMaskROI')
    tinies.sureDir(pv_liverMaskROI_dir) # crop liver ROI in liver mask
    art_liverMaskROI_dir = os.path.join(save_path, 'ART','liverMaskROI')
    tinies.sureDir(art_liverMaskROI_dir) # crop liver ROI in liver mask

    pv_tumorMaskROI_dir = os.path.join(save_path, 'PV','tumorMaskROI')
    tinies.sureDir(pv_tumorMaskROI_dir) # crop tumor ROI in tumor mask
    art_tumorMaskROI_dir = os.path.join(save_path, 'ART','tumorMaskROI')
    tinies.sureDir(art_tumorMaskROI_dir) # crop tumor ROI in tumor mask

    for bt in batch_tags:
        # bt = batch_tags[0]
        print('----- crop 3D ROI for batch {}  -------'.format(bt))
        imgs_dir = os.path.join(proj_root, 'data_cleaned_CECT_annotations/nnunet_formatted_data/nnunet_{}_img/'.format(bt))

        if case_choose is not None:
            caselist = [i.split('_A_')[0] for i in os.listdir(imgs_dir) if '_A_' in i and i.split("_A_")[0] in case_choose]
        else:
            caselist = [i.split('_A_')[0] for i in os.listdir(imgs_dir) if '_A_' in i]
        caselist.sort()

        # liver_mask_path = os.path.join(proj_root, 'data_cleaned_CECT_annotations/livermask/{}_byModel_postprocessed/'.format(bt))
        liver_mask_path = os.path.join(proj_root, 'data_cleaned_CECT_annotations/livermask/{}_final_liver/'.format(bt))
        tumor_mask_path = os.path.join(proj_root,'data_cleaned_CECT_annotations/cleaned_{}_tumorMaskByModel/'.format(bt))
        # tumor_mask_path = os.path.join(proj_root, 'data_cleaned_CECT_annotations/nnunet_formatted_data/nnunet_{}_all_tumor_mask_byModel/'.format(bt))
        #/ HCC_std1 / HCC_proj / data_cleaned_CECT_annotations / cleaned_b1to2_allTumorAnnotations / BA_003324578 ///
        # Running the Pool
        pool = Pool(NUM_OF_WORKERS)
        # caseName = caselist[0]
        results = pool.map(crop_liverBbox_3d, caselist)



    #-------- compute the liver boxes statistics -------
    for bt in batch_tags:
        # bt = batch_tags[0]
        print('\n----- compute liver boxes for batch {}  -------'.format(bt))
        imgs_dir = os.path.join(proj_root, 'data_cleaned_CECT_annotations/nnunet_formatted_data/nnunet_{}_img/'.format(bt))

        if case_choose is not None:
            caselist = [i.split('_A_')[0] for i in os.listdir(imgs_dir) if '_A_' in i and i.split("_A_")[0] in case_choose]
        else:
            caselist = [i.split('_A_')[0] for i in os.listdir(imgs_dir) if '_A_' in i]
        caselist.sort()
        liver_box_pv_x_list = []
        liver_box_pv_y_list = []
        liver_box_pv_z_list = []
        liver_box_art_x_list = []
        liver_box_art_y_list = []
        liver_box_art_z_list = []
        for caseName in caselist:
            # caseName = caselist[0]
            # caseName = 'BA_001024450'
            liver_box = np.loadtxt(os.path.join(liver_box_path, caseName+'.txt'))
            liver_box_pv_x_list.append(liver_box[0])
            liver_box_pv_y_list.append(liver_box[1])
            liver_box_pv_z_list.append(liver_box[2])
            liver_box_art_x_list.append(liver_box[3])
            liver_box_art_y_list.append(liver_box[4])
            liver_box_art_z_list.append(liver_box[5])
        # max liver shape
        liver_box_x_max = max(liver_box_pv_x_list + liver_box_art_x_list)
        liver_box_y_max = max(liver_box_pv_y_list + liver_box_art_y_list)
        liver_box_z_max = max(liver_box_pv_z_list + liver_box_art_z_list)

        target_shape_max = [liver_box_x_max, liver_box_y_max, liver_box_z_max] # x,y,z # 321-1, devisible to 32; 449-1, devisible to to 32. 

        print('for {}, the max shape={}'.format(bt, str(target_shape_max)))

        # median liver shape
        liver_box_x_median = np.median(liver_box_pv_x_list + liver_box_art_x_list)
        liver_box_y_median = np.median(liver_box_pv_y_list + liver_box_art_y_list)
        liver_box_z_median = np.median(liver_box_pv_z_list + liver_box_art_z_list)

        target_shape_median = [liver_box_x_median, liver_box_y_median, liver_box_z_median] # x,y,z # 321-1, devisible to 32; 449-1, devisible to to 32. 

        print('for {}, the median shape={}'.format(bt, str(target_shape_median)))

        # min liver shape
        liver_box_x_min = min(liver_box_pv_x_list + liver_box_art_x_list)
        liver_box_y_min = min(liver_box_pv_y_list + liver_box_art_y_list)
        liver_box_z_min = min(liver_box_pv_z_list + liver_box_art_z_list)

        target_shape_min = [liver_box_x_min, liver_box_y_min, liver_box_z_min] # x,y,z # 321-1, devisible to 32; 449-1, devisible to to 32. 

        print('for {}, the min shape={}'.format(bt, str(target_shape_min)))


        # ----- for batch b1to2  -------
        # for b1to2, the max shape=[410.0, 332.0, 45.0]
        # for b1to2, the median shape=[292.0, 239.0, 30.0]
        # for b1to2, the min shape=[213.0, 175.0, 18.0]
        # ----- for batch b3  -------
        # for b3, the max shape=[392.0, 346.0, 46.0]
        # for b3, the median shape=[293.0, 247.0, 31.0]
        # for b3, the min shape=[219.0, 156.0, 15.0]
        # ----- for batch b4  -------
        # for b4, the max shape=[450.0, 335.0, 45.0]
        # for b4, the median shape=[294.0, 245.0, 30.0]
        # for b4, the min shape=[194.0, 153.0, 15.0]

        
        # 20240110
        # ----- compute liver boxes for batch b1to2  -------
        # for b1to2, the max shape=[410.0, 332.0, 45.0]
        # for b1to2, the median shape=[292.0, 241.0, 30.0]
        # for b1to2, the min shape=[213.0, 175.0, 18.0]

        # ----- compute liver boxes for batch b3  -------
        # for b3, the max shape=[406.0, 346.0, 46.0]
        # for b3, the median shape=[293.0, 247.0, 31.0]
        # for b3, the min shape=[219.0, 156.0, 15.0]

        # ----- compute liver boxes for batch b4  -------
        # for b4, the max shape=[450.0, 335.0, 45.0]
        # for b4, the median shape=[292.0, 245.0, 30.0]
        # for b4, the min shape=[153.0, 153.0, 12.0]


        
        # ----- compute liver boxes for batch SRRH  -------
        # for SRRH, the max shape=[403.0, 327.0, 43.0]
        # for SRRH, the median shape=[292.5, 249.0, 30.0]
        # for SRRH, the min shape=[195.0, 184.0, 17.0]



    #-------- compute the tumor boxes statistics -------
    for bt in batch_tags:
        # bt = batch_tags[0]
        print('\n----- compute tumor boxes for batch {}  -------'.format(bt))
        imgs_dir = os.path.join(proj_root, 'data_cleaned_CECT_annotations/nnunet_formatted_data/nnunet_{}_img/'.format(bt))

        caselist = [i.split('_A_')[0] for i in os.listdir(imgs_dir) if '_A_' in i]
        caselist.sort()
        tumor_box_pv_x_list = []
        tumor_box_pv_y_list = []
        tumor_box_pv_z_list = []
        tumor_box_art_x_list = []
        tumor_box_art_y_list = []
        tumor_box_art_z_list = []
        pv_tumor_zero_cnt = 0
        art_tumor_zero_cnt = 0
        for caseName in caselist:
            # caseName = caselist[1]
            # caseName = 'BA_001024450'
            pv_tumorMaskROI_f = os.path.join(pv_tumorMaskROI_dir, '{}.nii.gz'.format(caseName))
            art_tumorMaskROI_f = os.path.join(art_tumorMaskROI_dir, '{}.nii.gz'.format(caseName))
            
            pv_tumorMaskROI,_ = load(pv_tumorMaskROI_f)
            art_tumorMaskROI,_ = load(art_tumorMaskROI_f)

            if len(np.unique(pv_tumorMaskROI))==2:
                pv_tumor_box_coords = find_box_coords(pv_tumorMaskROI)
                pv_tumor_box = [pv_tumor_box_coords[1]-pv_tumor_box_coords[0], pv_tumor_box_coords[3]-pv_tumor_box_coords[2], pv_tumor_box_coords[5]-pv_tumor_box_coords[4]]

                tumor_box_pv_x_list.append(pv_tumor_box[0])
                tumor_box_pv_y_list.append(pv_tumor_box[1])
                tumor_box_pv_z_list.append(pv_tumor_box[2])
            elif len(np.unique(pv_tumorMaskROI))==1:
                pv_tumor_box = [0, 0, 0]
                pv_tumor_zero_cnt += 1
            else:
                raise ValueError('{}: pv tumor has more than 2 unique values'.format(caseName))

            if len(np.unique(art_tumorMaskROI))==2:
                art_tumor_box_coords = find_box_coords(art_tumorMaskROI)
                art_tumor_box = [art_tumor_box_coords[1]-art_tumor_box_coords[0], art_tumor_box_coords[3]-art_tumor_box_coords[2], art_tumor_box_coords[5]-art_tumor_box_coords[4]]
                
                tumor_box_art_x_list.append(art_tumor_box[0])
                tumor_box_art_y_list.append(art_tumor_box[1])
                tumor_box_art_z_list.append(art_tumor_box[2])
            elif len(np.unique(art_tumorMaskROI))==1:
                art_tumor_box = [0, 0, 0]
                art_tumor_zero_cnt += 1
            else:
                raise ValueError('{}: art tumor has more than 2 unique values'.format(caseName))

        #
        print('for {}, {} pv mask contain no tumor, {} art mask contain no tumor'.format(bt, pv_tumor_zero_cnt, art_tumor_zero_cnt))

        # max tumor shape
        tumor_box_x_max = max(tumor_box_pv_x_list + tumor_box_art_x_list)
        tumor_box_y_max = max(tumor_box_pv_y_list + tumor_box_art_y_list)
        tumor_box_z_max = max(tumor_box_pv_z_list + tumor_box_art_z_list)

        target_shape_max = [tumor_box_x_max, tumor_box_y_max, tumor_box_z_max] # x,y,z # 321-1, devisible to 32; 449-1, devisible to to 32. 

        print('for {}, the max shape={}'.format(bt, str(target_shape_max)))

        # median tumor shape
        tumor_box_x_median = np.median(tumor_box_pv_x_list + tumor_box_art_x_list)
        tumor_box_y_median = np.median(tumor_box_pv_y_list + tumor_box_art_y_list)
        tumor_box_z_median = np.median(tumor_box_pv_z_list + tumor_box_art_z_list)

        target_shape_median = [tumor_box_x_median, tumor_box_y_median, tumor_box_z_median] # x,y,z # 321-1, devisible to 32; 449-1, devisible to to 32. 

        print('for {}, the median shape={}'.format(bt, str(target_shape_median)))

        # min tumor shape
        tumor_box_x_min = min(tumor_box_pv_x_list + tumor_box_art_x_list)
        tumor_box_y_min = min(tumor_box_pv_y_list + tumor_box_art_y_list)
        tumor_box_z_min = min(tumor_box_pv_z_list + tumor_box_art_z_list)

        target_shape_min = [tumor_box_x_min, tumor_box_y_min, tumor_box_z_min] # x,y,z # 321-1, devisible to 32; 449-1, devisible to to 32. 

        print('for {}, the min shape={}'.format(bt, str(target_shape_min)))

        # ----- compute tumor boxes for batch b1to2  -------
        # for b1to2, 9 pv mask contain no tumor, 12 art mask contain no tumor
        # for b1to2, the max shape=[347, 254, 33]
        # for b1to2, the median shape=[64.0, 64.0, 7.0]
        # for b1to2, the min shape=[12, 6, 0]

        # ----- compute tumor boxes for batch b3  -------
        # for b3, 9 pv mask contain no tumor, 10 art mask contain no tumor
        # for b3, the max shape=[368, 227, 37]
        # for b3, the median shape=[60.0, 61.0, 6.0]
        # for b3, the min shape=[12, 16, 0]

        # ----- compute tumor boxes for batch b4  -------
        # for b4, 31 pv mask contain no tumor, 33 art mask contain no tumor
        # for b4, the max shape=[387, 252, 38]
        # for b4, the median shape=[54.0, 55.0, 5.0]
        # for b4, the min shape=[3, 3, 0]

        # ----- compute tumor boxes for batch SRRH  -------
        # for SRRH, 22 pv mask contain no tumor, 29 art mask contain no tumor
        # for SRRH, the max shape=[343, 307, 41]
        # for SRRH, the median shape=[86.0, 88.0, 9.0]
        # for SRRH, the min shape=[3, 2, 0]
