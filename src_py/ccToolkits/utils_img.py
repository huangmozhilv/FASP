import os
import shutil
import math
import copy

from numpy.lib.arraysetops import unique
from scipy.ndimage import label
import tqdm
import SimpleITK as sitk
import numpy as np
import cv2
from scipy import ndimage


def crop_central(img, target_size=[512, 512, None]):
    # some patients have irregular image size, e.g. 201569362, the image size is [512, 666, 102]. 666 is not divisible to 8, therefore will result in error in U-Net like models. therefore, here crop the image to 512. Some patients might have image size < 512, will pad with 0 to size of 512.
    x, y, z = img.shape
    target_x, target_y, target_z = target_size

    if target_x is None:
        target_x = x

    if target_y is None:
        target_y = y

    if target_z is None:
        target_z = z

    #
    max_x = max(target_x, x)
    max_y = max(target_y, y)
    max_z = max(target_z, z)

    # get the largest candidate area. if outside the original image, pad it.
    tmp_img = np.zeros([max_x, max_y, max_z], dtype=int)
    tmp_x, tmp_y, tmp_z = tmp_img.shape
    centroid = [int(tmp_x/2), int(tmp_y/2), int(tmp_z/2)] # this is not always the real centroid. if x/y/z is odd, it will be a little smaller than the real one. Here no need to have the real one.

    bbox_raw_min = np.asarray([centroid[0]-(x/2), centroid[1]-(y/2), centroid[2]-(z/2)], dtype=int)
    # bbox_raw_max = np.asarray([centroid[0]+(x/2)-1, centroid[1]+(y/2)-1, centroid[2]+(z/2)-1], dtype=int)
    bbox_raw_max = np.asarray([bbox_raw_min[0] + x -1, bbox_raw_min[1] + y-1, bbox_raw_min[2] + z-1], dtype=int)

    tmp_img[np.ix_(
        range(bbox_raw_min[0], bbox_raw_max[0]+1),
        range(bbox_raw_min[1], bbox_raw_max[1]+1),
        range(bbox_raw_min[2], bbox_raw_max[2]+1)
    )] = img


    # extract the target image area
    bbox_new_min = np.asarray([centroid[0]-(target_x/2), centroid[1]-(target_y/2), centroid[2]-(target_z/2)], dtype=int)
    bbox_new_max = np.asarray([bbox_new_min[0] + target_x -1, bbox_new_min[1] + target_y -1, bbox_new_min[2] + target_z - 1], dtype=int)

    img_new = tmp_img[np.ix_(
        range(bbox_new_min[0], bbox_new_max[0]+1),
        range(bbox_new_min[1], bbox_new_max[1]+1),
        range(bbox_new_min[2], bbox_new_max[2]+1)
    )]

    return img_new


def window_image(img, window_level, window_width):
    img_min = window_level - window_width/2
    img_max = window_level + window_width/2

    img[img<img_min] = img_min
    img[img>img_max] = img_max

    return img

def norm2range(img, tgt_range = [0,255], mask=None):
    # normalize the image to tgt_range
    ymin = min(tgt_range)
    ymax = max(tgt_range)

    if mask is None:
        xmax = np.max(img)
        xmin = np.min(img)
    else:
        image_masked=img[mask!=0] # this will output a ndarray with size of (n,), where n is the number of pixels where mask!=0
        xmax = image_masked.max()
        xmin = image_masked.min()
    norm_img = ((img-xmin)*(ymax-ymin)/(xmax-xmin))+ymin #-->[0,255]
    if mask is None:
        pass
    else:
        norm_img *= mask
    return norm_img

def norm_Zscore(image, mask=None):
    if mask is None:
        image_masked = copy.deepcopy(image)
    else:
        image_masked=image[mask!=0] # this will output a ndarray with size of (n,), where n is the number of pixels where mask!=0
    # mask_slice = np.zeros(image.shape)
    # mask_slice[np.where(np.sum(image, axis=(len(image.shape)-2, len(image.shape)-1)))] = 1
    min_pixel = image_masked.min()
    max_pixel = image_masked.max()
    # image,min_c,max_c=clip_img(image,min_pixel,max_pixel,0.1)
    norm_img = (image -min_pixel) / (max_pixel - min_pixel)
    norm_img = norm_img * mask
    return norm_img

def itensity_normalize_one_volume(volume):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzero region
    inputs:
        volume: the input nd volume
    outputs:
        out: the normalized nd volume
    """
    # pixels = volume[volume > 0] # wrong? by Chao.
    pixels = volume[volume != 0]
    mean = pixels.mean()
    std  = pixels.std()
    out = (volume - mean)/(std + 1e-20)
    # random normal too slow
    #out_random = np.random.normal(0, 1, size = volume.shape)
    out_random = np.zeros(volume.shape)
    out[volume == 0] = out_random[volume == 0]

    return out

def vol_intensity_normTo0to1(volume, mask=None, tgt_range=None):
    """
    normalize the itensity of an nd volume based on the mean and std of nonzero region
    inputs:
        volume: the input nd volume
        mask: choices:(1) None;(2)'non0region';(3)np.ndarray, binary. 0/1
        tgt_range: choices. (1) None; (2) a list of two elements, i.e. lower and upper limits of HU, e.g [-70,190]
    outputs:
        out: the normalized nd volume
    """
    if mask is None:
        pixels = volume
    elif type(mask) is np.ndarray:
        mask = mask != 0
        pixels = volume*mask
    elif type(mask) is str and mask=='non0region':
        mask = volume != 0
        pixels = volume*mask
    else:
        raise ValueError('this mask is not supported')

    if tgt_range is None:
        tgt_max = pixels.max()
        tgt_min  = pixels.min()
    else:
        tgt_min = min(tgt_range)
        tgt_max = max(tgt_range)
        volume = np.clip(volume, tgt_min, tgt_max)

    out = (volume - tgt_min)/(tgt_max-tgt_min)

    if mask is not None:
        out = out * mask

    return out

def interpolate_spacing(spacing, refer_spacing):
    # aim: for any element that is none in spacing, replace it with the counterpart in refer_spacing
    # spacing and refer_spacing have same order, e.g. x,y,z
    newSpacing = []
    for i in range(len(spacing)):
        if spacing[i] is not None:
            s_i = round(spacing[i], 3)
        else:
            s_i = round(refer_spacing[i], 3)
        newSpacing.append(s_i)
    newSpacing = np.asarray(newSpacing, dtype=float)

    return newSpacing
    

def resample2fixedSpacing(volume, oldSpacing, newSpacing, refer_file_path, interpolate_method=sitk.sitkBSpline): 
    # sitk.sitkLinear
    '''
    aim: to resample raw image to unified spacing before training

    also works for 2-D?
    Resample dat(i.e. one 3-D sitk image/GT) to destine resolution. but keep the origin, direction be the same.
    Volume: 3D numpy array, z, y, x.
    oldSpacing: z,y,x
    newSpacing: z,y,x
    refer_file_path: source to get origin, direction, and oldSpacing. Here we use the image_file path.
    ''' 
    # in the project, oldSpacing, origin, direction will be extracted from gt_file as the refer_file_path.     oldSpacing: x,y,z
    sitk_refer = sitk.ReadImage(refer_file_path)
    # extract first modality as sitk_refer if there are multiple modalities
    if sitk_refer.GetDimension() == 4:
        sitk_refer = sitk.Extract(sitk_refer, (sitk_refer.GetSize()[0], sitk_refer.GetSize()[1], sitk_refer.GetSize()[2], 0), (0,0,0,0))
    origin = sitk_refer.GetOrigin()
    # oldSpacing =  sitk_refer.GetSpacing()
    direction = sitk_refer.GetDirection()

    # prepare oldSize, oldSpacing, newSpacing, newSize in order of [x,y,z]
    oldSize = np.asarray(volume.shape, dtype=float)[::-1]
    oldSpacing = np.asarray([round(i, 3) for i in oldSpacing], dtype=float)
    oldSpacing = oldSpacing[::-1]

    newSpacing = np.asarray([round(i, 3) for i in newSpacing], dtype=float)
    newSpacing = newSpacing[::-1]

    # compute new size, assuming same volume of tissue (not number of total pixels) before and after resampled 
    newSize = np.asarray(oldSize * oldSpacing/newSpacing, dtype=int)

    # create sitk_old from array and set appropriate meta-data
    sitk_old = sitk.GetImageFromArray(volume)

    sitk_old.SetOrigin(origin)
    sitk_old.SetSpacing(oldSpacing)
    sitk_old.SetDirection(direction)
    sitk_new = sitk.Resample(sitk_old, newSize.tolist(), sitk.Transform(), interpolate_method, origin, newSpacing, direction)

    newVolume = sitk.GetArrayFromImage(sitk_new)

    return newVolume

def resample2fixedSize(volume, oldSpacing, newSize, refer_file_path, interpolate_method=sitk.sitkNearestNeighbor):
    '''
    aim: to resample predicted segmentation to the original size of raw image.

    also works for 2-D?
    Goal---resample to fixed size with new spacing, but keep the origin, direction be the same.
    volume: 3-D numpy array, z, y, x. In this code package, this is the final predicted label map, its shape is that of cropped non-zero region resampled to fixed spacings. 
    oldSpacing: z,y,x. the spacing of the volume. In this project, it's the isotropical spacing.
    newSize: z,y,x. in this project, its shape is that of the cropped non-zero region.
    refer_file_path: source to get origin, direction. Here we use the image_file path.
    ''' 
    # in the project, newSpacing, origin, direction will be extracted from gt_file as the refer_file_path
    sitk_refer = sitk.ReadImage(refer_file_path)
    # extract first modality as sitk_refer if there are multiple modalities
    if sitk_refer.GetDimension() == 4:
        sitk_refer = sitk.Extract(sitk_refer, (sitk_refer.GetSize()[0], sitk_refer.GetSize()[1], sitk_refer.GetSize()[2], 0), (0,0,0,0))
    origin = sitk_refer.GetOrigin()
    # newSpacing =  sitk_refer.GetSpacing()
    direction = sitk_refer.GetDirection()

    # prepare oldSize, oldSpacing, newSpacing, newSize in order of [x,y,z]
    oldSpacing = np.asarray(oldSpacing, dtype=float)[::-1]
    oldSize = volume.shape[::-1]

    newSize = np.asarray(newSize, dtype=int)[::-1]
    newSpacing = np.asarray(oldSize * oldSpacing/newSize, dtype=float)
    # compute new size, assuming same volume of tissue (not number of total pixels) before and after resampled 


    # create sitk_old from array and set appropriate meta-data
    sitk_old = sitk.GetImageFromArray(volume)

    sitk_old.SetOrigin(origin)
    sitk_old.SetSpacing(oldSpacing)
    sitk_old.SetDirection(direction)
    sitk_new = sitk.Resample(sitk_old, newSize.tolist(), sitk.Transform(), interpolate_method, origin, newSpacing, direction)

    newVolume = sitk.GetArrayFromImage(sitk_new)

    return newVolume

def fill_holes(src_vol, struct, iters):
    # to fill holes
    # first erode to remove very small predicted lesions to reduce false positives.
    # then dilate to recover overal.
    src_vol = ndimage.binary_erosion(src_vol, structure=struct, iterations=iters).astype(src_vol.dtype)

    src_vol = ndimage.binary_dilation(src_vol, structure=struct, iterations=iters).astype(src_vol.dtype)

    src_vol = ndimage.binary_erosion(src_vol, structure=struct, iterations=iters).astype(src_vol.dtype)

    src_vol = ndimage.binary_dilation(src_vol, structure=struct, iterations=iters).astype(src_vol.dtype)

def pad_to_shape(arr, new_shape):
    # arr: 3D or 2D
    old_shape = arr.shape
    pad_all_list = [new_shape[i]-old_shape[i] for i in range(len(new_shape))]
    pad_ends_list = [(pad//2, pad//2+pad%2) for pad in pad_all_list]
    # print(pad_ends_list)
    out = np.pad(arr,tuple(pad_ends_list),
                  mode = 'constant')
    return out

# copied from nnunet/connected_component
# in fact, this function can only be used to remove the components smaller than a minimum_valid_object_size.
def remove_all_but_the_largest_connected_component(image: np.ndarray, for_which_classes: list, volume_per_voxel: float,
                                                   minimum_valid_object_size: dict = None):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :param minimum_valid_object_size: Only objects larger than minimum_valid_object_size will be removed. Keys in
    minimum_valid_object_size must match entries in for_which_classes
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    largest_removed = {}
    kept_size = {}
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel

        largest_removed[c] = None
        kept_size[c] = None

        if num_objects > 0:
            # we always keep the largest object. We could also consider removing the largest object if it is smaller
            # than minimum_valid_object_size in the future but we don't do that now.
            maximum_size = max(object_sizes.values())
            kept_size[c] = maximum_size

            for object_id in range(1, num_objects + 1):
                # we only remove objects that are not the largest
                if object_sizes[object_id] != maximum_size:
                    # we only remove objects that are smaller than minimum_valid_object_size
                    remove = True
                    if minimum_valid_object_size is not None:
                        remove = object_sizes[object_id] < minimum_valid_object_size[c]
                    if remove:
                        image[(lmap == object_id) & mask] = 0
                        if largest_removed[c] is None:
                            largest_removed[c] = object_sizes[object_id]
                        else:
                            largest_removed[c] = max(largest_removed[c], object_sizes[object_id])
    return image, largest_removed, kept_size

def find_connected_components(image: np.ndarray, for_which_classes: list, volume_per_voxel: float):
    """
    removes all but the largest connected component, individually for each class
    :param image:
    :param for_which_classes: can be None. Should be list of int. Can also be something like [(1, 2), 2, 4].
    Here (1, 2) will be treated as a joint region, not individual classes (example LiTS here we can use (1, 2)
    to use all foreground classes together)
    :return:
    """
    if for_which_classes is None:
        for_which_classes = np.unique(image)
        for_which_classes = for_which_classes[for_which_classes > 0]

    assert 0 not in for_which_classes, "cannot remove background"
    for c in for_which_classes:
        if isinstance(c, (list, tuple)):
            c = tuple(c)  # otherwise it cant be used as key in the dict
            mask = np.zeros_like(image, dtype=bool)
            for cl in c:
                mask[image == cl] = True
        else:
            mask = image == c
        # get labelmap and number of objects
        lmap, num_objects = label(mask.astype(int))

        # collect object sizes
        object_sizes = {}
        object_voxel_num_list = {}
        for object_id in range(1, num_objects + 1):
            object_sizes[object_id] = (lmap == object_id).sum() * volume_per_voxel
            object_voxel_num_list[object_id] = (lmap == object_id).sum()
            

    return object_sizes, object_voxel_num_list