
import os
import sys
sys.path.append(os.getcwd())

from asyncio.log import logger
from builtins import NotImplementedError
from importlib.abc import PathEntryFinder
from time import time
import copy
import json
from unittest.mock import patch
import  scipy
import numpy as np
import pandas as pd
import skimage
from skimage.transform import resize

# from batchgenerators.augmentations.crop_and_pad_augmentations import crop
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import Compose
from batchgenerators.utilities.file_and_folder_operations import *
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase # DataLoader
# from batchgenerators.augmentations.utils import pad_nd_image
from batchgenerators.transforms.spatial_transforms import SpatialTransform_2, MirrorTransform
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianNoiseTransform, GaussianBlurTransform

from survPred import config
from ccToolkits import utils_img
from ccToolkits import plotFuncs

def random_scale(arr, allSeg, target_shape, fluctuation_limits=[-0.2, 0.2]):
    # arr: [phases, z,y,x]?
    # fluctuation_limits should be less than 1
    # fluctuation_limits = [-0.2, 0.2]
    old_shape = arr.shape[1:]
    current_min_fluct = min([target_shape[i]/old_shape[i] for i in range(3)])-1
    ratio = np.random.uniform(fluctuation_limits[0],min(fluctuation_limits[1],current_min_fluct),1)[0]
    new_shape = [int(old_shape[i]*(1+ratio)) for i in range(3)]
    # print(new_shape)
    arr_out = np.zeros([arr.shape[0]]+new_shape, dtype=np.float32)
    allSeg_out = np.zeros([arr.shape[0]]+new_shape, dtype=np.uint8)

    for i in range(arr.shape[0]):
        arr_out[i] = resize(arr[i].astype(float),new_shape,order=3, mode="edge", anti_aliasing=False).astype(arr_out.dtype)
        allSeg_out[i] = resize(allSeg[i].astype(float), new_shape, order=0,anti_aliasing=False).astype(allSeg_out.dtype) # 语义分割标签图像的数值发生了改变，数据类型也发生了改变，最关键的是数值也发生了改变。仔细查阅官方文档，添加anti_aliasing=False选项即可，因为默认是进行高斯滤波的；

        # The order of interpolation. The order has to be in the range 0-5:
        # 0: Nearest-neighbor
        # 1: Bi-linear (default)
        # 2: Bi-quadratic
        # 3: Bi-cubic
        # 4: Bi-quartic
        # 5: Bi-quintic
        # 使用 skimage 支持的数据类型创建一个数组，调整其大小，并（如有必要）将其恢复为原始数据类型。


        # Default is 0 if image.dtype is bool and 1 otherwise.

    return arr_out,allSeg_out,ratio

def random_pad_to_shape(arr, allSeg, new_shape):
    # arr: c, z, y, x
    old_shape = arr.shape[1:]
    pad_num_list = [new_shape[i]-old_shape[i] for i in range(len(new_shape))]
    # print('pad_num_list:{}'.format(str(pad_num_list)))
    # pad_ends_list = [(pad//2, pad//2+pad%2) for pad in pad_all_list]
    pad_starts_list = [np.random.choice(list(range(1,pad+1)), 1)[0] if pad!=0 else 0 for pad in pad_num_list]
    # print('pad_num_list:{}'.format(str(pad_num_list)))
    pads_list = [(pad_starts_list[i], pad_num_list[i]-pad_starts_list[i]) for i in range(len(pad_num_list))]
    # print(pad_ends_list)
    arr_out = np.zeros([arr.shape[0]] + new_shape,dtype=np.float32)
    allSeg_out = np.zeros([arr.shape[0]] + new_shape, dtype=np.uint8)

    for i in range(arr_out.shape[0]):
        arr_out[i] = np.pad(arr[i],tuple(pads_list), mode = 'constant')
        allSeg_out[i] = np.pad(allSeg[i],tuple(pads_list), mode = 'constant')

    return arr_out,allSeg_out

def even_pad_to_shape(arr,allSeg, new_shape):
    # arr: c, z, y, x
    arr_out = np.zeros([arr.shape[0]] + new_shape,dtype=np.float32)
    allSeg_out = np.zeros([arr.shape[0]] + new_shape, dtype=np.uint8)

    for i in range(arr_out.shape[0]): # two modalities
        arr_out[i] = utils_img.pad_to_shape(arr[i], new_shape)
        allSeg_out[i] = utils_img.pad_to_shape(allSeg[i], new_shape)

    return arr_out, allSeg_out



###################
class survDataLoader(SlimDataLoaderBase):
    def __init__(self, task_names, data_root_dict, imgs_dir, pat_ids, batch_size, patch_size,mode='train',clin=False,
                 return_incomplete=False, shuffle=True, infinite=True, if_prep_tumorMask=False, num_threads_in_multithreaded=1, seed_for_shuffle=1234,
                 surv_endpoint_fname=None,if_draw_gt=False):
        """
        patch_size is the spatial size the retured batch will have
        mode = 'train','infer'
        """
        super().__init__(pat_ids, batch_size, num_threads_in_multithreaded)
        self.mode = mode
        self.clin = clin
        self.if_prep_tumorMask = if_prep_tumorMask

        self.num_restarted = 0
        self.current_position = 0
        self.was_initialized = False

        self.seed_for_shuffle = seed_for_shuffle
        self.return_incomplete = return_incomplete
        self.shuffle = shuffle
        self.infinite = infinite

        self.task_names = task_names
        self.data_root_dict = data_root_dict
        self.patch_size = patch_size
        self.num_modalities = 2 # pv & art

        self.imgs_dir=imgs_dir


        # added by Chao
        self.pat_ids = pat_ids
        if not if_draw_gt:
            self.art_data_path = os.path.join(self.imgs_dir, 'ART', 'imgROI/')# data_path.replace('PV','ART')
            self.pv_data_path = os.path.join(self.imgs_dir, 'PV', 'imgROI/')
        else:
            self.art_data_path = os.path.join(self.imgs_dir, 'ART', 'imgROI_GT/')  # data_path.replace('PV','ART')
            self.pv_data_path = os.path.join(self.imgs_dir, 'PV', 'imgROI_GT/')

        # liver
        if not if_draw_gt:
            self.art_liverMask_path = os.path.join(self.imgs_dir, 'ART', 'liverMaskROI/')
            self.pv_liverMask_path = os.path.join(self.imgs_dir, 'PV', 'liverMaskROI/')
        else:
            self.art_liverMask_path = os.path.join(self.imgs_dir, 'ART', 'liverMaskROI_GT/')
            self.pv_liverMask_path = os.path.join(self.imgs_dir, 'PV', 'liverMaskROI_GT/')

        if self.if_prep_tumorMask and not if_draw_gt:
            # tumor
            self.art_tumorMask_path = os.path.join(self.imgs_dir, 'ART', 'tumorMaskROI/')
            self.pv_tumorMask_path = os.path.join(self.imgs_dir, 'PV', 'tumorMaskROI/')
        elif if_draw_gt:
            self.art_tumorMask_path = os.path.join(self.imgs_dir, 'ART', 'tumorMaskROI_GT/')
            self.pv_tumorMask_path = os.path.join(self.imgs_dir, 'PV', 'tumorMaskROI_GT/')
        else:
            pass

        #
        surv_dir = os.path.join(data_root_dict['nonImg'],'survival') #survival_bkp_BF20240406
        
        if surv_endpoint_fname is None:
            surv_endpoint_f = os.path.join(surv_dir, 'surv_endpoint.csv')
        else:
            surv_endpoint_f = os.path.join(surv_dir, surv_endpoint_fname)

        # get all image filenames
        self.art_filenames, self.pv_filenames= self.get_file_names(pat_ids)
        # self.indices = list(range(len(pat_ids)))
        sample_ids = [i.strip('.npy') for i in self.pv_filenames if i.endswith('.npy')]
        self.sample_ids = sorted(sample_ids) 
        self.indices = list(range(len(self.sample_ids)))
        
        surv_endpoint_df = pd.read_csv(surv_endpoint_f)

        self.surv_dict = dict()
        self.task_indices = dict()
        for outcome in self.task_names: # ['recur','death']:
            self.surv_dict[outcome] = dict()
            for case in pat_ids:
                # case = cases[0]
                self.surv_dict[outcome][case] = dict()
                case_ = case
                # if "TCGA" in case:
                #     case_ = case # f"TCGA-{case[-6:-4]}-{case[-4:]}"
                # else:
                #     case_ = case
                try:
                    self.surv_dict[outcome][case]['status'] =surv_endpoint_df['{}_status'.format(outcome)][surv_endpoint_df['pat_id']==case_].values[0]  
                    self.surv_dict[outcome][case]['time'] = surv_endpoint_df['{}_time'.format(outcome)][surv_endpoint_df['pat_id']==case_].values[0] 
                except:
                    print('cannot load surv data for {}'.format(case_))
            self.task_indices[outcome] = [i for i in self.indices if self.sample_ids[i] in surv_endpoint_df['pat_id'][surv_endpoint_df['{}_status'.format(outcome)]==1].values.tolist()]
        

        # clin features
        if surv_endpoint_fname is None:
            clin_f = os.path.join(surv_dir, 'surv_clinical_features.csv')
        else:
            testSite = surv_endpoint_fname.split('_surv_')[0]
            clin_f = os.path.join(surv_dir, '{}_surv_clinical_features.csv'.format(testSite))
        try:
            clin_df = pd.read_csv(clin_f)

            self.clin_dict = dict()

            for case in pat_ids:
                if config.clin_feats is None:
                    self.clin_dict[case] = None
                else:
                    self.clin_dict[case] = clin_df[config.clin_feats][clin_df['pat_id']==case].values.tolist()[0] # out is a list
        except:
            raise ValueError('error for extracting data from: {}'.format(str(clin_f)))


    def reset(self):
        # Prevents the random order for each epoch being the same
        if self.shuffle:
            rs = np.random.RandomState(self.num_restarted)

            # Here the data is shuffled but one can easily replace this with a
            # shuffle of indices for when one wants to load the data while generating
            # a batch in real-time, for example.
            #
            # Eg. rs.shuffle(self._data_indices)
            # rs.shuffle(self._data)
            rs.shuffle(self.indices)
        else:
            pass
        self.was_initialized = True
        self.num_restarted = self.num_restarted + 1

        # Select a starting point for this subprocess. The self.thread_id is set by
        # MultithreadedAugmentor and is in the range [0, num_of_threads_in_mt)
        # Multiplying it with batch_size gives every subprocess a unique starting
        # point WHILE taking into consideration the size of the batch
        self.current_position = self.thread_id*self.batch_size

    # added by Chao
    def get_file_names(self, pat_ids):
        # customized for this surv task
        pv_files = [i for i in os.listdir(self.pv_data_path) if i.endswith('.npy') and i.strip('.npy') in pat_ids]
        pv_files.sort()
    
        art_files = [i for i in pv_files]
        art_files.sort()
    
        print('\nNumber of dataloader samples: ', len(pv_files))
        return art_files, pv_files


    @staticmethod
    def load_sample(self, sample_id):
        # sample_id = 'BA_000847313'
        # sample_id = 'BA_003441373'
        # sample_id = 'BA_002245621'
        # zoom_size=[16,64,64]
        filename = '{}.npy'.format(sample_id)

        image_art = np.load(os.path.join(self.art_data_path, filename)).astype(np.float32)
        image_pv = np.load(os.path.join(self.pv_data_path, filename)).astype(np.float32) # z,y,x?

        if "tumorROI" in self.imgs_dir:
            image_art = scipy.ndimage.zoom(image_art,  np.asarray([image_art.shape[0], 64, 64])/np.asarray(image_art.shape))
            image_pv = scipy.ndimage.zoom(image_pv, np.asarray([image_pv.shape[0], 64, 64])/np.asarray(image_pv.shape)) #[image_pv.shape[0], 64, 64]

        data = np.concatenate([np.expand_dims(image_art, 0), np.expand_dims(image_pv, 0)], axis=0) # 2,z,y,x

        # liver
        liverMask_art = np.load(os.path.join(self.art_liverMask_path, filename)).astype(np.uint8)
        liverMask_pv = np.load(os.path.join(self.pv_liverMask_path, filename)).astype(np.uint8) # z,y,x?

        if "tumorROI" in self.imgs_dir:
            liverMask_art = scipy.ndimage.zoom(liverMask_art,  np.asarray([liverMask_art.shape[0], 64, 64])/np.asarray(liverMask_art.shape))
            liverMask_pv = scipy.ndimage.zoom(liverMask_pv, np.asarray([liverMask_pv.shape[0], 64, 64])/np.asarray(liverMask_pv.shape))

        liverMask_all = np.concatenate([np.expand_dims(liverMask_art, 0), np.expand_dims(liverMask_pv, 0)], axis=0) # 2,z,y,x

        # finalMask: contain all objects segmentations in one mask
        finalMask_all = copy.deepcopy(liverMask_all)

        if self.if_prep_tumorMask:
            # tumor
            tumorMask_art = np.load(os.path.join(self.art_tumorMask_path, filename)).astype(np.uint8)
            tumorMask_pv = np.load(os.path.join(self.pv_tumorMask_path, filename)).astype(np.uint8) # z,y,x?

            if "tumorROI" in self.imgs_dir:
                tumorMask_art = scipy.ndimage.zoom(tumorMask_art,np.asarray([tumorMask_art.shape[0], 64, 64]) / np.asarray(tumorMask_art.shape))
                tumorMask_pv= scipy.ndimage.zoom(tumorMask_pv,np.asarray([tumorMask_pv.shape[0], 64, 64]) / np.asarray(tumorMask_pv.shape))

            tumorMask_all = np.concatenate([np.expand_dims(tumorMask_art, 0), np.expand_dims(tumorMask_pv, 0)], axis=0) # 2,z,y,x
            
            # integrate tumor into liver
            finalMask_all[tumorMask_all==1] = 2
        else:
            tumorMask_all = None


        #
        target_shape = self.patch_size

        if self.mode=='train':
            # random scaling: the output shape should not be larger than the target_shape
            if np.random.uniform(0,1)<0.2: # debug
                data, finalMask_all,ratio = random_scale(data, finalMask_all, target_shape, fluctuation_limits=[-0.2,0.2])
            # random padding
            data, finalMask_all = random_pad_to_shape(data, finalMask_all, target_shape)
        else:
            data, finalMask_all = even_pad_to_shape(data, finalMask_all, target_shape)

        # if len(np.unique(data))==1:
        #     raise ValueError('data is all 0 after pad')

        # load label
        task_survs = dict()
        for task in ['recur','death']:
            if task in self.task_names:
                task_survs[task] = self.surv_dict[task][sample_id]
            else:
                task_survs[task] = {
                    'status': None,
                    'time':None
                }

        # data = data*liverMask_all.astype(data.dtype) #已换成在vol_intensity_normTo0to1()中实现liverMask_all内norm to 0~1, liverMask_all外都是0

        out = dict()
        out['data'] = data
        out ['finalMask_all'] = finalMask_all.astype(np.uint8)
        out ['recur_surv'] = task_survs['recur']
        out ['death_surv'] = task_survs['death']

        if self.clin:
            out ['clin_feats'] = self.clin_dict[sample_id]
        else:
            pass

        return out



    def get_indices_cc(self):
        # adapted from get_indices() by Chao.
        # import ipdb; ipdb.set_trace()

        if self.infinite:
            self.num_restarted += 1 #手动改变，不然infinite的情况没有reset这一步意味着self.num_restarted一直不变。
            rs = np.random.RandomState(self.num_restarted)
            # Here the data is shuffled but one can easily replace this with a 
            # shuffle of indices for when one wants to load the data while generating
            # a batch in real-time, for example.
            #
            # Eg. rs.shuffle(self._data_indices) 
            # rs.shuffle(self._data)
            indices_1 = []

            for outcome in self.task_names:
                rs.shuffle(self.task_indices[outcome])
                indices_1.extend(list(np.random.choice(self.task_indices[outcome], 1, replace=False))) # replace=放回

            indices_1 = list(set(indices_1))

            indices_left = copy.deepcopy(self.indices)
            for i in indices_1:
                indices_left.remove(i)

            indices_2 = list(np.random.choice(indices_left, self.batch_size-len(indices_1), replace=False))

            indices = indices_1 + indices_2
            rs.shuffle(indices)
            return indices
        else:
            # if self.last_reached:
            #     print('last reached')
            #     self.reset()
            #     raise StopIteration

            if not self.was_initialized:
                self.reset()

            idx = self.current_position

            if idx < len(self._data):
                # Next starting point. This skips the length of one batch for
                # this process AS WELL AS all the other processes (i.e, self.number_of_threads_in_multithreaded)
                # Since the processes already have unique (but contiguous) starting 
                # points due to the initialization of self.current_position in 
                # reset(), they continue to not overlap.
                self.current_position = idx + self.batch_size*self.number_of_threads_in_multithreaded

                # Having assured that the next starting point is safe, we simply
                # return the next batch. Additionally, we take into consideration
                # that the idx+batch_size might exceed the dataset size so we take 
                # min(len(self._data),idx+self.batch_size) as the end index
                indices = list(range(idx, min(len(self._data),idx+self.batch_size)))

                # to ensure at least one event for each task in indices, required to compute cox loss
                if self.mode == 'train':
                     # to ensure at least one event for each task in indices, required to compute cox loss. by Chao
                    if_contain_event_list = []
                    for task in self.task_names:
                        if_contain_event = len([i for i in indices if i in self.task_indices[task]])>0
                        if_contain_event_list.append(if_contain_event)
                    if all(if_contain_event_list):
                        pass
                    else:
                        indices = [indices[0]] # if set None, will cause transform to be failed
                else:
                    pass

                return indices
            else:
                self.was_initialized=False
                raise StopIteration


    def generate_train_batch(self):
        # import ipdb; ipdb.set_trace()
        # DataLoader has its own methods for selecting what patients to use next, see its Documentation
        # separate recur and death sampling as it is rather difficult to obtain a batch containing both negative and positive cases in respect to both recur and death since the positive cases are rare in the datasets. however, this is required to compute cox loss. as a result Chao adopt the round-robin method to update the model by updating the model with the loss of recur and loss of death in turn.
        # idx = self.get_indices() # get_indices() is the func of Dataloader(). # can be substitued by customed func.
        # from ccToolkits import tinies
        # tinies.ForkedPdb().set_trace()

        indices = self.get_indices_cc()
        samples_for_batch = [self.sample_ids[i] for i in indices]

        # initialize empty array
        data = np.zeros((len(indices), self.num_modalities, *self.patch_size), dtype=np.float32)
        seg = np.zeros((len(indices), self.num_modalities, *self.patch_size), dtype=np.uint8)


        lab = [None] * len(indices)
        if self.clin:
            # clin_data = np.zeros((self.batch_size,len(config.clin_feats)), dtype=np.float32)
            clin_data = np.zeros((len(indices),len(config.clin_feats)), dtype=np.float32)
        else:
            pass
        metadata = []
        sample_ids = []

        # iterate over samples_for_batch and include them in the batch
        recur_status_list = list()
        death_status_list = list()
        for i, j in enumerate(samples_for_batch):
            try:
                sample_loaded = self.load_sample(self, j)
                patient_data = sample_loaded['data']
                patient_seg = sample_loaded['finalMask_all']

                seg[i]=patient_seg

                patient_recur_surv = sample_loaded['recur_surv']
                patient_death_surv = sample_loaded['death_surv']
                if self.clin:
                    clin_data[i]= sample_loaded['clin_feats']
                else:
                    pass

                # # for debug
                # import SimpleITK as sitk
                # for k in range(2):
                #     sitk_img = sitk.GetImageFromArray(patient_data[0,:])
                #     sitk.WriteImage(sitk_img, '/data/cHuang/HCC_proj/results/{}_{}.nii.gz'.format(j,k))

                # data[i] = utils_img.norm2range(patient_data,tgt_range=[0,1]) # vol_intensity_normTo0to1() was used in data_prep
                
                # crop imgROI based on seg
                data[i] = utils_img.vol_intensity_normTo0to1(patient_data, patient_seg, tgt_range=config.liver_HU_range)

                # data[i] = utils_img.vol_intensity_normTo0to1_with_specifiedMax(patient_data,config.liver_HU_range[1])

                # out_dir = '/data/cHuang/HCC_proj/results/debug'
                # plotFuncs.select_plot_imgs(patient_data[0,:], False, 5, ncols=3, out_f=os.path.join(out_dir, '{}_slices_pv_img_after_norm.png'.format(j)))
                # plotFuncs.select_plot_imgs(patient_data[1,:], False, 5, ncols=3, out_f=os.path.join(out_dir, '{}_slices_art_img_after_norm.png'.format(j)))
                lab[i] = {
                    'recur_surv': patient_recur_surv,
                    'death_surv': patient_death_surv
                }
                recur_status_list.append(patient_recur_surv['status'])
                death_status_list.append(patient_death_surv['status'])
                # metadata.append(patient_metadata)
                sample_ids.append(j)
                # print('loaded sample for {}'.format(j))
            except ValueError:
                print('load sample error with sample_id: {}'.format(j))
                raise ValueError
        assert np.all(np.isfinite(data)), 'some patients have NA or Inf in the image:{}'.format(str(sample_ids))

        # the keys('data' and 'seg') should never be used to denote other objects as they will be used in transform by batchgenerators.
        out = {'data': data,'surv':lab, 'metadata':metadata, 'names':sample_ids,"seg":seg.astype(np.uint8)}


        if self.clin:
            out['clin_data'] = clin_data

        return out


def get_train_transform(patch_size):
    # 部分data augmentation的比例参考/data/cHuang/HCC_proj/src_py/nnUNet/nnunet/training/data_augmentation/default_data_augmentation.py
    # we now create a list of transforms. These are not necessarily the best transforms to use for BraTS, this is just
    # to showcase some things
    tr_transforms = []

    # the first thing we want to run is the SpatialTransform. It reduces the size of our data to patch_size and thus
    # also reduces the computational cost of all subsequent operations. All subsequent operations do not modify the
    # shape and do not transform spatially, so no border artifacts will be introduced
    # Here we use the new SpatialTransform_2 which uses a new way of parameterizing elastic_deform
    # We use all spatial transformations with a probability of 0.2 per sample. This means that 1 - (1 - 0.1) ** 3 = 27% of samples will be augmented, the rest will just be cropped
    # tr_transforms.append(
    #     SpatialTransform_2(
    #         patch_size, [i // 2 for i in patch_size],
    #         do_elastic_deform=False, deformation_scale=(0, 0.25),
    #         do_rotation=True,
    #         angle_x=(- 5 / 360. * 2 * np.pi, 5 / 360. * 2 * np.pi),
    #         angle_y=(- 5 / 360. * 2 * np.pi, 5 / 360. * 2 * np.pi),
    #         angle_z=(- 5 / 360. * 2 * np.pi, 5 / 360. * 2 * np.pi),
    #         # do_scale=False, scale=(0.95, 1.05),
    #         border_mode_data='constant', border_cval_data=0,
    #         border_mode_seg='constant', border_cval_seg=0,
    #         order_seg=1, 
    #         order_data=3,
    #         # random_crop=False, # applied in dataloader
    #         # p_el_per_sample=0.1, 
    #         # p_scale_per_sample=0.1, 
    #         p_rot_per_sample=0.2
    #     )
    # )

    # now we mirror along all axes
    tr_transforms.append(MirrorTransform(axes=(0, 1, 2), p_per_sample=0.2)) #0.2 # this will apply to both data_key="data" and data_key="seg"

    # brightness transform for 15% of samples
    # tr_transforms.append(BrightnessMultiplicativeTransform((0.5,1), per_channel=True, p_per_sample=0.5)) #0.15
    tr_transforms.append(BrightnessMultiplicativeTransform((0.5,1), per_channel=True, p_per_sample=0.2)) #0.15 this will apply to only data_key="data" as default

    # # gamma transform. This is a nonlinear transformation of intensity values
    # # (https://en.wikipedia.org/wiki/Gamma_correction)
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=False, per_channel=True, p_per_sample=0.15))
    # # we can also invert the image, apply the transform and then invert back
    # tr_transforms.append(GammaTransform(gamma_range=(0.5, 2), invert_image=True, per_channel=True, p_per_sample=0.15))

    # # Gaussian Noise
    # tr_transforms.append(GaussianNoiseTransform(noise_variance=(0, 0.05), p_per_sample=0.15))

    # blurring. Some BraTS cases have very blurry modalities. This can simulate more patients with this problem and
    # thus make the model more robust to it
    # tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.5, 1.5), different_sigma_per_channel=True,
    #                                            p_per_channel=0.5, p_per_sample=0.5)) # 0.15
    tr_transforms.append(GaussianBlurTransform(blur_sigma=(0.8, 1.2), different_sigma_per_channel=True,
                                               p_per_channel=0.2, p_per_sample=0.2)) # 0.15 # this will apply to only data_key="data" as default

    # now we compose these transforms together
    tr_transforms = Compose(tr_transforms)
    return tr_transforms



if __name__ == "__main__":
    print('running surv_dataloader.py')
    batch_tag = 'b1to4' # 'TCGA_LIHC' # 'b1to4'
    surv_endpoint_fname = 'surv_endpoint.csv' # "TCGA_LIHC_surv_endpoint.csv"

    data_root_dict = config.data_root_dict
    imgs_dir = os.path.join(data_root_dict['img'], 'tumorROI_3d_liverNoDilate_xyzMedian_no_zscore/alligned_bbx')
    with open(os.path.join(data_root_dict['nonImg'],'survival', 'data_splits_survPred_{}.json'.format(batch_tag)), mode='r') as f:
        data_splits = json.load(f)

    num_threads_for_brats_example = 8
    task_names = ['recur']

    # train_ids = data_splits['dev']['fold0']['train']
    val_ids = data_splits['dev']['mCVsFold4.Rep1']['train'] # data_splits['dev']['fold0']['val']
    # val_ids = ["BA_002884237", "BA_003286846", "BA_002850767", "BA_005350179", "BA_005547672",
    #                "BA_005566627", "BA_2100146798",
    #               "BA_005189884", "BA_002413188", "BA_000829763", "BA_000999283"]
    patch_size = [48, 80,80] #[45, 70,70] # [257,257,289] # [48,97,129] [48, 352, 480]
    batch_size = 6

    # I recommend you don't use 'iteration oder all training data' as epoch because in patch based training this is
    # really not super well defined. If you leave all arguments as default then each batch sill contain randomly
    # selected patients. Since we don't care about epochs here we can set num_threads_in_multithreaded to anything.
    num_threads_in_multithreaded =6

    # dataloader = survDataLoader(task_names, data_root_dict, imgs_dir, train_ids, batch_size, patch_size,mode='train',clin=True, num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=True, infinite=False,if_prep_tumorMask=True) # 必须要跟MultiThreadedAugmenter中的num_processes一样。

    dataloader = survDataLoader(task_names, data_root_dict, imgs_dir, val_ids, batch_size, patch_size,mode='infer',clin=True,
                                num_threads_in_multithreaded=num_threads_in_multithreaded, return_incomplete=True, infinite=False,
                                if_prep_tumorMask=True, surv_endpoint_fname = None) # 必须要跟MultiThreadedAugmenter中的num_processes一样。

    self = dataloader
    # dataloader.get_indices_cc()

    # batch = next(dataloader)



    # pred_transforms = get_train_transform(expe_config.patch_size)
    pred_gen = MultiThreadedAugmenter(dataloader, None, num_processes=num_threads_in_multithreaded, num_cached_per_queue=2, pin_memory=False)

    import time
    import math
    for epoch in range(10):
        # epoch=0
        try_names_1 = []
        try_names_2 = []
        print('epoch={}'.format(epoch))
        start = time.time()
        for batch_id, pred_batch in enumerate(dataloader):
        # for batch_id, pred_batch in enumerate(pred_gen):
        # for batch_id in range(math.ceil(len(val_ids)/batch_size)):
            # pred_batch = next(pred_gen) # next会触发StopIteration。用enumerate将其放在for后面就不会触发。
            print('epoch {}:batch {},number={}, elapse={}'.format(epoch,batch_id, len(pred_batch['names']), time.time()-start))
            start = time.time()
            try_names_1.append(pred_batch['names'])
            try_names_2.extend(pred_batch['names'])
    
        print('len(val_ids)={}, len(try_names_1)={}, len(try_names_2)={}, len(list(set(try_names_2)))={}'.format(len(val_ids),len(try_names_1),len(try_names_2),len(list(set(try_names_2)))))
    
    
    
    