import math

import numpy as np
import matplotlib
# matplotlib.use('Qt4Agg', force=True)
from matplotlib import colors
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import cv2

colorsList = ['#000000', '#FF0000', '#FFFF00', '#00FF00', '#FF00FF','#0000FF'] # hex color: black, red, yellow, green, pink, blue

# selectSlices with foreground to plot
def selectSlices(seg, num=4):
    z = seg.shape[0]
    if seg.sum() == 0:
        # return [int(i) for i in np.random.choice(z, 1).tolist()] # a random slice could be all 0 as it was padded with 0.
        step = math.ceil(z/num) # will result in ~4 points
        out = list(range(0, z, step))
    else:
        maxis = np.max(seg, axis=(1,2)) > 0
        nonzero_indices = [i for i in np.nonzero(maxis)[0]]
        step = math.ceil(len(nonzero_indices)/num) # will result in ~4 points
        out = list(range(nonzero_indices[0], nonzero_indices[-1], step))
    return out

def plot_label_wt_legend(image_array, labels_dict, colorsList, fig_title):
    '''
    labels_dict = {
            '0':'background',
            '1':'OFFT',
            '2':'SM',
            '3':'VAT',
            '4':'SAT',
            '5':'IMAT'
        }
    '''
    mycmaps = colors.ListedColormap(colorsList)
    norm = plt.Normalize(vmin = 0, vmax = len(colorsList)-1)

    image_out = plt.figure(figsize=(8,8))
    plt.title('{}'.format(fig_title))
    plt.xlim([0, image_array.shape[0]])
    plt.ylim([0, image_array.shape[1]])

    plt.imshow(image_array, cmap=mycmaps, origin='lower', vmin=0, vmax=len(colorsList)-1)

    patch_colors = [mycmaps(norm(int(key))) for key, val in labels_dict.items()]

    patches = [mpatches.Patch(color=patch_colors[int(key)], label="{}".format(labels_dict[key])) for key, val in labels_dict.items()]
    # put those patched as legend-handles into the legend
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.show()

# def myshow(images_list, titles_list, ncols, nrows=1, figsize=(15,5)):
#     #on Mac: the correct_bias func will auto change the matplotlib backend. so below is required when needing interactive GUI.
#     import matplotlib
#     matplotlib.use('Qt4Agg', force=True)
#     matplotlib.get_backend()

#     num_subfig = len(images_list)
#     fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize) # ,sharex=True, sharey=True
#     ax = axes.ravel()
#     for i in range(num_subfig):
#         ax[i].imshow(images_list[i], cmap="gray")
#         ax[i].set_title(titles_list[i])
#     fig.tight_layout()
#     plt.show(block=False)
#     plt.show()

    
def plot_imgs_PV_ART(images_list, isLabel_list, titles_list, ncols=None, nrows=None, figsize=None, out_f=None):
    # plot a list of images/masks
    # 

    #on Mac: the correct_bias func will auto change the matplotlib backend. so below is required when needing interactive GUI.
    # import matplotlib
    # matplotlib.use('Qt4Agg', force=True)
    # matplotlib.get_backend()

    num_subfig = len(images_list[0])
    if ncols is None:
        if nrows is None:
            ncols = 3
            nrows = int(np.ceil(num_subfig/ncols))
        else:
            ncols = int(np.ceil(num_subfig/nrows))
    else:
        nrows = int(np.ceil(num_subfig/ncols))
        
    if figsize is None:
        figsize = tuple([5*ncols, 5*nrows*2])
    
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows * 2, figsize=figsize)  # ,sharex=True, sharey=True
    # ax = axes.ravel()
    # for i in range(num_subfig):
    #     ax[i].imshow(images_list[i], cmap=cmap_list[i]) # if cmap=None, it's for RGB.
    #     ax[i].set_title(titles_list[i])
    #     ax[i].set_xlabel('x')
    #     ax[i].set_ylabel('y')
    # for i in range(num_subfig):
    #     if isLabel_list[i]:
    #         mycmaps = colors.ListedColormap(colorslist)
    #         ax[i].imshow(images_list[i], cmap=mycmaps, vmin=0, vmax=len(colorslist)-1)
    #     else:
    #         ax[i].imshow(images_list[i], cmap='gray') # if cmap=None, it's for RGB.
    #     if titles_list is not None:
    #         ax[i].set_title(titles_list[i])
    #     ax[i].set_xlabel('x')
    #     ax[i].set_ylabel('y')
    count_0 = 0
    count_1 = 0
    for i in range(0, nrows * 2, 1):
        for j in range(ncols):
            if count_0 != len(images_list[0]):
                if i % 2 == 0:
                    axes[i, j].set_title(titles_list[count_0] + "_V")
                    axes[i, j].imshow(images_list[0][count_0], cmap='gray')
                    count_0 = count_0 + 1
            if count_1 != len(images_list[0]):
                if i % 2 != 0:
                    axes[i, j].set_title(titles_list[count_1] + "_A")
                    axes[i, j].imshow(images_list[1][count_1], cmap='gray')
                    count_1 = count_1 + 1
        
    fig.tight_layout()
    if out_f is None:
        plt.show(block=False)
    else:
        plt.savefig(out_f)
    # plt.clf()
    plt.close()


def plot_imgs(images_list, isLabel_list, titles_list, ncols=None, nrows=None, figsize=None, out_f=None):
    # plot a list of images/masks
    # 

    #on Mac: the correct_bias func will auto change the matplotlib backend. so below is required when needing interactive GUI.
    # import matplotlib
    # matplotlib.use('Qt4Agg', force=True)
    # matplotlib.get_backend()

    num_subfig = len(images_list)
    if ncols is None:
        if nrows is None:
            ncols = 3
            nrows = int(np.ceil(num_subfig/ncols))
        else:
            ncols = int(np.ceil(num_subfig/nrows))
    else:
        nrows = int(np.ceil(num_subfig/ncols))
        
    if figsize is None:
        figsize = tuple([7*ncols, 7*nrows])
    
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize) # ,sharex=True, sharey=True
    ax = axes.ravel()
    # for i in range(num_subfig):
    #     ax[i].imshow(images_list[i], cmap=cmap_list[i]) # if cmap=None, it's for RGB.
    #     ax[i].set_title(titles_list[i])
    #     ax[i].set_xlabel('x')
    #     ax[i].set_ylabel('y')
    for i in range(num_subfig):
        if isLabel_list[i]:
            mycmaps = colors.ListedColormap(colorsList)
            ax[i].imshow(images_list[i], cmap=mycmaps, vmin=0, vmax=len(colorsList)-1)
        else:
            ax[i].imshow(images_list[i], cmap='gray') # if cmap=None, it's for RGB.
        if titles_list is not None:
            ax[i].set_title(titles_list[i])
        ax[i].set_xlabel('x')
        ax[i].set_ylabel('y')
        
    fig.tight_layout()
    if out_f is None:
        plt.show(block=False)
    else:
        plt.savefig(out_f)
    # plt.clf()
    plt.close()

def select_plot_imgs(image_vol, isLabel,num_slices_to_select, ncols=None, nrows=None, figsize=None, out_f=None):
    slices_selected = selectSlices(image_vol, num_slices_to_select)
    images_list = [image_vol[i] for i in slices_selected]
    isLabel_list = [isLabel for i in slices_selected]
    titles_list = ['slice{}'.format(i) for i in slices_selected]
    plot_imgs(images_list, isLabel_list, titles_list, ncols=ncols, out_f=out_f)

# def findPlotSegmentationContours(img, seg_list, color_list=[(0, 0, 255)]):
#     # seg_list: one seg for one object.e.g. [liver, liver_cancer]. each numpy array with dtype of np.uint8
#     # color_list: e.g. (0, 0, 255) is red as we use cv2.COLOR_GRAY2BGR
#     img = np.float32(img) # int16 is not fit to cv2.cvtColor
#     if len(img.shape) == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

#     cnt_list = list()
#     for i in range(len(seg_list)):
#         seg = np.uint8(seg_list[i])
#         color = color_list[i]
        
#         contours, hierarchy = cv2.findContours(seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

#         img = cv2.drawContours(img, contours, -1, color, 2)
#         cnt = cv2.drawContours(np.zeros((seg.shape)), contours, -1, (255,255,255))
#         cnt_list.append(cnt)
#     return img, cnt_list

def findPlotSegmentationContours(img, seg, lab_list, color_list=[(0, 255, 0), (255,0,0)]):
    # lab_list: e.g. [0, 1, 2]
    # color_list: e.g. (0,255,0) is green; (255,0,0) is red
    seg_list = []
    for lab in list(lab_list[1:]):
        tmp = np.uint8(seg>=lab)
        seg_list.append(tmp)

    img = np.float32(img) # int16 is not fit to cv2.cvtColor
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cnt_list = list()
    for i in range(len(seg_list)):
        seg_tmp = seg_list[i]
        color = color_list[i]
        
        contours, hierarchy = cv2.findContours(seg_tmp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # seg should be of dtype of np.uint8?

        img = cv2.drawContours(img, contours, -1, color, 5)
        cnt = cv2.drawContours(np.zeros((seg_tmp.shape)), contours, -1, (255,255,255))
        cnt_list.append(cnt)
    return img, cnt_list

def save_2D_full_label_to_RGB_image(full_lab, RGB_dict, out_fp='try.png'):
    # RGB_dict = {
    # 0:[0,0,0], # background
    # 1:[200, 0, 0], # tumor
    # 2:[150, 200, 150]# stroma
    # }
    lab_rgb = np.zeros(list(full_lab.shape)+[3], np.uint8)

    for k, v in RGB_dict.items():
        # k = 1
        rgb_vals = RGB_dict[k]
        lab_rgb[...,0] = np.where(full_lab==k, rgb_vals[0], lab_rgb[...,0])
        lab_rgb[...,1] = np.where(full_lab==k, rgb_vals[1], lab_rgb[...,1])
        lab_rgb[...,2] = np.where(full_lab==k, rgb_vals[2], lab_rgb[...,2])
    cv2.imwrite(out_fp, cv2.cvtColor(lab_rgb, cv2.COLOR_BGR2RGB))