

import numpy as np
import torch

# dice value for classes except for bg
def dice_coef_torch(pred_label, gt_label, c_list=[0,1,2], c_excluded=[0]):
    # list of classes
    dice_c = []
    for c in range(len(c_list)): # dice not for bg
        if c in c_excluded:
            pass
        else:
            # intersection
            ints = torch.sum(((pred_label == c_list[c]) * 1) * ((gt_label == c_list[c]) * 1))
            # sum
            sums = torch.sum(((pred_label == c_list[c]) * 1) + ((gt_label == c_list[c]) * 1)) + 0.0001
            dice_c.append((2.0 * ints) / sums)

    return dice_c

# dice value for classes except for bg
def dice_coef_numpy(pred_label, gt_label):
    # list of classes
    c_list = np.unique(gt_label)

    dice_c = []
    for c in range(1,len(c_list)): # dice not for bg
        # intersection
        ints = np.sum(((pred_label == c_list[c]) * 1) * ((gt_label == c_list[c]) * 1))
        # sum
        sums = np.sum(((pred_label == c_list[c]) * 1) + ((gt_label == c_list[c]) * 1)) + 0.0001
        dice_c.append((2.0 * ints) / sums)

    return dice_c