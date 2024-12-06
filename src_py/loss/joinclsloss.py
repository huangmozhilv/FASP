import torch.nn as nn
import torch
from torch.nn import functional as F
import numpy as np
from loss.lovasz_loss import lovasz_softmax
from loss.focal_loss import FocalLoss

class Liu_FocalLoss(nn.Module):
    # refer to: CVPR2022,Partial Class Activation Attention for Semantic Segmentation
    ''' focal loss '''

    def __init__(self, gamma=2, ignore_index=255):
        super(Liu_FocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.crit = nn.BCELoss(reduction='none')

    def binary_focal_loss(self, input, target, valid_mask):
        input = input[valid_mask]
        target = target[valid_mask]
        pt = torch.where(target == 1, input, 1 - input)
        ce_loss = self.crit(input, target)
        loss = torch.pow(1 - pt, self.gamma) * ce_loss
        loss = loss.mean()
        return loss

    def forward(self, input, target):
        valid_mask = (target != self.ignore_index)
        K = target.shape[1]
        total_loss = 0
        for i in range(K):
            total_loss += self.binary_focal_loss(input[:, i], target[:, i], valid_mask[:, i])
        return total_loss / K

class JointClsLoss(nn.Module):
    # refer to: CVPR2022,Partial Class Activation Attention for Semantic Segmentation
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, reduction='mean',bins=(1, 2, 4)):
        super(JointClsLoss, self).__init__()
        self.ignore_index = ignore_index
        # self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        # self.dsn_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cls_criterion = Liu_FocalLoss(ignore_index=ignore_index)
        self.bins = bins

        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label_onehot, bin_size, th=0): #tumor can be very small. and the percentage can be very low. #0.01
        cls_percentage = F.adaptive_avg_pool3d(label_onehot, (bin_size,bin_size,bin_size)) # bin labels for each class are computed separately. 
        cls_label = torch.where(cls_percentage > 0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage))
        cls_label[(cls_percentage < th) & (cls_percentage > 0)] = self.ignore_index
        return cls_label
    #
    # def get_onehot_label(self,label):
    #     label=label.numpy()
    #     for k in range(label.shape[0]):
    #         _mask = [[(label[k]== i) for i in range(3)]]
    #         mask=np.array(_mask).astype(np.float32)
    #         if k==0:
    #             mask_onehot=mask
    #         else:
    #             # print(mask_onehot.shape,mask.shape)
    #             mask_onehot=np.concatenate([mask_onehot,mask],axis=0)
    #     # print(mask_onehot.shape)
    #     return mask_onehot

    def get_onehot_label(self,label):
        for k in range(label.shape[0]):
            # label.shape: b, z, y, x
            _mask = torch.cat([torch.tensor(label[k]== i).unsqueeze(0) for i in range(3)],0).type(torch.float32).unsqueeze(0)
            if k==0:
                mask_onehot=_mask
            else:
                # print(mask_onehot.shape,_mask.shape)
                mask_onehot=torch.cat([mask_onehot,_mask],0)
        # print(mask_onehot.shape)
        return mask_onehot

    def forward(self, preds, target_dict):
        cls_loss = 0
        target = target_dict['target']
        target_onehot=self.get_onehot_label(target)
        # target_onehot=torch.tensor(target_onehot).cuda()
        # for cls_pred, bin_size in zip(preds, self.bins):
        for i in range(len(self.bins)):
            cls_pred = preds[i]
            bin_size = self.bins[i]
            cls_gt = self.get_bin_label(target_onehot, bin_size)
            # lovasz_liver_loss = lovasz_softmax(F.softmax(cls_pred, dim=1),  cls_gt[1], ignore=10)
            # lovasz_tumor_loss = lovasz_softmax(F.softmax(cls_pred, dim=2), cls_gt[2], ignore=10)
            single_cls_loss=self.cls_criterion(cls_pred, cls_gt)
            # print(lovasz_liver_loss,lovasz_tumor_loss,single_cls_loss)
            # cls_loss+=(lovasz_liver_loss+lovasz_tumor_loss+single_cls_loss)/3
            cls_loss += single_cls_loss
        cls_loss=cls_loss/len(self.bins)
        return cls_loss
    

class JointClsLoss2(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, reduction='mean',bins=(1, 2, 4)):
        super(JointClsLoss2, self).__init__()
        self.ignore_index = ignore_index
        # self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        # self.dsn_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cls_criterion = FocalLoss(gamma=2)
        self.bins = bins

        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label, bin_size, th=0): #tumor can be very small. and the percentage can be very low. #0.01
        # label. 0=background, 1=liver, 2=tumor
        # print('label values: {}'.format(str(torch.unique(label, return_counts=True))))
        cls_label = F.adaptive_max_pool3d(label.to(torch.float32), (bin_size,bin_size,bin_size)) # bin labels for each class are computed separately. 
        return cls_label


    def forward(self, preds, target_dict):
        # target： 0=background, 1=liver, 2=tumor. size: [batch_size, z,y,x]. e.g.[2,48,256,320]
        # preds: a list, length same as self.bins. each element in list is of size:[batch_size, 3(class number), bin_size,bin_size,bin_size]. e.g.[2,3,4,4,4]
        cls_loss = 0
        target = target_dict['target']
        # for cls_pred, bin_size in zip(preds, self.bins):
        for i in range(len(self.bins)):
            cls_pred = preds[i]
            bin_size = self.bins[i]
            cls_gt = self.get_bin_label(target, bin_size)
            # lovasz_liver_loss = lovasz_softmax(F.softmax(cls_pred, dim=1),  cls_gt[1], ignore=10)
            # lovasz_tumor_loss = lovasz_softmax(F.softmax(cls_pred, dim=2), cls_gt[2], ignore=10)
            single_cls_loss=self.cls_criterion(cls_pred, cls_gt)
            # print(lovasz_liver_loss,lovasz_tumor_loss,single_cls_loss)
            # cls_loss+=(lovasz_liver_loss+lovasz_tumor_loss+single_cls_loss)/3
            cls_loss += single_cls_loss
        cls_loss=cls_loss/len(self.bins)
        return cls_loss


class JointClsLoss3(nn.Module):
    # refer to: CVPR2022,Partial Class Activation Attention for Semantic Segmentation
    # based on JointClsLoss: add downsample for target
    '''
    DSN : We need to consider two supervision for the model.
    '''

    def __init__(self, ignore_index=255, reduction='mean',bins=(4,4,2)):
        super(JointClsLoss3, self).__init__()
        self.ignore_index = ignore_index
        # self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        # self.dsn_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cls_criterion = Liu_FocalLoss(ignore_index=ignore_index)
        self.bins = bins

        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label_onehot, bin_size, th=0): #tumor can be very small. and the percentage can be very low. #0.01
        cls_percentage = F.adaptive_avg_pool3d(label_onehot, (bin_size,bin_size,bin_size)) # bin labels for each class are computed separately. 
        cls_label = torch.where(cls_percentage > 0, torch.ones_like(cls_percentage), torch.zeros_like(cls_percentage))
        cls_label[(cls_percentage < th) & (cls_percentage > 0)] = self.ignore_index
        return cls_label
    #
    # def get_onehot_label(self,label):
    #     label=label.numpy()
    #     for k in range(label.shape[0]):
    #         _mask = [[(label[k]== i) for i in range(3)]]
    #         mask=np.array(_mask).astype(np.float32)
    #         if k==0:
    #             mask_onehot=mask
    #         else:
    #             # print(mask_onehot.shape,mask.shape)
    #             mask_onehot=np.concatenate([mask_onehot,mask],axis=0)
    #     # print(mask_onehot.shape)
    #     return mask_onehot

    def get_onehot_label(self,label):
        for k in range(label.shape[0]):
            # label.shape: b, z, y, x
            _mask = torch.cat([torch.tensor(label[k]== i).unsqueeze(0) for i in range(3)],0).type(torch.float32).unsqueeze(0)
            if k==0:
                mask_onehot=_mask
            else:
                # print(mask_onehot.shape,_mask.shape)
                mask_onehot=torch.cat([mask_onehot,_mask],0)
        # print(mask_onehot.shape)
        return mask_onehot

    def forward(self, preds, target_dict):
        cls_loss = 0
        ds_weights = [1, 0.5, 0.4, 0.3, 0.2, 0.1]

        target = target_dict['target']
        inSize_list = target_dict['inSize']
        # target_onehot=self.get_onehot_label(target)
        # target_onehot=torch.tensor(target_onehot).cuda()
        # for cls_pred, bin_size in zip(preds, self.bins):
        for i in range(len(self.bins)):
            cls_pred = preds[i]
            bin_size = self.bins[i]
            inSize = inSize_list[i]
            target_tmp = target.unsqueeze(1)
            target_downSamp = F.interpolate(target_tmp, list(inSize))
            target_onehot=self.get_onehot_label(target_downSamp.squeeze(1))
            
            cls_gt = self.get_bin_label(target_onehot, bin_size)
            # lovasz_liver_loss = lovasz_softmax(F.softmax(cls_pred, dim=1),  cls_gt[1], ignore=10)
            # lovasz_tumor_loss = lovasz_softmax(F.softmax(cls_pred, dim=2), cls_gt[2], ignore=10)
            single_cls_loss=self.cls_criterion(cls_pred, cls_gt)
            # print(lovasz_liver_loss,lovasz_tumor_loss,single_cls_loss)
            # cls_loss+=(lovasz_liver_loss+lovasz_tumor_loss+single_cls_loss)/3
            cls_loss += single_cls_loss * ds_weights[i]
        cls_loss=cls_loss/len(self.bins)
        return cls_loss
    

class JointClsLoss4(nn.Module):
    '''
    DSN : We need to consider two supervision for the model.
    based on JointClsLoss2: add downsample for target
    '''

    def __init__(self, ignore_index=255, reduction='mean',bins=(4,4,2)):
        super(JointClsLoss4, self).__init__()
        self.ignore_index = ignore_index
        # self.seg_criterion = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        # self.dsn_criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, reduction=reduction, weight=weight)
        self.cls_criterion = FocalLoss(gamma=2)
        self.bins = bins

        self.cls_weight = 1.0
        if not reduction:
            print("disabled the reduction.")

    def get_bin_label(self, label, bin_size, th=0): #tumor can be very small. and the percentage can be very low. #0.01
        # label. 0=background, 1=liver, 2=tumor
        # print('label values: {}'.format(str(torch.unique(label, return_counts=True))))
        cls_label = F.adaptive_max_pool3d(label.to(torch.float32), (bin_size,bin_size,bin_size)) # bin labels for each class are computed separately. 
        return cls_label


    def forward(self, preds, target_dict):
        # target： 0=background, 1=liver, 2=tumor. size: [batch_size, z,y,x]. e.g.[2,48,256,320]
        # preds: a list, length same as self.bins. each element in list is of size:[batch_size, 3(class number), bin_size,bin_size,bin_size]. e.g.[2,3,4,4,4]
        cls_loss = 0
        ds_weights = [1, 0.5, 0.4, 0.3, 0.2, 0.1]

        target = target_dict['target']
        inSize_list = target_dict['inSize']
        # for cls_pred, bin_size in zip(preds, self.bins):
        for i in range(len(self.bins)):
            cls_pred = preds[i]
            bin_size = self.bins[i]
            inSize = inSize_list[i]
            target_tmp = target.unsqueeze(1)
            target_downSamp = F.interpolate(target_tmp, list(inSize))
            cls_gt = self.get_bin_label(target_downSamp.squeeze(1), bin_size)
            # lovasz_liver_loss = lovasz_softmax(F.softmax(cls_pred, dim=1),  cls_gt[1], ignore=10)
            # lovasz_tumor_loss = lovasz_softmax(F.softmax(cls_pred, dim=2), cls_gt[2], ignore=10)
            single_cls_loss=self.cls_criterion(cls_pred, cls_gt)
            # print(lovasz_liver_loss,lovasz_tumor_loss,single_cls_loss)
            # cls_loss+=(lovasz_liver_loss+lovasz_tumor_loss+single_cls_loss)/3
            cls_loss += single_cls_loss * ds_weights[i]
        cls_loss=cls_loss/len(self.bins)
        return cls_loss

if __name__ == "unitest":
    JC_LOSS = JointClsLoss(bins=(4,4,2))

    JC_LOSS2 = JointClsLoss2(bins=(4,4,2))

    JC_LOSS2 = JointClsLoss3(bins=(4,4,2))