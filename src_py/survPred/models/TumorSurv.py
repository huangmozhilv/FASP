
import os
import sys
sys.path.append(os.getcwd())

from turtle import forward
from typing import OrderedDict
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import survPred.config as config


# when kernel_size is odd, stride is even, e.g. k=3,s=2, output size is floor() of the (w-k+2p)/2+1.因此，为了使得k is odd, s=2时输出尺寸是输入尺寸的1/2，使用自定义的Conv3dSame替换Conv3d。
class Conv3dSame(torch.nn.Conv3d):
    # will cause extra GPU memory cost with a padded x. so Chao suggest to use it only for stride>1. if still OOM, replace Conv3dSame(stride=2) with maxpooling.
    # output size: math.ceil(i / s)
    # works for any kernel_size, stride, dilation
    # modified from Conv2dSame in :https://github.com/pytorch/pytorch/issues/67551
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i_h, i_w, i_d = x.size()[-3:]

        pad_h = self.calc_same_pad(i=i_h, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=i_w, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])
        pad_d = self.calc_same_pad(i=i_d, k=self.kernel_size[2], s=self.stride[2], d=self.dilation[2])

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            x = F.pad(
                x, [pad_d // 2, pad_d - pad_d // 2, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                # in F.pad: The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward. For example, to pad only the last dimension of the input tensor, then pad has the form (padding_left,padding_right); to pad the last 2 dimensions of the input tensor, then use (padding_left,padding_right, padding_top,padding_bottom); to pad the last 3 dimensions, use (padding_left,padding_right,padding_top,padding_bottom,padding_front,padding_back)
            )
        return F.conv3d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


class Conv2dSame(torch.nn.Conv2d):
    # will cause extra GPU memory cost with a padded x. so Chao suggest to use it only for stride>1. if still OOM, replace Conv3dSame(stride=2) with maxpooling.
    # output size: math.ceil(i / s)
    # works for any kernel_size, stride, dilation
    # modified from Conv2dSame in :https://github.com/pytorch/pytorch/issues/67551
    def calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        return max((math.ceil(i / s) - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        i_h, i_w = x.size()[-2:]

        pad_h = self.calc_same_pad(i=i_h, k=self.kernel_size[0], s=self.stride[0], d=self.dilation[0])
        pad_w = self.calc_same_pad(i=i_w, k=self.kernel_size[1], s=self.stride[1], d=self.dilation[1])

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                # in F.pad: The padding size by which to pad some dimensions of input are described starting from the last dimension and moving forward. For example, to pad only the last dimension of the input tensor, then pad has the form (padding_left,padding_right); to pad the last 2 dimensions of the input tensor, then use (padding_left,padding_right, padding_top,padding_bottom); to pad the last 3 dimensions, use (padding_left,padding_right,padding_top,padding_bottom,padding_front,padding_back)
            )
        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups
        )


def norm(n_out, type='BN'):
    if type == 'BN':
        norm = nn.BatchNorm3d(n_out, eps=1.001e-5, affine=True)
    elif type == 'IN':
        norm = nn.InstanceNorm3d(n_out, affine=True)
    return norm


# def activation(type='LeakyReLU'):
def activation(type='ReLU'):
    if type == 'ReLU':
        act = nn.ReLU(inplace=True)  # 可能导致dyingReLU
        # act = nn.ReLU(inplace=False)
    elif type == 'LeakyReLU':
        act = nn.LeakyReLU(inplace=True)
        # act = nn.LeakyReLU(inplace=False)
    return act


##### SelfCBAM：如下。参考自selection-and-reinforcement module(SRAM)参考https://github.com/dreamenwalker/KMAP-Net/blob/main/attentionNet1117debug1.py
class channelAttention(nn.Module):
    # modified from CAM of SRAM from:https://github.com/dreamenwalker/KMAP-Net/blob/main/attentionNet1117debug1.py
    """ Channel attention module"""

    # def __init__(self, in_dim):
    def __init__(self):  # reduction_ratio is the same as the C in the SEnet
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)  # out: one signal for each channel for each batch sample
        self.max_pool = nn.AdaptiveMaxPool3d(1)  # out: one signal for each channel for each batch sample
        # self.fc = nn.Sequential(
        #     nn.Linear(channel, channel // reduction, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(channel // reduction, channel, bias=False),
        #     nn.Sigmoid()
        # )
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        # b, c, _, _,_ = x.size()
        avgpool_channel = self.avg_pool(x)  # [n,c,1,1,1]
        # avgpool_weight_tmp = avgpool_channel-torch.mean(avgpool_channel,dim=1).unsqueeze(dim=1)
        avgpool_weight_tmp = avgpool_channel - torch.mean(avgpool_channel, keepdim=True, dim=1)
        avgpool_weight = self.act(avgpool_weight_tmp)
        avgpool_feature = torch.mul(avgpool_weight, x)

        maxpool_weight_tmp = self.max_pool(avgpool_feature)
        maxpool_weight = self.sigmoid(maxpool_weight_tmp)
        # maxpool_weight = self.softmax(maxpool_weight_tmp)
        # maxpool_weight = torch.mul(maxpool_weight_tmp,x) # reinforcement
        # maxpool_weight = self.softmax(maxpool_weight)

        # out = self.act(avgpool_feature)
        out = torch.mul(maxpool_weight, x) + x
        # y = self.fc(y).view(b, c, 1, 1)
        # return x * y.expand_as(x)
        return out


class spatialAttention(nn.Module):
    # modified from SAM of SRAM in refer1:https://github.com/dreamenwalker/KMAP-Net/blob/main/attentionNet1117debug1.py#L102
    def __init__(self, in_chan):
        super().__init__()

        self.spaAtt7 = nn.Sequential(
            nn.Conv3d(in_chan, 1, kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(inplace=True)
        )
        self.spaAtt3 = nn.Sequential(
            nn.Conv3d(in_chan, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.spaAtt1 = nn.Sequential(
        # nn.Conv3d(in_chan, 1, kernel_size=1, bias=False),
        # nn.ReLU(inplace=True)
        # ) # by Chao.

        # self.pool_conv = nn.Sequential(
        #     nn.Conv3d(4, 1, kernel_size=7, bias=False),
        #     nn.Sigmoid()
        # )
        self.pool_conv = nn.Conv3d(4, 1, kernel_size=7, stride=1, padding=3, bias=False)
        self.pool_act = nn.Sigmoid()

    def forward(self, x):
        maxpool_spatial, _ = torch.max(x, dim=1, keepdim=True)  # torch.max（）返回是个turtle，包含了 value，index。所以这里用,_来表示,index
        avgpool_spatial = torch.mean(x, dim=1, keepdim=True)
        # 拼接 max、avg、1x1卷积输出的 feature map
        spaAtt7 = self.spaAtt7(x)
        spaAtt3 = self.spaAtt3(x)
        # spaAtt1 = self.spaAtt1(x) # by Chao

        spa_pool = torch.cat([maxpool_spatial, avgpool_spatial, spaAtt7, spaAtt3], dim=1)
        # spa_pool =  torch.cat([maxpool_spatial, avgpool_spatial, spaAtt7, spaAtt3, spaAtt1], dim=1) # by Chao
        ## 选择7*7来聚焦局部空间信息
        pool_conv = self.pool_conv(spa_pool)
        spatial_att = self.pool_act(pool_conv)
        # return self.pool_conv(spa_pool)
        # out = pool_conv + x
        out = spatial_att * x + x
        return out


class CSAMmodule(nn.Module):
    # refer1: https://github.com/dreamenwalker/KMAP-Net/blob/main/attentionNet1117debug1.py#L114
    # refer2: https://github.com/junfu1115/DANet/blob/master/encoding/models/sseg/danet.py#L95
    def __init__(self, in_chan, if_CA=False, if_SA=False):
        super().__init__()
        self.if_CA = if_CA
        self.if_SA = if_SA
        self.channel_attention = channelAttention()
        self.spatial_attention = spatialAttention(in_chan=in_chan)

    def forward(self, input_xs):
        if self.if_CA:
            out = self.channel_attention(input_xs)
        else:
            out = input_xs

        if self.if_SA:
            out = self.spatial_attention(out)
        else:
            pass

        return out


##### CSAMmodule: as above.

def patch_split(input, bin_size):
    """
    refer: https://github.com/lsa1997/PCAA/blob/main/networks/caanet.py
    bin_size: a list. will split the input of [Z,H,W] into bins with number for Z,H,W equal to [bin_num_z, bin_num_h, bin_num_w]
    Z,H,W should be divisable to each element in bin_size.
    bz: bin number for Z. rz: bin size for z.
    b c (bz rz) (bh rh) (bw rw) -> b (bz bh bw) rz rh rw c.
    """
    B, C, Z, H, W = input.size()
    bin_num_z = bin_size[0]
    bin_num_h = bin_size[1]
    bin_num_w = bin_size[2]

    rZ = Z // bin_num_z
    rH = H // bin_num_h  # rH: bin h
    rW = W // bin_num_w
    out = input.view(B, C, bin_num_z, rZ, bin_num_h, rH, bin_num_w, rW)
    out = out.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()  # [B, bin_num_z, bin_num_h, bin_num_w, rZ,, rH, rW, C]
    out = out.view(B, -1, rZ, rH, rW, C)  # [B, bin_num_z * bin_num_h * bin_num_w, rZ, rH, rW, C]
    return out


def patch_recover(input, bin_size):
    """
    refer: https://github.com/lsa1997/PCAA/blob/main/networks/caanet.py
    bz: bin number for Z. rz: bin size for z.
    b (bh bw) rh rw c -> b c (bh rh) (bw rw)
    """
    B, N, rZ, rH, rW, C = input.size()
    bin_num_z = bin_size[0]
    bin_num_h = bin_size[1]
    bin_num_w = bin_size[2]

    Z = rZ * bin_num_z
    H = rH * bin_num_h
    W = rW * bin_num_w
    out = input.view(B, bin_num_z, bin_num_h, bin_num_w, rZ, rH, rW, C)
    out = out.permute(0, 7, 1, 3, 5, 2, 4, 6).contiguous()  # [B, C, bin_num_h, rH, bin_num_w, rW]
    out = out.view(B, C, Z, H, W)  # [B, C, H, W]
    return out


class GCN(nn.Module):
    # refer: https://github.com/lsa1997/PCAA/blob/main/networks/caanet.py
    def __init__(self, num_node, num_channel):
        super(GCN, self).__init__()
        self.conv1 = nn.Conv2d(num_node, num_node, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Linear(num_channel, num_channel, bias=False)

    def forward(self, x):
        # x: [B, bin_num_h * bin_num_w, K, C]. so use conv2d
        out = self.conv1(x)
        out = self.relu(out + x)
        out = self.conv2(out)
        return out


class ASPPModule(nn.Module):
    # refer: https://github.com/lsa1997/PCAA/blob/main/networks/deeplabv3.py#L8
    """
    Reference:
        Chen, Liang-Chieh, et al. *"Rethinking Atrous Convolution for Semantic Image Segmentation."*
    """

    def __init__(self, features, inner_features=32, out_features=64, dilations=(12, 24, 36)):
        super(ASPPModule, self).__init__()
        self.conv1 = nn.Sequential(nn.AdaptiveAvgPool3d((1, 1, 1)),
                                   nn.Conv3d(features, inner_features, kernel_size=1, padding=0, dilation=1,
                                             bias=False),
                                   norm(inner_features),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv3d(features, inner_features, kernel_size=1, padding=0, dilation=1, bias=False),
            norm(inner_features),
            nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv3d(features, inner_features, kernel_size=3, padding=dilations[0], dilation=dilations[0], bias=False),
            norm(inner_features),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv3d(features, inner_features, kernel_size=3, padding=dilations[1], dilation=dilations[1], bias=False),
            norm(inner_features),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv3d(features, inner_features, kernel_size=3, padding=dilations[2], dilation=dilations[2], bias=False),
            norm(inner_features),
            nn.ReLU(inplace=True))

        self.bottleneck = nn.Sequential(
            nn.Conv3d(inner_features * 5, out_features, kernel_size=3, stride=1, padding=1, bias=False),
            norm(out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        _, _, z, h, w = x.size()
        feat1 = F.interpolate(self.conv1(x), size=(z, h, w), mode='trilinear', align_corners=False)
        feat2 = self.conv2(x)
        feat3 = self.conv3(x)
        feat4 = self.conv4(x)
        feat5 = self.conv5(x)
        out = torch.cat((feat1, feat2, feat3, feat4, feat5), 1)
        out = self.bottleneck(out)
        return out


class CAAM(nn.Module):
    """
    Class Activation Attention Module
    # refer: https://github.com/lsa1997/PCAA/blob/main/networks/caanet.py
    """

    def __init__(self, feat_in, num_classes=3, bin_size=[4, 4, 4]):
        super(CAAM, self).__init__()
        feat_inner = feat_in // 2
        self.bin_size = bin_size
        # self.dropout = nn.Dropout3d(0.1)

        # self.conv_cam = nn.Conv3d(feat_in, num_classes, kernel_size=1)
        # self.conv_cam = nn.Sequential(
        #     nn.Conv3d(feat_in,feat_in, kernel_size=3, stride=1, padding=1, bias=False),
        #     norm(feat_in),
        #     activation(),
        #     nn.Conv3d(feat_in, num_classes, kernel_size=1)
        # )
        self.conv_cam = nn.Sequential(
            ASPPModule(feat_in, inner_features=feat_in, out_features=feat_in * 2, dilations=(12, 24, 36)),  # 8, 16, 24
            nn.Conv3d(feat_in * 2, num_classes, kernel_size=1)
        )


        # self.camAtt = spatialAttention(num_classes)

        self.conv_fuse = nn.Sequential(
            nn.Conv3d(feat_in + 3, feat_in, kernel_size=1, bias=False),
            norm(feat_in),
            activation(),
            nn.Conv3d(feat_in, feat_in, kernel_size=3, padding=1, bias=False),
            norm(feat_in),
            activation()
        )

    def forward(self, x):
        # cam = self.conv_cam(self.dropout(x))  # [B, K,z,y,x]
        cam0 = self.conv_cam(x)  # [B, K,z,y,x]

        #### hide ###
        # cls_score = self.sigmoid(self.pool_cam(cam0)) # [B,K, bin_num_z,bin_num_y, bin_num_x] # 为何使用sigmoid而不是softmax？因为一个像素点可能属于多个类别？by Chao.

        # # residual = x # [B, C, z,y,x]
        # cam = patch_split(cam0, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, K]
        # x = patch_split(x, self.bin_size) # [B, bin_num_h * bin_num_w, rH, rW, C]

        # B = cam.shape[0]
        # rZ = cam.shape[2]
        # rH = cam.shape[3]
        # rW = cam.shape[4]
        # K = cam.shape[-1]
        # C = x.shape[-1]
        # cam = cam.view(B, -1, rZ*rH*rW, K) # [B, bin_num_z * bin_num_h * bin_num_w, rZ * rH * rW, K]
        # x = x.view(B, -1, rZ*rH*rW, C) # [B, bin_num_z * bin_num_h * bin_num_w, rZ * rH * rW, C]

        # bin_confidence = cls_score.view(B,K,-1).transpose(1,2).unsqueeze(3) # [B, bin_num_z * bin_num_h * bin_num_w, K, 1]
        # pixel_confidence = F.softmax(cam, dim=2) # [B, bin_num_z * bin_num_h * bin_num_w, rZ * rH * rW, K]

        # local_feats = torch.matmul(pixel_confidence.transpose(2, 3), x) * bin_confidence # [B, bin_num_z * bin_num_h * bin_num_w, K, C]
        # local_feats = self.gcn(local_feats) # [B, bin_num_z * bin_num_h * bin_num_w, K, C]
        # global_feats = self.fuse(local_feats) # [B, 1, K, C]
        # global_feats = self.relu(global_feats).repeat(1, x.shape[1], 1, 1) # [B, bin_num_z * bin_num_h * bin_num_w, K, C]

        # query = self.proj_query(x) # [B, bin_num_z * bin_num_h * bin_num_w, rZ * rH * rW, C//2]
        # key = self.proj_key(local_feats) # [B, bin_num_z * bin_num_h * bin_num_w, K, C//2]
        # value = self.proj_value(global_feats) # [B, bin_num_z * bin_num_h * bin_num_w, K, C//2]

        # aff_map = torch.matmul(query, key.transpose(2, 3)) # [B, bin_num_z * bin_num_h * bin_num_w, rZ * rH * rW, K]
        # aff_map = F.softmax(aff_map, dim=-1)
        # out = torch.matmul(aff_map, value) # [B, bin_num_z * bin_num_h * bin_num_w, rZ * rH * rW, C]

        # out = out.view(B, -1, rZ, rH, rW, value.shape[-1]) # [B, bin_num_z * bin_num_h * bin_num_w, rZ, rH, rW, C]
        # out = patch_recover(out, self.bin_size) # [B, C, Z, H, W]

        # # out = residual + self.conv_out(out)
        # out = self.conv_out(out)
        #### hide ###
        cls_score = None

        # cam0 = self.camAtt(cam0)
        out = self.conv_fuse(torch.cat([x, F.softmax(cam0, dim=1)], dim=1))

        return out, cls_score, cam0


class CSAMbasicBlock(nn.Module):
    # refer to "conv_block" in https://github.com/dreamenwalker/KMAP-Net/blob/main/attentionNet1117debug1.py#L155
    def __init__(self, in_chan, out_chans_list, stride, reduction=None, if_CA=False, if_SA=False, if_CAAM=False,
                 bin_size=4):
        # # Arguments
        #     input_tensor: input tensor
        #     kernels: defualt 3, the kernel size of middle conv layer at main path
        #     out_chan_list: list of integers, the out_chan_lists of 3 conv layer at main path
        super().__init__()
        self.if_CA = if_CA
        self.if_SA = if_SA
        self.if_CAAM = if_CAAM
        self.stride = stride

        self.bin_size = [bin_size for i in range(3)]  # 3D

        out_chans_1, out_chans_2, out_chans_3 = out_chans_list

        self.op1 = nn.Sequential(
            Conv3dSame(in_chan, out_chans_1, kernel_size=3, stride=stride, bias=False),
            nn.BatchNorm3d(out_chans_1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_chans_1, out_chans_2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(out_chans_2),
            nn.ReLU(inplace=True)
        )

        if if_CAAM:
            self.CAAM = CAAM(out_chans_2, num_classes=3, bin_size=self.bin_size)
        else:
            self.CSAM = CSAMmodule(in_chan=out_chans_2, if_CA=if_CA, if_SA=if_SA)

        self.shortcut_op = nn.Sequential(
            Conv3dSame(in_chan, out_chans_3, kernel_size=3, stride=stride, bias=False),
            nn.BatchNorm3d(out_chans_3)
        )
        self.act = nn.ReLU(inplace=True)

    def forward(self, input_tensor):
        x = self.op1(input_tensor)
        if self.if_CAAM:
            x, cls_score, cam0 = self.CAAM(x)
        else:
            x = self.CSAM(x)
            cls_score = None
            cam0 = None

        shortcut = self.shortcut_op(input_tensor)
        out = self.act(x + shortcut)

        return out, cls_score, cam0


# adapted from https://github.com/haamoon/mmtm/blob/1c81cfefad5532cfb39193b8af3840ac3346e897/mmtm.py
class IPA(nn.Module):
    # IPA: inter-phase attention
    def __init__(self, dim_ph1, dim_ph2, ratio):
        super(IPA, self).__init__()
        # import ipdb; ipdb.set_trace()
        dim = dim_ph1 + dim_ph2
        dim_out = int(2 * dim / ratio)
        self.fc_squeeze = nn.Linear(dim, dim_out)

        self.fc_ph1 = nn.Linear(dim_out, dim_ph1)
        self.fc_ph2 = nn.Linear(dim_out, dim_ph2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, ART, PV):
        squeeze_array = []
        for tensor in [ART, PV]:
            tview = tensor.view(tensor.shape[:2] + (-1,))
            squeeze_array.append(torch.mean(tview, dim=-1))
        squeeze = torch.cat(squeeze_array, 1)
        # import ipdb; ipdb.set_trace()

        excitation = self.fc_squeeze(squeeze)
        excitation = self.relu(excitation)

        ART_out = self.fc_ph1(excitation)
        PV_out = self.fc_ph2(excitation)

        ART_out = self.sigmoid(ART_out)
        PV_out = self.sigmoid(PV_out)

        dim_diff = len(ART.shape) - len(ART_out.shape)
        ART_out = ART_out.view(ART_out.shape + (1,) * dim_diff)

        dim_diff = len(PV.shape) - len(PV_out.shape)
        PV_out = PV_out.view(PV_out.shape + (1,) * dim_diff)

        return ART * ART_out, PV * PV_out

# SSFE: scale-specific feature extractor. former name is scaleDecoder.
class SSFE(nn.Module):
    def __init__(self, inChan, outChan, out_size=[3, 11, 15], stride=2, n_phase=2):
        super().__init__()
        self.n_phase = n_phase
        self.mp = nn.AdaptiveMaxPool3d(out_size)
        self.conv1 = nn.Sequential(
            nn.Conv3d(inChan, outChan, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(outChan),
            nn.ReLU(inplace=True)
        )
        if n_phase == 2:
            self.conv2 = nn.Sequential(
                nn.Conv3d(inChan, outChan, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm3d(outChan),
                nn.ReLU(inplace=True)
            )
            self.conv_fuse = nn.Sequential(
                nn.Conv3d(outChan * 2, outChan, kernel_size=1, bias=False),
                nn.BatchNorm3d(outChan),
                nn.ReLU(inplace=True)
            )
        self.fc = nn.Sequential(
            nn.Linear(out_size[0] * out_size[1] * out_size[2] * outChan, 128),
            activation(),
            nn.Linear(128, 64),
            activation()
        )

    def forward(self, x_ph_1, x_ph_2=None):
        out = self.mp(x_ph_1)
        out = self.conv1(out)

        if x_ph_2 is not None:
            out_ph_2 = self.mp(x_ph_2)
            out_ph_2 = self.conv2(out_ph_2)

            out = self.conv_fuse(torch.cat((out, out_ph_2), dim=1))

        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out


class tumorSurv(nn.Module):
    # refer: https://codeocean.com/capsule/5978670/tree/v1. this code is from Olivier lab in Stanford
    # 根据LiverNet_CAAM8Mmtm_xyzMedian_resizeToMedian的实验结果：1/4分辨率层的seg loss一直比1/8,1/16,1/32这三层的大很多，而后面三层的seg loss水平差不多。因此本网络只保留后三层的seg输出。
    # 另外还做了其他改动，例如增加encoder的宽度
    def __init__(self, task_names, n_phase=2, clin=False):
        super().__init__()
        self.n_phase = n_phase
        self.task_names = task_names
        self.clin = clin

        # self.inplane = 32
        self.bins = [4, 4, 4, 4, 4]  # [4,4,4,4,2]
        self.num_class = 3

        # outChan_list = [16, 32, 64, 128, 256, 512]  # [16,24,48,96,192,384]
        # outChan_list = [32, 64, 64, 96,96, 512]
        outChan_list = [16,32,64,128,256,512]
        self.stride_list = stride_list = [(1, 2, 2), (1, 2, 2), 2, 2, 2, 2]  # stride: int or tuple

        ## ART
        self.Conv_in_ph1 = nn.Sequential(
            # nn.Conv3d(2,outChan_list[0],kernel_size=3,padding=1, bias=False),
            # norm(outChan_list[0]),
            # activation(),
            # Conv3dSame(outChan_list[0],outChan_list[0],kernel_size=3,stride=stride_list[0], bias=False),
            Conv3dSame(1, outChan_list[0], kernel_size=3, stride=stride_list[0], bias=False),
            norm(outChan_list[0]),
            activation(),
            nn.Conv3d(outChan_list[0], outChan_list[0], kernel_size=3, padding=1, bias=False),
            norm(outChan_list[0]),
            activation()
        )

        self.Conv_hidden_1_ph1 = CSAMbasicBlock(outChan_list[0], [outChan_list[1], outChan_list[1], outChan_list[1]], stride=stride_list[1], if_SA=False, if_CA=False, if_CAAM=False)  # out: [48,64,80]
        self.Conv_hidden_2_ph1 = CSAMbasicBlock(outChan_list[1], [outChan_list[2], outChan_list[2], outChan_list[2]], stride=stride_list[2], if_SA=False, if_CA=False, if_CAAM=False)  # out: [24,32,40]
        self.Conv_hidden_3_ph1 = CSAMbasicBlock(outChan_list[2], [outChan_list[3], outChan_list[3], outChan_list[3]], stride=stride_list[3], if_SA=False, if_CA=False, if_CAAM=False)  # out: [12,16,20]
        self.Conv_hidden_4_ph1 = CSAMbasicBlock(outChan_list[3], [outChan_list[4], outChan_list[4], outChan_list[4]], stride=stride_list[4], if_SA=False, if_CA=False, if_CAAM=False)  # out: [6,8,10]
        self.Conv_hidden_5_ph1 = CSAMbasicBlock(outChan_list[4], [outChan_list[5], outChan_list[5], outChan_list[5]], stride=stride_list[5], if_SA=False, if_CA=False, if_CAAM=False)  # out: [3,4,5]

        #
        if self.n_phase == 1:
            # pass
            self.convBlock_fuse = nn.Sequential(
                nn.Conv3d(outChan_list[5], 128, kernel_size=1, bias=False),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            )
        elif self.n_phase > 2:
            raise ValueError('n_phase should be 1 or 2')
        elif self.n_phase == 2:
            ## PV
            self.Conv_in_ph2 = nn.Sequential(
                # nn.Conv3d(2,outChan_list[0],kernel_size=3,padding=1, bias=False),
                # norm(outChan_list[0]),
                # activation(),
                # Conv3dSame(outChan_list[0],outChan_list[0],kernel_size=3,stride=stride_list[0], bias=False),
                Conv3dSame(1, outChan_list[0], kernel_size=3, stride=stride_list[0], bias=False),
                norm(outChan_list[0]),
                activation(),
                nn.Conv3d(outChan_list[0], outChan_list[0], kernel_size=3, padding=1, bias=False),
                norm(outChan_list[0]),
                activation()
            )

            self.Conv_hidden_1_ph2 = CSAMbasicBlock(outChan_list[0], [outChan_list[1], outChan_list[1], outChan_list[1]],
                                             stride=stride_list[1], if_SA=False, if_CA=False,
                                             if_CAAM=False)  # out: [48,64,80]
            self.Conv_hidden_2_ph2 = CSAMbasicBlock(outChan_list[1], [outChan_list[2], outChan_list[2], outChan_list[2]],
                                             stride=stride_list[2], if_SA=False, if_CA=False,
                                             if_CAAM=False)  # out: [24,32,40]
            self.Conv_hidden_3_ph2 = CSAMbasicBlock(outChan_list[2], [outChan_list[3], outChan_list[3], outChan_list[3]],
                                             stride=stride_list[3], if_SA=False, if_CA=False,
                                             if_CAAM=False)  # out: [12,16,20]
            self.Conv_hidden_4_ph2 = CSAMbasicBlock(outChan_list[3], [outChan_list[4], outChan_list[4], outChan_list[4]],
                                             stride=stride_list[4], if_SA=False, if_CA=False,
                                             if_CAAM=False)  # out: [6,8,10]
            self.Conv_hidden_5_ph2 = CSAMbasicBlock(outChan_list[4], [outChan_list[5], outChan_list[5], outChan_list[5]],
                                             stride=stride_list[5], if_SA=False, if_CA=False,
                                             if_CAAM=False)  # out: [3,4,5]

            ## IPA
            self.IPA1 = IPA(outChan_list[0], outChan_list[0], 4)
            self.IPA2 = IPA(outChan_list[1], outChan_list[1], 4)
            self.IPA3 = IPA(outChan_list[2], outChan_list[2], 4)
            self.IPA4 = IPA(outChan_list[3], outChan_list[3], 4)
            self.IPA5 = IPA(outChan_list[4], outChan_list[4], 4)

            self.convBlock_fuse = nn.Sequential(
                nn.Conv3d(outChan_list[3] * 2, 256, kernel_size=1, bias=False), #128 256
                nn.BatchNorm3d(256),
                nn.ReLU(inplace=True),
                nn.Conv3d(256, 128, kernel_size=1, bias=False),
                nn.BatchNorm3d(128),
                nn.ReLU(inplace=True)
            )

        # multiscale_decoder
        self.SSFE1 = SSFE(outChan_list[0], outChan_list[0])
        self.SSFE2 = SSFE(outChan_list[1], outChan_list[0])
        self.SSFE3 = SSFE(outChan_list[2], outChan_list[0])
        self.SSFE4 = SSFE(outChan_list[3], outChan_list[0])

        self.decoder_deepest = nn.Sequential(
            nn.Linear(38400, 128),  # 13440, 73728, 36864, 7680 76032  126720
            activation(),
            nn.Linear(128, 64),
            activation()
        )

        #
        self.decoder_dict = nn.ModuleDict()
        for i in task_names:
            self.decoder_dict[i] = nn.Sequential(
                nn.Linear(320, 128),  # 13440, 73728, 36864, 7680
                activation(),
                nn.Linear(128, 64),
                activation()
            )

        # can concatenate the output of previous layers with clinical features or other features before input to the final classfication layer
        self.classifier_dict = nn.ModuleDict()
        # self.LeakyReLu_dict = nn.ModuleDict()
        for i in task_names:
            if self.clin:
                self.classifier_dict[i] = nn.Linear(64 + len(config.clin_feats), 1)  # 64
            else:
                self.classifier_dict[i] = nn.Linear(64, 1)
            # self.LeakyReLu_dict[i] = nn.LeakyReLU(negative_slope=-1) # LeakyReLU was added here in Olivier's code, why? Chao think this is not needed according to the cox's equations. In my view, h(t|x) should be ok to be smaller than h0(t).

    def forward(self, x_ph1, x_ph2=None, clin_data=None):
        model_res = dict()
        model_res['t_sne'] = dict()

        model_res['cls_score_art'] = []
        model_res['cls_score_pv'] = []

        model_res['featMapSize'] = []

        model_res['seg_pred_art'] = []
        model_res['seg_pred_pv'] = []

        cam0_art_list = []
        cam0_pv_list = []

        model_res["cam"] = []

        #
        scale_fc_list = []

        # 1
        model_res['featMapSize'].append(x_ph1.shape)
        # x_ph1: [1,48,80,80]
        x1_ph1 = self.Conv_in_ph1(x_ph1)  # [16,48,40,40]

        if self.n_phase == 1:
            pass
        elif self.n_phase > 2:
            raise ValueError('n_phase should be 1 or 2')
        elif self.n_phase == 2:
            x1_ph2 = self.Conv_in_ph2(x_ph2)  # 

        if self.n_phase == 1:
            scale_fc = self.SSFE1(x1_ph1)
        elif self.n_phase == 2:
            scale_fc = self.SSFE1(x1_ph1, x1_ph2)
        scale_fc_list.append(scale_fc)

        model_res['featMapSize'].append(x1_ph1.shape)
        model_res["cam"].append(x1_ph1)
        # 2
        if self.n_phase == 1:
            pass
        elif self.n_phase > 2:
            raise ValueError('n_phase should be 1 or 2')
        elif self.n_phase == 2:
            x1_ph1, x1_ph2 = self.IPA1(x1_ph1, x1_ph2)
            x1_ph2, cls_score_ph2, cam0_ph2 = self.Conv_hidden_1_ph2(x1_ph2)  # [32,48,20,20]
        x1_ph1, cls_score_ph1, cam0_ph1 = self.Conv_hidden_1_ph1(x1_ph1)  # [6]

        if self.n_phase == 1:
            scale_fc = self.SSFE2(x1_ph1)
        elif self.n_phase == 2:
            scale_fc = self.SSFE2(x1_ph1, x1_ph2)
        scale_fc_list.append(scale_fc)

        model_res["cam"].append(x1_ph1)
        model_res['cls_score_art'].append(cls_score_ph1)
        model_res['cls_score_pv'].append(cls_score_ph2)

        model_res['featMapSize'].append(x1_ph1.shape)
        cam0_art_list.append(cam0_ph1)
        cam0_pv_list.append(cam0_ph2)

        # 3
        if self.n_phase == 1:
            pass
        elif self.n_phase > 2:
            raise ValueError('n_phase should be 1 or 2')
        elif self.n_phase == 2:
            x1_ph1, x1_ph2 = self.IPA2(x1_ph1, x1_ph2)
            x1_ph2, cls_score_ph2, cam0_ph2 = self.Conv_hidden_2_ph2(x1_ph2)  # [64,24,10,10]
        x1_ph1, cls_score_ph1, cam0_ph1 = self.Conv_hidden_2_ph1(x1_ph1)  # [6]

        if self.n_phase == 1:
            scale_fc = self.SSFE3(x1_ph1)
        elif self.n_phase == 2:
            scale_fc = self.SSFE3(x1_ph1, x1_ph2)
        scale_fc_list.append(scale_fc)

        model_res['cls_score_art'].append(cls_score_ph1)
        model_res['cls_score_pv'].append(cls_score_ph2)

        model_res['featMapSize'].append(x1_ph1.shape)

        cam0_art_list.append(cam0_ph1)
        cam0_pv_list.append(cam0_ph2)

        # 4
        if self.n_phase == 1:
            pass
        elif self.n_phase > 2:
            raise ValueError('n_phase should be 1 or 2')
        elif self.n_phase == 2:
            x1_ph1, x1_ph2 = self.IPA3(x1_ph1, x1_ph2)
            x1_ph2, cls_score_ph2, cam0_ph2 = self.Conv_hidden_3_ph2(x1_ph2)  # [128,12,5,5]
        x1_ph1, cls_score_ph1, cam0_ph1 = self.Conv_hidden_3_ph1(x1_ph1)  # [6]

        if self.n_phase == 1:
            scale_fc = self.SSFE4(x1_ph1)
        elif self.n_phase == 2:
            scale_fc = self.SSFE4(x1_ph1, x1_ph2)
        scale_fc_list.append(scale_fc)

        model_res['cls_score_art'].append(cls_score_ph1)
        model_res['cls_score_pv'].append(cls_score_ph2)

        model_res['featMapSize'].append(x1_ph1.shape)

        cam0_art_list.append(cam0_ph1)
        cam0_pv_list.append(cam0_ph2)

        # collect all levels seg
        model_res['seg_pred_art'] = cam0_art_list
        model_res['seg_pred_pv'] = cam0_pv_list

        # bottleneck
        if self.n_phase == 1:
            # x1 = torch.flatten(x1_ph1,1)
            x1 = self.convBlock_fuse(x1_ph1)
            x1 = torch.flatten(x1, 1)
        elif self.n_phase > 2:
            raise ValueError('n_phase should be 1 or 2')
        elif self.n_phase == 2:
            x1 = torch.cat((x1_ph1, x1_ph2), dim=1) # x1: 256, 12, 5, 5
            x1 = self.convBlock_fuse(x1)  # 

            #AVE_POOL
            # x1= nn.AdaptiveAvgPool3d(1)(x1)
            x1 = torch.flatten(x1, 1) # [38400]
        logits_dict = dict()  # this is a 'risk score', which represents the logarithm of the ratio of teh individual hazard to the baseline hazard. so, the hazard ratio?

        x1 = self.decoder_deepest(x1) # [64]
        x1 = torch.cat(scale_fc_list + [x1], dim=1) # [320]

        for i in self.task_names:
            # the following code credits to https://github.com/mahmoodlab/MCAT/models/model_coattn.py
            decoder_out = self.decoder_dict[i](x1)
            decoder_out = torch.flatten(decoder_out, 1) # 64

            model_res['t_sne'][i] = decoder_out
            #
            if self.clin:
                decoder_out = torch.cat((decoder_out, clin_data), dim=1)
            else:
                pass
            logits_dict[i] = self.classifier_dict[i](decoder_out)
            # logits_dict[i] = self.LeakyReLu_dict[i](logits_dict[i]) # why olivier 2020 Nature machine intelligence paper add this step.是不是因为基线风险h0(t)肯定是最低的，有任何危险因素存在的h(t)风险值肯定都比h0(t)高，所以这里使用negative_lope=-1的leakyReLu以限制网络输出始终大于0，这样h(t)/h0(t)始终大于等于1.
            # logits_dict[i] = torch.clamp(logits_dict[i], max=88.7) #here, the tensors dtype are float32, so if >88.7, torch.exp() in the cox loss will give inf and the final output of cox loss will be NaN. In fact, this does not solve the problem thoroughly. as the following sum operation will also give number >88.7 and result in inf.

        model_res['logits'] = logits_dict
        return model_res


if __name__ == "__main__":
    task_names = ['recur']  # 'recur','death'
    x_ph2 = torch.randn([6, 1,48,80,80]) #[6, 2, 44, 271, 322] [6, 2, 48, 352, 480]
    x_ph1 = torch.randn([6, 1,48,80,80])
    clin_data = torch.randn([6, 5])
    # bin_size=[]
    # for bin_h, bin_w in zip((2,4,4,4), (4)):
    #     print(bin_h,bin_w)

    self = tumorSurv(task_names, n_phase=2, clin=False)

    model_res = self(x_ph1, x_ph2, clin_data=clin_data)

    #   