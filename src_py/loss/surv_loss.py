import numpy as np

import torch
from torch import Tensor
import torch.nn as nn

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cox_loss_DeepSurv(fail_indicator,logits, ties='noties'):
    # negative log partiral likelihood # the trainng is to maximize this.
    #### WARNING ####
    # to use cumsum below, the hazards should be ordered in descending order of the survival time. see: https://github.com/jaredleekatzman/DeepSurv/blob/41eed003e5b892c81e7855e400861fa7a2d9da4f/deepsurv/

    # negative log partial likelihood loss
    # refer: https://codeocean.com/capsule/5978670/tree/v1. this code is from cox_regression_loss of Olivier lab in Stanford
    # refer: https://github.com/runopti/stg/blob/master/python/stg/losses.py
    # refer: https://github.com/jaredleekatzman/DeepSurv/blob/master/deepsurv/deep_surv.py
    # DeepSurv解读：https://www.cnblogs.com/CZiFan/p/12674144.html

    # ties: 'noties' or 'efron' or 'breslow'
    if ties=='noties':
        hazards = torch.exp(logits)
        log_sum_hazards = torch.log(torch.cumsum(hazards, dim=0)) #需要提前排序
        uncensored_likelihood = logits - log_sum_hazards
        censored_likelihood = uncensored_likelihood * fail_indicator
        num_events = torch.sum(fail_indicator)
        neg_part_likelihood = -torch.sum(censored_likelihood)/num_events
        
    else:
        raise NotImplementedError
    
    # this will cause "RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn"
    # if num_events ==0:
    #     return 0
    # to avoid this, Chao force each batch to have at least one observation with event occured.
    return neg_part_likelihood

class cox_loss_Olivier(nn.Module):
    # this code is modified from: https://codeocean.com/capsule/5978670/tree/v1. Only variable names were changed.

    def forward(self, log_hazards, ytime, ystatus):
        model_risk = log_hazards
        # import ipdb; ipdb.set_trace()

        idx = ytime.sort(descending=True)[1] #怎么排序？我感觉应该是升序
        ystatus = ystatus[idx]
        model_risk = model_risk[idx]

        hazard_ratio = torch.exp(model_risk)
        
        log_risk = torch.log(torch.cumsum(hazard_ratio,dim=0))
        uncensored_likelihood = model_risk - log_risk
        censored_likelihood = uncensored_likelihood * ystatus
        neg_likelihood = -torch.sum(censored_likelihood)
        num_events = torch.sum(ystatus)
        if num_events ==0:
            return 0
        return neg_likelihood/num_events


class cox_loss_cox_nnet(nn.Module):
    def forward(self, logRR, ytime, ystatus):
        # negative log partiral likelihood # the traing is to maximize this.
        # the underlying algorithm was the same as: negative_log_likelihood of https://github.com/traversc/cox-nnet/blob/gh-pages/cox_nnet/cox_nnet.py
        # code writing refers to both negative_log_likelihood of https://github.com/traversc/cox-nnet/blob/gh-pages/cox_nnet/cox_nnet.py, and CoxSurvLoss of https://github.com/mahmoodlab/MCAT/blob/master/utils/utils.py
        # logRR: [n]. log relative risk of the batch samples. the output of the final fully connected layer. some researchers like Olivier also apply leakyReLU before the output
        # ytime: survival time.
        # ystatus: event=1, no event=0.
        batch_size = len(ytime)
        R_mat = np.zeros([batch_size, batch_size], dtype=int) # init risk sets as explained inhttps://mathweb.ucsd.edu/~rxu/math284/slect5.pdf 
        for i in range(batch_size):
            for j in range(batch_size):
                R_mat[i,j] = ytime[j] >= ytime[i]
        R_mat = torch.FloatTensor(R_mat).to(device)
        # import ipdb; ipdb.set_trace()
        # print('logRR:{}'.format(str(logRR)))
        theta = logRR.reshape(-1) # recast logRR as vector
        exp_theta = torch.exp(theta)
        loss = -torch.mean((theta-torch.log(torch.sum(exp_theta*R_mat, dim=1)))*ystatus) ##exp_theta * R_mat ~ sum the exp_thetas of the patients with greater time e.g., R(t)
        #e.g., all columns of product will have same value or zero, then do a rowSum

        return loss 





class CoxPHLoss(torch.nn.Module):
    # this code is from: https://github.com/havakv/pycox/blob/69940e0b28c8851cb6a2ca66083f857aee902022/pycox/models/loss.py#L407
    # this code is recommended in: https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/#coxph
    # Deepsurv: personalized treatment recommender system using a Cox proportional hazards deep neural network. BMC Medical Research Methodology, 18(1), 2018. https://bmcmedresmethodol.biomedcentral.com/articles/10.1186/s12874-018-0482-1
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    def forward(self, log_h: Tensor, durations: Tensor, events: Tensor) -> Tensor:
        return cox_ph_loss(log_h, durations, events)
    

def cox_ph_loss(log_h: Tensor, durations: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    # this code is from: https://github.com/havakv/pycox/blob/69940e0b28c8851cb6a2ca66083f857aee902022/pycox/models/loss.py#L407
    # this code is recommended in: https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/#coxph
    """Loss for CoxPH model. If data is sorted by descending duration, see `cox_ph_loss_sorted`.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    idx = durations.sort(descending=True)[1]
    events = events[idx]
    log_h = log_h[idx]
    return cox_ph_loss_sorted(log_h, events, eps)

def cox_ph_loss_sorted(log_h: Tensor, events: Tensor, eps: float = 1e-7) -> Tensor:
    # this code is from: https://github.com/havakv/pycox/blob/69940e0b28c8851cb6a2ca66083f857aee902022/pycox/models/loss.py#L407
    # this code is recommended in: https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/#coxph
    """Requires the input to be sorted by descending duration time.
    See DatasetDurationSorted.
    We calculate the negative log of $(\frac{h_i}{\sum_{j \in R_i} h_j})^d$,
    where h = exp(log_h) are the hazards and R is the risk set, and d is event.
    We just compute a cumulative sum, and not the true Risk sets. This is a
    limitation, but simple and fast.
    """
    if events.dtype is torch.bool:
        events = events.float()
    events = events.view(-1)
    log_h = log_h.view(-1)
    gamma = log_h.max()
    log_cumsum_h = log_h.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
    return - log_h.sub(log_cumsum_h).mul(events).sum().div(events.sum())


class NegativeLogLikelihood(nn.Module):
    # this code is modified from NegativeLogLikelihood for DeepSurv by czifan:https://github.com/czifan/DeepSurv.pytorch/blob/f572a8a7d3ce5ad10609bd273ce3accaf7ea4b66/networks.py#L66
    # def __init__(self, config):
    #     super(NegativeLogLikelihood, self).__init__()
    #     self.L2_reg = config['l2_reg']
    #     self.reg = Regularization(order=2, weight_decay=self.L2_reg)

    # def forward(self, risk_pred, y, e, model):
    #     R_mat = torch.ones(y.shape[0], y.shape[0])
    #     R_mat[(y.T - y) > 0] = 0
    #     log_loss = torch.exp(risk_pred) * R_mat
    #     log_loss = torch.sum(log_loss, dim=0) / torch.sum(R_mat, dim=0)
    #     log_loss = torch.log(log_loss).reshape(-1, 1)
    #     neg_log_loss = -torch.sum((risk_pred-log_loss) * e) / torch.sum(e)
    #     l2_loss = self.reg(model)
    #     return neg_log_loss + l2_loss
    def forward(self, risk_pred, ytime, ystatus):
        # import ipdb; ipdb.set_trace()
        ytime = ytime.reshape(-1,1)
        ystatus = ystatus.reshape(-1,1)
        R_mat = torch.ones(ytime.shape[0], ytime.shape[0])
        R_mat[(ytime.T - ytime) > 0] = 0
        R_mat = torch.FloatTensor(R_mat).to(device)
        log_loss = torch.exp(risk_pred) * R_mat
        log_loss = torch.sum(log_loss, dim=0) / torch.sum(R_mat, dim=0)
        log_loss = torch.log(log_loss).reshape(-1, 1)
        num_events = torch.sum(ystatus)
        if num_events==0:
            return 0
        else:
            neg_log_loss = -torch.sum((risk_pred-log_loss) * ystatus) / num_events
            return neg_log_loss

class NegLogPartialLikelihood(nn.Module):
    # updated on date: 20230919
    # this code is modified from NegativeLogLikelihood for DeepSurv by czifan:https://github.com/czifan/DeepSurv.pytorch/blob/f572a8a7d3ce5ad10609bd273ce3accaf7ea4b66/networks.py#L66
    # refer: the NegLogPartialLikelihood in DeepSurv paper and 《医学统计学》（第三版）（主编孙振球）
    # refer: 《DeepSurv: personalized treatment recommender system using a Cox proportional hazards deep neural network》
    # refer: https://humboldt-wi.github.io/blog/research/information_systems_1920/group2_survivalanalysis/#coxph
    # refer: Time-to-Event Prediction with Neural Networks and Cox Regression (https://jmlr.org/papers/volume20/18-424/18-424.pdf)
    def forward(self, log_risk, ytime, ystatus):
        # MUST: ystatus is a list which should should contain at least one event

        # log_risk: the output of the neural network.
        # import ipdb; ipdb.set_trace()
        # log_risk -= torch.max(log_risk) 
        
        # log_risk=torch.randn([6,1])
        # ytime=torch.tensor([10,20,30,5,1,12])
        # ystatus=torch.tensor([1,0,0,1,1,0])


        # log_risk = log_risk - torch.max(log_risk) ## scale log_risk to [-inf,0] to avoid any number beyond 88.7 giving exp(number) to be larger than 3.40282e+38 which would be stored as inf. After scaling with this method, the result and the gradients will not change as those were produced mathmatcially. (refer:https://effectivemachinelearning.com/PyTorch/7._Numerical_stability_in_PyTorch). (softmax运算的上溢和下溢: https://zhuanlan.zhihu.com/p/29376573) this is not right in this case as torch.exp(log_risk) should multiply to R_mat.
        ytime = ytime.reshape(-1,1) # log_risk = torch.
        ystatus = ystatus.reshape(-1,1)
        num_events = torch.sum(ystatus)

        # if num_events==0:
        #     # return torch.tensor(0, dtype=ytime.dtype).to(device) # ytime is int?
        #     return torch.tensor(0, dtype=log_risk.dtype).to(device)
        # else:
        R_mat = torch.zeros(ytime.shape[0], ytime.shape[0]) # let T_i denotes the possibly censored event time of individual i and in R_mat: 1 denotes an individual at risk at time T_i (not censored and have not experienced the event before time T_i) 
        R_mat[(ytime - ytime.T) >= 0] = 1 # to have idx corresponding to log_risk
        R_mat = torch.FloatTensor(R_mat).to(device)
        risk_R_mat = torch.exp(log_risk) * R_mat
        # loss_tmp = torch.sum(loss_tmp, dim=0) / torch.sum(R_mat, dim=0)
        sum_risk_R_mat = torch.sum(risk_R_mat, dim=0)
        log_sum_risk_R_mat = torch.log(sum_risk_R_mat).reshape(-1, 1)
        neg_log_loss = -torch.sum((log_risk-log_sum_risk_R_mat) * ystatus) / num_events
        return neg_log_loss

