import torch
import torch.nn as nn

class multiTaskUncertaintySurvLoss(nn.Module):
    # algorithm the same as that in the code of the 2nd author of the paper.
    # theorey based on: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR2018)
    # code learned partly from 
    # (refer1: code from the 2nd author of the paper: https://github.com/yaringal/multi-task-learning-example) (chao's code is logically same as this one)
    # (refer2:https://github.com/Mikoto10032/AutomaticWeightedLoss) and (refer3:https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/blob/a7b8430e9f3da69bbb7cde9fccea77800c9ceb00/src/models/multi_task_learning_model.py#L58)
    """automatically weighted multi-task loss
    Params:
        task_num: int,the number of tasks
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, task_num=2):
        super(multiTaskUncertaintySurvLoss, self).__init__()
        # self.params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True)) for i in range(task_num)]
        self.params = torch.nn.Parameter(torch.zeros(task_num, requires_grad=True)) #log variance. same as that in the 2nd author's code.

    def forward(self, loss_list):
        # import ipdb; ipdb.set_trace()
        current_loss = 0
        # for i in range(len(loss_list)):
        #     # print('{}: {}'.format(i, loss))
        #     # import ipdb; ipdb.set_trace()
        #     current_loss = current_loss + (loss_list[i]/torch.exp(self.params[i])) + 0.5*self.params[i] # here assume the cox loss aproximate t
        # return current_loss
        for i in range(len(loss_list)):
            precision = torch.exp(-self.params[i])
            current_loss = current_loss + (precision * loss_list[i]) + self.params[i] # here assume the cox loss aproximate t
        # if logvar become negative and explode, might use torch.abs(log_var)? by some one in the github.
        return torch.mean(current_loss) # not sure why use mean here. the author said that "Loss have batch_size values which have 20 tensor values, so it uses torch.mean(loss) to take the average of 20 values". However, this is not ture here.

class multiTaskUncertaintySurvLoss_sameAsPaper(nn.Module):
    # algorithm the same as that in the paper.
    # theorey based on: Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics (CVPR2018)
    # code learned partly from 
    # (refer1: code from the 2nd author of the paper: https://github.com/yaringal/multi-task-learning-example) (chao's code is logically same as this one)
    # (refer2:https://github.com/Mikoto10032/AutomaticWeightedLoss) and (refer3:https://github.com/maudzung/TTNet-Real-time-Analysis-System-for-Table-Tennis-Pytorch/blob/a7b8430e9f3da69bbb7cde9fccea77800c9ceb00/src/models/multi_task_learning_model.py#L58)
    """automatically weighted multi-task loss
    Params:
        task_num: int,the number of tasks
        x: multi-task loss
    Examples:
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, task_num=2):
        super(multiTaskUncertaintySurvLoss_sameAsPaper, self).__init__()
        # self.params = [torch.nn.Parameter(torch.zeros(1, requires_grad=True)) for i in range(task_num)]
        self.params = torch.nn.Parameter(torch.zeros(task_num, requires_grad=True)) #log variance. same as that in the 2nd author's code.

    def forward(self, loss_list):
        # import ipdb; ipdb.set_trace()
        current_loss = 0
        for i in range(len(loss_list)):
            # according to the paper, the aim is to predict s:=log(sigma^2), so 1/(sigma^2) = exp(-s). precision here is 1/(sigma^2), the weight before the loss item
            precision = torch.exp(-self.params[i])
            current_loss = current_loss + (precision * loss_list[i]) + 0.5 * self.params[i] # here assume the cox loss aproximate t
        # if logvar become negative and explode, might use torch.abs(log_var)? by some one in the github.
        return torch.mean(current_loss) # not sure why use mean here. the author said that "Loss have batch_size values which have 20 tensor values, so it uses torch.mean(loss) to take the average of 20 values". However, this is not ture here.

if __name__=='__main__':
    x=[0.3,0.4]
    self = multiTaskUncertaintySurvLoss(2)