"""
This module provide the attack method of "CW".
L2 distance metrics especially
"""
from __future__ import division
from __future__ import print_function

from builtins import range
import logging
import numpy as np

import torch
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.autograd.gradcheck import zero_gradients
import torch.utils.data.dataloader as Data
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import numpy as np
from .base import Attack

__all__ = ['CW_L2_Attack', 'CW_L2']


# https://github.com/advboxes/AdvBox/blob/master/adversarialbox/attacks/cw2_pytorch.py

class CW_L2_Attack(Attack):
    """
    Uses Adam to minimize the CW L2 objective function

    Paper link: https://arxiv.org/abs/1608.04644
    """

    def __init__(self, model):
        super(CW_L2_Attack, self).__init__(model)

        self._model = model._model
        mean, std = model._preprocess
        self.mean = torch.from_numpy(mean)
        self.std = torch.from_numpy(std)

    def _apply(self,
               adversary,
               max_iterations=1000,
               lr=0.01,
               const=10.0,
               binary_search_steps=10,
               kappa=40,
               num_labels=10):
        # 本方法还是主要讨论无定向攻击

        self._adversary = adversary
        img = self._adversary.original.copy()
        pre_label = adversary.original_label

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        # 定向
        if adversary.is_targeted_attack:
            pass
        # 无定向攻击
        else:
            # 攻击目标标签 必须使用one hot编码
            tlab = F.one_hot(pre_label, num_classes=10)
            # Variable(torch.from_numpy(np.eye(num_labels)[pre_label]).to(device).float())
        boxmin, boxmax = self.model.bounds()
        # print("boxmin={}, boxmax={}".format(boxmin, boxmax))
        # logging.info("boxmin={}, boxmax={}".format(boxmin, boxmax))
        # print(tlab)
        shape = adversary.original.shape
        # c的初始化边界
        lower_bound = 0
        c = const
        upper_bound = 1e10
        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = [np.zeros(shape)]
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        boxmul = (boxmax - boxmin) / 2.
        boxplus = (boxmin + boxmax) / 2.

        for outer_step in range(binary_search_steps):
            # print("o_bestl2={} c={}".format(o_bestl2,c)  )
            # logging.info("o_bestl2={} c={}".format(o_bestl2, c))
            # 把原始图像转换成图像数据和扰动的形态
            timg = torch.from_numpy(np.arctanh((img - boxplus) / boxmul * 0.999999)).to(device).float()
            modifier = torch.zeros_like(timg).to(device).float()
            # 图像数据的扰动量梯度可以获取
            modifier.requires_grad = True
            # 定义优化器 仅优化modifier
            optimizer = torch.optim.Adam([modifier], lr=lr)

            for iteration in range(1, max_iterations + 1):
                optimizer.zero_grad()
                # 定义新输入
                new_img = torch.tanh(modifier + timg) * boxmul + boxplus
                output = self._model((new_img - self.mean) / self.std)
                # 定义cw中的损失函数
                loss2 = torch.linalg.norm((new_img.view(1, -1) - timg.view(1, -1)), ord=2, dim=1)[0]
                # 对原始类别的信息分数
                real = output[0][pre_label]
                # 除了正确类别的信息分数之外的最大信息分数
                other = torch.max((1 - tlab[0]) * output)
                if adversary.is_targeted_attack:
                    pass
                    # loss1 = other - real + k
                else:
                    # 无目标攻击的置信度损失函数，让正确类别的信息分数下降，让其他类别的信息分数上涨
                    loss1 = -other + real + k
                    # loss1=other-real+k
                loss1 = torch.clamp(loss1, min=0)
                # 这里的的 c 相当于调节系数
                loss1 = c * loss1
                loss = loss1 + loss2
                loss.backward(retain_graph=True)
                optimizer.step()
                l2 = loss2
                sc = output.data.cpu().numpy()
                # 输出的是概率
                # pro=F.softmax(self._model(new_img),dim=1)[0].data.cpu().numpy()[target_label]
                # 获取模型对样本的预测类别
                # np.argmax 取得最大值的下标索引
                pred = np.argmax(sc)

                if iteration % (max_iterations // 10) == 0:
                    # print("iteration={} loss={} loss1={} loss2={} pred={}".format(iteration,loss,loss1,loss2,pred))
                    logging.info(
                        "iteration={} loss={} loss1={} loss2={} pred={}".format(iteration, loss, loss1, loss2, pred))

                if adversary.is_targeted_attack:
                    pass
                    # if (l2 < o_bestl2) and (np.argmax(sc) == target_label):
                    #     # print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                    #     print("attack success l2={} target_label={}".format(l2, target_label))
                    #     o_bestl2 = l2
                    #     o_bestscore = pred
                    #     o_bestattack = new_img.data.cpu().numpy()
                else:
                    # 更新l2范数，和 adv_img
                    if (l2 < o_bestl2) and (pred != pre_label):
                        # print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                        # print("attack success l2={} label={}".format(l2,pred))
                        logging.info("attack success l2={} label={}".format(l2, pred))
                        o_bestl2 = l2
                        o_bestscore = pred
                        o_bestattack = new_img.data.cpu().numpy()

            confidence_old = -1

            if adversary.is_targeted_attack:
                pass
                # if (o_bestscore == target_label) and (o_bestscore != -1):
                #     # 攻击成功 减小c
                #     upper_bound = min(upper_bound, c)
                #     if upper_bound < 1e9:
                #         print()
                #         confidence_old = c
                #         c = (lower_bound + upper_bound) / 2
                # else:
                #     lower_bound = max(lower_bound, c)
                #     confidence_old = c
                #     if upper_bound < 1e9:
                #         c = (lower_bound + upper_bound) / 2
                #     else:
                #         c *= 10

            else:
                # 如果攻击成功，那我们就要进行二分查找，重新迭代
                if (o_bestscore != pre_label) and (o_bestscore != -1):
                    # 攻击成功 减小c
                    upper_bound = min(upper_bound, c)
                    if upper_bound < 1e9:
                        confidence_old = c
                        c = (lower_bound + upper_bound) / 2
                # 如果没攻击成功，那我们就扩大c
                else:
                    lower_bound = max(lower_bound, c)
                    confidence_old = c
                    if upper_bound < 1e9:
                        c = (lower_bound + upper_bound) / 2
                    else:
                        c *= 10

            # print("outer_step={} c {}->{}".format(outer_step,confidence_old,c))
            # logging.info("outer_step={} c {}->{}".format(outer_step, confidence_old, c))
        # print(o_bestattack)

        if o_bestscore != -1:
            """
                If adversarial_label the target label that we are finding.
                The adversarial_example and adversarial_label will be accepted and
                True will be returned.
                :return: bool
            """
            if adversary.try_accept_the_example(o_bestattack, o_bestscore):
                return adversary

        return adversary

    def inf2box(self, x):
        '''
        input:  -\infty <-> + \infty
        output: 0 <-> 1
        '''
        return 1 / 2 * (torch.tanh(x) + 1)

    def box2inf(self, x):
        '''
        input: x, 0 <-> 1
        output: -\infty <-> + \infty
        '''

        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))


CW_L2 = CW_L2_Attack
