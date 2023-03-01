# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from advertorch.utils import clamp
from advertorch.utils import jacobian

from .base import Attack
from .base import LabelMixin


class JacobianSaliencyMapAttack(Attack, LabelMixin):
    """
    Jacobian Saliency Map Attack
    This includes Algorithm 1 and 3 in v1, https://arxiv.org/abs/1511.07528v1

    :param predict: forward pass function.
    :param num_classes: number of clasess.
    :param clip_min: mininum value per input dimension.
    :param clip_max: maximum value per input dimension.
    :param gamma: highest percentage of pixels can be modified
    :param theta: perturb length, range is either [theta, 0], [0, theta]

    """

    def __init__(self, predict, num_classes,
                 clip_min=0.0, clip_max=1.0, loss_fn=None,
                 theta=1.0, gamma=1.0, comply_cleverhans=False):
        super(JacobianSaliencyMapAttack, self).__init__(
            predict, loss_fn, clip_min, clip_max)
        self.num_classes = num_classes
        self.theta = theta
        self.gamma = gamma
        self.comply_cleverhans = comply_cleverhans
        self.targeted = True

    def _compute_forward_derivative(self, xadv, y):
        # 计算雅克比矩阵
        jacobians = torch.stack([jacobian(self.predict, xadv, yadv) for yadv in range(self.num_classes)])
        grads = jacobians.view((jacobians.shape[0], jacobians.shape[1], -1))
        grads_target = grads[y, range(len(y)), :]
        grads_other = grads.sum(dim=0) - grads_target
        return grads_target, grads_other

    def _sum_pair(self, grads, dim_x):
        # 通过广播机制，构建大小为 [dim_x,dim_x] 的矩阵，矩阵第(i,j)号元素的内容为某个类别对像素i的偏导 grad_i 与 对像素j的偏导grad_j相加之后的结果
        return grads.view(-1, dim_x, 1) + grads.view(-1, 1, dim_x)

    def _and_pair(self, cond, dim_x):
        # 通过广播机制，构建大小为 [dim_x,dim_x] 的矩阵，矩阵第(i,j)号元素的内容为 cond_i 与 cond_j做 与运算 之后的结果
        # 这个操作可以用来确定本次迭代能更改的两个像素的下标
        return cond.view(-1, dim_x, 1) & cond.view(-1, 1, dim_x)

    def _saliency_map(self, search_space, grads_target, grads_other, y):

        dim_x = search_space.shape[1]

        # alpha in Algorithm 3 line 2
        gradsum_target = self._sum_pair(grads_target, dim_x)
        # beta in Algorithm 3 line 3
        gradsum_other = self._sum_pair(grads_other, dim_x)

        if self.theta > 0:
            # 如果 theta>0, 表示这是一次有目标攻击
            # 这里根据梯度和的结果构建 mask，选择 目标类别对两个像素的偏导和>0且其他类别对两个像素的偏导和<0的mask
            scores_mask = (torch.gt(gradsum_target, 0) &
                           torch.lt(gradsum_other, 0))
        else:
            # 如果 theta<=0, 表示这是无目标攻击
            # 这里根据梯度和的结果构建 mask，选择 目标类别对两个像素的偏导和<0且其他类别对两个像素的偏导和>0的mask
            scores_mask = (torch.lt(gradsum_target, 0) &
                           torch.gt(gradsum_other, 0))
        # torch.ne()——判断元素是否不相等
        # torch.ne(input, other, *, out=None) → Tensor
        # 功能：判断两个数组的元素是否不相等。(相当于是eq的反运算，下面的性质与eq类似)
        # 输出：返回与输入具有相同形状的张量数组，若对应位置上的元素不相等，则该位置上的元素是True，否则是False。

        # 先确定搜索空间中不为0的元素，这些元素表示那些像素点还可以更改
        # 计算出的 scores_mask 要与搜索空间取交集，这样才能确保剩下的像素是可以扰动的
        scores_mask &= self._and_pair(search_space.ne(0), dim_x)
        # 再对score_mask 的对角元素设置为0，
        # 这样设置的原因是后续的操作要选取两个像素点进行扰动，
        # 因此我们现在就要禁止选取的两个像素为同一个像素这种情况发生
        scores_mask[:, range(dim_x), range(dim_x)] = 0

        if self.comply_cleverhans:
            valid = torch.ones(scores_mask.shape[0]).byte()
        else:
            # tensor.any()功能: 如果张量tensor中存在一个元素为True, 那么返回True; 只有所有元素都是False时才返回False
            # 也就是说,如果 scores_mask 里面所有元素值都是0，那就证明没有合适的像素点可以选取了
            valid = scores_mask.view(-1, dim_x * dim_x).any(dim=1)
        # 计算显著值
        scores = scores_mask.float() * (-gradsum_target * gradsum_other)
        # 挑选最大的显著值，并得到其索引
        best = torch.max(scores.view(-1, dim_x * dim_x), 1)[1]
        # torch.remainder 计算除法的元素余数
        # p1 是通过取余数得到，p2 是整除得到，
        # 为什么要这么做呢，是因为我们将显著图拍扁成了一个向量来做最大化，因此要使用取余 整除这两个操作来完成两个像素位置信息复原
        p1 = torch.remainder(best, dim_x)
        p2 = (best / dim_x).long()
        return p1, p2, valid

    def _modify_xadv(self, xadv, batch_size, cond, p1, p2):
        ori_shape = xadv.shape
        xadv = xadv.view(batch_size, -1)
        for idx in range(batch_size):
            # 如果还有攻击的条件，那就继续攻击，没条件就不做任何操作
            if cond[idx] != 0:
                xadv[idx, p1[idx]] += self.theta
                xadv[idx, p2[idx]] += self.theta
        xadv = clamp(xadv, min=self.clip_min, max=self.clip_max)
        xadv = xadv.view(ori_shape)
        return xadv

    def _update_search_space(self, search_space, p1, p2, cond):
        for idx in range(len(cond)):
            if cond[idx] != 0:
                search_space[idx, p1[idx]] -= 1
                search_space[idx, p2[idx]] -= 1

    def perturb(self, x, y=None):
        x, y = self._verify_and_process_inputs(x, y)
        xadv = x
        batch_size = x.shape[0]
        dim_x = int(np.prod(x.shape[1:]))
        max_iters = int(dim_x * self.gamma / 2)
        search_space = x.new_ones(batch_size, dim_x).int()
        curr_step = 0
        yadv = self._get_predicted_label(xadv)

        # Algorithm 1
        while ((y != yadv).any() and curr_step < max_iters):
            grads_target, grads_other = self._compute_forward_derivative(xadv, y)

            # Algorithm 3
            p1, p2, valid = self._saliency_map(search_space, grads_target, grads_other, y)
            # 如果 攻击没成功 且 还有可以选择的像素；那就继续进行攻击
            # 这里的cond记录batch中每个样本还有没有继续攻击的条件（condition），如果没有条件，就不进行攻击
            cond = (y != yadv) & valid
            # 更新搜索空间
            self._update_search_space(search_space, p1, p2, cond)
            # 更新一个批次中每个样本的扰动
            xadv = self._modify_xadv(xadv, batch_size, cond, p1, p2)
            # 做预测
            yadv = self._get_predicted_label(xadv)

            curr_step += 1

        xadv = clamp(xadv, min=self.clip_min, max=self.clip_max)
        return xadv


JSMA = JacobianSaliencyMapAttack
