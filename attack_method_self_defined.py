import torch
import torch.nn as nn
from torchattacks.attack import Attack
from pixel_selector import pixel_selector_by_attribution
import torch.optim as optim
import torch.nn.functional as F

# adv_images = (A.mm(B_box) + RP).reshape(original_shape)
# Copyright (c) 2018-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import numpy as np
import torch

from advertorch.utils import clamp
from advertorch.utils import jacobian


class JSMA(Attack):
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

    def __init__(self, model, num_classes,
                 clip_min=0.0, clip_max=1.0,
                 theta=1.0, gamma=1.0,
                 comply_cleverhans=False, ):
        super().__init__('JSMA', model)
        self.num_classes = num_classes
        self.theta = theta
        self.gamma = gamma

        self.clip_min = clip_min
        self.clip_max = clip_max
        self.comply_cleverhans = comply_cleverhans
        self.targeted = True

        self._supported_mode = ['default', 'targeted']

    def _verify_and_process_inputs(self, x, y):
        if self.targeted:
            assert y is not None

        if not self.targeted:
            if y is None:
                y = self.model(x)

        x = x.detach().clone()
        y = y.detach().clone()
        return x, y

    def _get_predicted_label(self, x):
        """
        Compute predicted labels given x. Used to prevent label leaking
        during adversarial training.

        :param x: the model's input tensor.
        :return: tensor containing predicted labels.
        """
        with torch.no_grad():
            outputs = self.model(x)
        _, y = torch.max(outputs, dim=1)
        return y

    def _compute_forward_derivative(self, xadv, y):
        # 计算雅克比矩阵
        jacobians = torch.stack([jacobian(self.model, xadv, yadv) for yadv in range(self.num_classes)])
        grads = jacobians.view((jacobians.shape[0], jacobians.shape[1], -1))
        grads_target = grads[y, range(len(y)), :]
        grads_other = grads.sum(dim=0) - grads_target
        return grads_target, grads_other

    def _sum_pair(self, grads, dim_x):
        # 通过广播机制，构建大小为 [dim_x,dim_x] 的矩阵，矩阵第(i,j)号元素的内容为某个类别对像素i的偏导 grad_i 与 对像素j的偏导grad_j相加之后的结果
        return grads.view(-1, dim_x, 1) + grads.view(-1, 1, dim_x)

    def _and_pair(self, condition, dim_x):
        # 通过广播机制，构建大小为 [dim_x,dim_x] 的矩阵，矩阵第(i,j)号元素的内容为 cond_i 与 cond_j做 与运算 之后的结果
        # 这个操作可以用来确定本次迭代能更改的两个像素的下标
        return condition.view(-1, dim_x, 1) & condition.view(-1, 1, dim_x)

    def _saliency_map(self, search_space, grads_target, grads_other, y):

        dim_x = search_space.shape[1]

        # alpha in Algorithm 3 line 2
        gradsum_target = self._sum_pair(grads_target, dim_x)
        # beta in Algorithm 3 line 3
        gradsum_other = self._sum_pair(grads_other, dim_x)

        # 无论 theta 是大于0还是小于0，都是有目标攻击
        if self.theta > 0:
            # 如果 theta>0, 则使用原论文中的公式（8）来计算攻击显著图
            # 这里根据梯度和的结果构建 mask，选择 目标类别对两个像素的偏导和>0且其他类别对两个像素的偏导和<0的mask
            scores_mask = (torch.gt(gradsum_target, 0) &
                           torch.lt(gradsum_other, 0))
        else:
            # 如果 theta<=0, 则使用原论文中的公式（9）来计算攻击显著图
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

    def _modify_xadv(self, xadv, batch_size, condition, p1, p2):
        ori_shape = xadv.shape
        xadv = xadv.view(batch_size, -1)
        for idx in range(batch_size):
            # 如果还有攻击的条件，那就继续攻击，没条件就不做任何操作
            if condition[idx] != 0:
                xadv[idx, p1[idx]] += self.theta
                xadv[idx, p2[idx]] += self.theta
        xadv = clamp(xadv, min=self.clip_min, max=self.clip_max)
        xadv = xadv.view(ori_shape)
        return xadv

    def _update_search_space(self, search_space, p1, p2, condition):
        for idx in range(len(condition)):
            if condition[idx] != 0:
                search_space[idx, p1[idx]] -= 1
                search_space[idx, p2[idx]] -= 1

    def forward(self, x, y):
        x, y = x.detach().clone(), y.detach().clone()
        xadv = x
        batch_size = x.shape[0]
        dim_x = int(np.prod(x.shape[1:]))
        max_iters = int(dim_x * self.gamma / 2)
        search_space = x.new_ones(batch_size, dim_x).int()
        curr_step = 0
        yadv = self._get_predicted_label(xadv)

        # Algorithm 1
        # 这里 y！=yadv while循环继续则表明当前进行的是有目标攻击
        while ((y != yadv).any() and curr_step < max_iters):
            grads_target, grads_other = self._compute_forward_derivative(xadv, y)

            # Algorithm 3
            p1, p2, valid = self._saliency_map(search_space, grads_target, grads_other, y)
            # 如果 攻击没成功 且 还有可以选择的像素；那就继续进行攻击
            # 这里的 condition 记录batch中每个样本还有没有继续攻击的条件，如果没有条件，就不进行攻击
            # 还是在判定是否继续目标攻击
            condition = (y != yadv) & valid
            # 更新搜索空间
            self._update_search_space(search_space, p1, p2, condition)
            # 更新一个批次中每个样本的扰动
            xadv = self._modify_xadv(xadv, batch_size, condition, p1, p2)
            # 做预测
            yadv = self._get_predicted_label(xadv)

            curr_step += 1

        xadv = torch.clamp(xadv, min=self.clip_min, max=self.clip_max).detach()
        return xadv


# 这里主要定义两个基准函数 FGSM，PGD 在图像的可更改像素比较少的情况下进行对抗样本的生成的效果是怎样的
class Limited_FGSM(Attack):
    r"""
    FGSM in the paper 'Explaining and harnessing adversarial examples'
    [https://arxiv.org/abs/1412.6572]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.007)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        # >>> attack = torchattacks.FGSM(model, eps=0.007)
        # >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, A, KP, RP, eps=0.007):
        super().__init__("Limited_FGSM", model)
        self.eps = eps
        self.A = A.clone().detach()
        self.KP = KP.clone().detach()
        self.RP = RP.clone().detach()

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_origin = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # A is a matrix, size is n*k
        # KP is a matrix, size is k*1
        # RP is a matrix, size is n*1
        original_shape = images_origin.shape

        A, KP, RP = self.A, self.KP, self.RP
        KP_origin = KP.detach().clone()
        KP.requires_grad = True

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        images_origin_reconstruct = (A.mm(KP) + RP).reshape(original_shape)
        outputs = self.model(images_origin_reconstruct)

        # Calculate loss
        if self._targeted:
            cost = -loss(outputs, target_labels)
        else:
            cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, KP,
                                   retain_graph=False, create_graph=False)[0]
        # 只对 K Pixels 沿着梯度方向进行扰动操作，然后使用 K Pixels 来重建图像
        # 总的来说，图像的改变区域只有 K Pixels 区域
        adv_images = (A.mm(self.eps * grad.sign() + KP_origin) + RP).reshape(original_shape)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        return adv_images


class Limited_CW3(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """

    def __init__(self, model, A, KP, RP, c=1, kappa=0, steps=200, lr=0.01, ):
        super().__init__("Limited_CW3", model)
        self.A = A.clone().detach()
        self.KP = KP.clone().detach()
        self.RP = RP.clone().detach()

        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr

        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        images_origin = images.clone().detach().to(self.device)
        # A is a matrix, size is n*k
        # KP is a matrix, size is k*1
        # RP is a matrix, size is n*1
        original_shape = images_origin.shape
        A, KP, RP = self.A, self.KP, self.RP
        KP[KP == 0.0] = 0.1 / 255
        KP[KP == 1.0] = 1 - 0.1 / 255
        KP_origin = KP.detach().clone()
        KP_origin.requires_grad = True

        w = self.inverse_tanh_space(KP_origin).clone().detach()
        w.requires_grad = True

        def reconstruct_image(adv_KP):
            adv_images = (A.mm(adv_KP) + RP).reshape(original_shape)
            return adv_images

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        # w = torch.zeros_like(images).detach() # Requires 2x times
        # w = self.inverse_tanh_space(images).detach()
        # w.requires_grad = True

        best_adv_images = images.clone().detach()
        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        optimizer = optim.Adam([w], lr=self.lr)

        for step in range(self.steps):
            # Get adversarial images
            adv_images = reconstruct_image(self.tanh_space(w))

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)
            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward()
            optimizer.step()

            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()

            # filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            mask = (1 - correct) * (best_L2 > current_L2.detach())
            best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2

            mask = mask.view([-1] + [1] * (dim - 1))
            best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return best_adv_images
                prev_cost = cost.item()

        return best_adv_images

    def tanh_space(self, x):
        # box->inf
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # inf->box
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        labels = labels.cpu()
        one_hot_labels = torch.eye(len(outputs[0])).to(self.device)[labels]

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

        if self._targeted:
            return torch.clamp((i - j), min=-self.kappa)
        else:
            return torch.clamp((j - i), min=-self.kappa)


# deprecated
class Limited_PGD(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : Linf

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 0.3)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGD(model, eps=8/255, alpha=1/255, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, A, KP, RP, eps=0.3,
                 alpha=4 / 255, steps=40, random_start=True):
        super().__init__("Limited_PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps

        self.A = A.clone().detach()
        self.KP = KP.clone().detach()
        self.RP = RP.clone().detach()

        self.random_start = random_start
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_origin = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # A is a matrix, size is n*k
        # KP is a matrix, size is k*1
        # RP is a matrix, size is n*1

        original_shape = images_origin.shape
        A, KP, RP = self.A, self.KP, self.RP

        def reconstruct_image(KP):
            adv_images = (A.mm(KP) + RP).reshape(original_shape)
            return adv_images

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        adv_KP = KP.detach().clone()

        if self.random_start:
            # Starting at a uniformly random point
            adv_KP = adv_KP + torch.empty_like(adv_KP).uniform_(-self.eps, self.eps)
            adv_KP = torch.clamp(adv_KP, min=0, max=1).detach()

        for _ in range(self.steps):
            adv_KP = adv_KP.requires_grad_(True)
            adv_images = reconstruct_image(adv_KP)
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            grad = torch.autograd.grad(cost, adv_KP,
                                       retain_graph=False, create_graph=False)[0]

            # adv_images = adv_images.detach() + self.alpha * grad.sign()
            # delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            # adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            adv_KP = adv_KP.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_KP - KP, min=-self.eps, max=self.eps)
            adv_KP = torch.clamp(KP + delta, min=0, max=1).detach()

        adv_images = reconstruct_image(adv_KP)
        return adv_images


class Limited_PGDL2(Attack):
    r"""
    PGD in the paper 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    [https://arxiv.org/abs/1706.06083]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        eps (float): maximum perturbation. (Default: 1.0)
        alpha (float): step size. (Default: 0.2)
        steps (int): number of steps. (Default: 40)
        random_start (bool): using random initialization of delta. (Default: True)

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.PGDL2(model, eps=1.0, alpha=0.2, steps=40, random_start=True)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=1.0, alpha=0.2, steps=40, random_start=False, eps_for_division=1e-10,
                 pixel_k=1):
        super().__init__("Limited_PGDL2", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.pixel_k = pixel_k
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_origin = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        # A is a matrix, size is n*k
        # KP is a matrix, size is k*1
        # RP is a matrix, size is n*1

        original_shape = images_origin.shape
        A, KP, RP = pixel_selector_by_attribution(self.model, images_origin, labels, self.pixel_k)

        def reconstruct_image(KP):
            adv_images = (A.mm(KP) + RP).reshape(original_shape)
            return adv_images

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        loss = nn.CrossEntropyLoss()

        # adv_images = images.clone().detach()
        # batch_size = len(images)

        adv_KP = KP.detach().clone()

        if self.random_start:
            # Starting at a uniformly random point
            delta = torch.empty_like(adv_KP).normal_()
            n = delta.norm(p=2)
            r = torch.zeros_like(n).uniform_(0, 1)
            delta *= r / n * self.eps

        for _ in range(self.steps):
            adv_KP.requires_grad = True
            adv_images = reconstruct_image(adv_KP)
            outputs = self.model(adv_images)

            # Calculate loss
            if self._targeted:
                cost = -loss(outputs, target_labels)
            else:
                cost = loss(outputs, labels)

            # Update adversarial images
            # grad 和 adv_KP 都是K*1的矩阵
            grad = torch.autograd.grad(cost, adv_KP,
                                       retain_graph=False, create_graph=False)[0]
            grad_norms = torch.linalg.norm(grad, ord=2, dim=0).item() + self.eps_for_division
            # grad_norms.view(batch_size, 1, 1, 1) 只是为了让广播机制发挥作用，
            # grad =grad/grad_norms 是在规范化梯度向量，让梯度的变成单位向量（即模长=1）
            grad = grad / grad_norms
            adv_KP = adv_KP.detach() + self.alpha * grad

            delta = adv_KP - KP
            delta_norms = torch.linalg.norm(delta, ord=2, dim=0).item()
            # 求最大范数与扰动范数的比值
            factor = self.eps / delta_norms
            # 如果最大范数与累计扰动范数的比值 比1大，那就证明还可以继续添加扰动，则比值置为1
            # 如果最大范数与累计扰动范数的比值 比1小，那就证明不能继续添加扰动，则按照比例缩放扰动
            factor = min(factor, delta_norms)
            # factor = 1
            delta = delta * factor

            adv_KP = torch.clamp(KP + delta, min=0, max=1).detach()
        adv_images = reconstruct_image(adv_KP)
        return adv_images


class Limited_CW2(Attack):
    """
    Uses Adam to minimize the CW L2 objective function

    Paper link: https://arxiv.org/abs/1608.04644
    """

    def __init__(self, model, c=1, kappa=0, steps=200, lr=0.01, binary_search=1, pixel_k=1):
        super().__init__("Limited_CW2", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.pixel_k = pixel_k
        self.binary_search_steps = binary_search
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        # 本方法还是主要讨论无定向攻击
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        img = images.clone().detach()
        pre_label = labels.clone().detach()

        A, KP, RP = pixel_selector_by_attribution(self.model, img, labels, self.pixel_k)

        def reconstruct_image(adv_KP):
            adv_images = (A.mm(adv_KP) + RP).reshape(original_shape)
            return adv_images

        # 定向
        if self._targeted:
            pass
        # 无定向攻击
        else:
            # 攻击目标标签 必须使用one hot编码
            tlab = F.one_hot(pre_label, num_classes=10)

        # c的初始化边界
        upper_bound = self.c
        lower_bound = 0
        c = self.c
        o_c = -1

        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        o_bestattack = images.clone().detach()
        original_shape = images.shape

        adv_KP = KP.clone().detach()
        adv_KP[adv_KP == 0.0] = 1e-5
        adv_KP[adv_KP == 1.0] = 1. - 1e-5

        # 把原始图像转换成图像数据和扰动的形态
        w = self.box2inf(adv_KP.detach())
        w.requires_grad = True
        # 定义优化器 仅优化 w
        optimizer = torch.optim.Adam([w], lr=self.lr)

        for outer_step in range(self.binary_search_steps):
            # inner_attack_success = False
            for iteration in range(1, self.steps + 1):
                optimizer.zero_grad()
                # 定义新输入
                new_img = reconstruct_image(self.inf2box(w))
                output = self.model(new_img)
                # 定义cw中的损失函数
                loss2 = torch.linalg.norm((new_img.view(1, -1) - img.view(1, -1)), ord=2, dim=1).sum()
                # 对原始类别的信息分数
                real = output[0][pre_label[0]]
                # 除了正确类别的信息分数之外的最大信息分数
                other = torch.max((1 - tlab[0]) * output)
                if self._targeted:
                    pass
                    # loss1 = other - real + k
                else:
                    # 无目标攻击的置信度损失函数，让正确类别的信息分数下降，让其他类别的信息分数上涨
                    # 其实原本的loss函数应该是 求最大值, 但是pytorch 默认优化最小值, 因此我们应当把损失函数写成求最小值的形式
                    loss1 = -other + real + self.kappa
                    # loss1=other-real+k
                loss1 = torch.clamp(loss1, min=0)
                # 这里的的 c 相当于调节系数
                loss1 = self.c * loss1
                loss = loss1 + loss2

                loss.backward(retain_graph=True)
                optimizer.step()

                l2 = loss2.detach()
                # 输出的是概率
                # pro=F.softmax(self._model(new_img),dim=1)[0].data.cpu().numpy()[target_label]
                # 获取模型对样本的预测类别
                # torch.argmax 取得最大值的下标索引
                pred = torch.argmax(output, dim=1)
                """
                注意 pred 还是会是一个一维的tensor
                """

                # if iteration % (self.steps // 10) == 0:
                # print("iteration={} loss={} loss1={} loss2={} pred={}".format(iteration,loss,loss1,loss2,pred))
                # logging.info(
                #     "iteration={} loss={} loss1={} loss2={} pred={}".format(iteration, loss, loss1, loss2,
                #                                                             pred))

                if self._targeted:
                    pass
                    # if (l2 < o_bestl2) and (np.argmax(sc) == target_label):
                    #     # print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                    #     print("attack success l2={} target_label={}".format(l2, target_label))
                    #     o_bestl2 = l2
                    #     o_bestscore = pred
                    #     o_bestattack = new_img.data.cpu().numpy()
                else:
                    # 攻击成功则更新 l2 范数，和 adv_img
                    # 因为每一轮迭代都能找到比上一次较好的值，
                    # 因此只要本次迭代是成功的，那就认为本次迭代得到的结果是比较好的
                    if (l2 < o_bestl2) and (pred[0] != pre_label[0]):
                        # print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                        # print("attack success l2={} label={}".format(l2,pred))
                        # logging.info("attack success l2={} label={}".format(l2, pred))
                        # inner_attack_success=True
                        o_bestl2 = l2
                        o_bestscore = pred
                        o_bestattack = reconstruct_image(self.inf2box(w.detach()))

            # -------------外循环判定-----------
            if self._targeted:
                pass
                # if (o_bestscore == target_label) and (o_bestscore != -1):
                #     # 攻击成功 减小c
                #     upper_bound = min(upper_bound, c)
                #     if upper_bound < 1e9:
                #         print()
                #         o_c = c
                #         c = (lower_bound + upper_bound) / 2
                # else:
                #     lower_bound = max(lower_bound, c)
                #     o_c = c
                #     if upper_bound < 1e9:
                #         c = (lower_bound + upper_bound) / 2
                #     else:
                #         c *= 10

            else:
                # 这里有一个不合理的地方：只要有一次内循环攻击成功，超参数c就一直在变小.
                # 那我们就要进行二分查找，重新迭代
                if (o_bestscore != pre_label) and (o_bestscore != -1):
                    # 攻击成功 减小c
                    upper_bound = min(upper_bound, c)
                    if upper_bound < 1e9:
                        o_c = c
                        c = (lower_bound + upper_bound) / 2
                # 如果之前都没攻击成功，那我们就扩大c
                else:
                    lower_bound = max(lower_bound, c)
                    o_c = c
                    if upper_bound < 1e9:
                        c = (lower_bound + upper_bound) / 2
                    else:
                        c *= 10
                        c = max(self.c, c)

            # print("outer_step={} c {}->{}".format(outer_step,o_c,c))
            # logging.info("outer_step={} c {}->{}".format(outer_step, o_c, c))
        # print(o_bestattack)

        # if o_bestscore != -1:
        #     """
        #         If adversarial_label the target label that we are finding.
        #         The adversarial_example and adversarial_label will be accepted and
        #         True will be returned.
        #         :return: bool
        #     """
        #     if adversary.try_accept_the_example(o_bestattack, o_bestscore):
        #         return adversary
        # adv_image = reconstruct_image(self.inf2box(w.detach()))
        # o_bestattack
        return o_bestattack

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


class Limited_CW(Attack):
    r"""
    CW in the paper 'Towards Evaluating the Robustness of Neural Networks'
    [https://arxiv.org/abs/1608.04644]

    Distance Measure : L2

    Arguments:
        model (nn.Module): model to attack.
        c (float): c in the paper. parameter for box-constraint. (Default: 1e-4)
            :math:`minimize \Vert\frac{1}{2}(tanh(w)+1)-x\Vert^2_2+c\cdot f(\frac{1}{2}(tanh(w)+1))`
        kappa (float): kappa (also written as 'confidence') in the paper. (Default: 0)
            :math:`f(x')=max(max\{Z(x')_i:i\neq t\} -Z(x')_t, - \kappa)`
        steps (int): number of steps. (Default: 1000)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    .. warning:: With default c, you can't easily get adversarial images. Set higher c like 1.

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`,        `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.

    Examples::
        >>> attack = torchattacks.CW(model, c=1e-4, kappa=0, steps=1000, lr=0.01)
        >>> adv_images = attack(images, labels)

    .. note:: Binary search for c is NOT IMPLEMENTED methods in the paper due to time consuming.

    """

    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01, pixel_k=1):
        super().__init__("Limited_CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.pixel_k = pixel_k
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        r"""
        Overridden.
        """
        images_origin = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)

        if self._targeted:
            target_labels = self._get_target_label(images, labels)

        original_shape = images_origin.shape
        A, KP, RP = pixel_selector_by_attribution(self.model, images_origin, labels, self.pixel_k)

        adv_KP = KP.clone().detach()
        adv_KP[adv_KP == 0.0] = 1. / 255 * 0.1
        adv_KP[adv_KP == 1.0] = 1. - 1. / 255 * 0.1

        # adv_KP.requires_grad = True

        def reconstruct_image(adv_KP):
            adv_images = (A.mm(adv_KP) + RP).reshape(original_shape)
            return adv_images

        # w = torch.zeros_like(images).detach() # Requires 2x times

        best_adv_images = images.clone().detach()

        best_L2 = 1e10 * torch.ones((len(images))).to(self.device)
        prev_cost = 1e10
        dim = len(images.shape)

        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        w = self.inverse_tanh_space(adv_KP)
        w.requires_grad = True
        optimizer = optim.Adam([w], lr=self.lr)
        adv_images = reconstruct_image(self.tanh_space(w))

        for step in range(self.steps):
            # Get adversarial images

            # Calculate loss
            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = self.model(adv_images)

            if self._targeted:
                f_loss = self.f(outputs, target_labels).sum()
            else:
                f_loss = -self.f(outputs, labels).sum()

            cost = L2_loss + self.c * f_loss

            optimizer.zero_grad()
            cost.backward(retain_graph=True)
            # print(w.grad_fn)
            optimizer.step()

            adv_images = reconstruct_image(self.tanh_space(w))
            # is_same = (adv_images==images_origin)

            # print('step', step)
            # Update adversarial images
            _, pre = torch.max(outputs.detach(), 1)
            correct = (pre == labels).float()
            if correct[0] == False:
                return adv_images
            # filter out images that get either correct predictions or non-decreasing loss,
            # i.e., only images that are both misclassified and loss-decreasing are left
            # mask = (1 - correct) * (best_L2 > current_L2.detach())
            # best_L2 = mask * current_L2.detach() + (1 - mask) * best_L2
            #
            # mask = mask.view([-1] + [1] * (dim - 1))
            # best_adv_images = mask * adv_images.detach() + (1 - mask) * best_adv_images

            # Early stop when loss does not converge.
            # max(.,1) To prevent MODULO BY ZERO error in the next step.
            if step % max(self.steps // 10, 1) == 0:
                if cost.item() > prev_cost:
                    return adv_images
                prev_cost = cost.item()
        is_same = (adv_images == images_origin)
        x = 1
        return adv_images

    def tanh_space(self, x):
        return 1 / 2 * (torch.tanh(x) + 1)

    def inverse_tanh_space(self, x):
        # torch.atanh is only for torch >= 1.7.0
        return self.atanh(x * 2 - 1)

    def atanh(self, x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    # f-function in the paper
    def f(self, outputs, labels):
        one_hot_labels = torch.eye(len(outputs[0]))[labels].to(self.device)

        i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
        j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit

        if self._targeted:
            return torch.clamp((i - j), min=-self.kappa)
        else:
            return torch.clamp((j - i), min=-self.kappa)
