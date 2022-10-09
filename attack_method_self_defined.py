import torch
import torch.nn as nn
from torchattacks.attack import Attack
from pixel_selector import select_major_contribution_pixels


# adv_images = (A.mm(B_box) + RP).reshape(original_shape)

# 这里主要定义两个基准函数 FGSM，PGD 在图像的可更改像素比较少的情况下进行对抗样本的生成的效果是怎样的
class Limted_FGSM(Attack):
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
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.007):
        super().__init__("Limted_FGSM", model)
        self.eps = eps
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
        A, KP, RP = select_major_contribution_pixels(self.model, images_origin, labels)
        KP_origin = KP.detach().clone()
        KP = KP.requires_grad_(True)

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

    def __init__(self, model, eps=0.3,
                 alpha=2 / 255, steps=40, random_start=True):
        super().__init__("Limited_PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
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
        A, KP, RP = select_major_contribution_pixels(self.model, images_origin, labels)

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
