import torch
import torch.nn as nn
from torchattacks.attack import Attack
from pixel_selector import select_major_contribution_pixels
import torch.optim as optim
import torch.nn.functional as F


# adv_images = (A.mm(B_box) + RP).reshape(original_shape)

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
        >>> attack = torchattacks.FGSM(model, eps=0.007)
        >>> adv_images = attack(images, labels)

    """

    def __init__(self, model, eps=0.007, sample_rate=0.1):
        super().__init__("Limited_FGSM", model)
        self.eps = eps
        self.sample_rate = sample_rate
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
        A, KP, RP = select_major_contribution_pixels(self.model, images_origin, labels, rate=self.sample_rate)
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
                 alpha=2 / 255, steps=40, sample_rate=0.1, random_start=True):
        super().__init__("Limited_PGD", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.sample_rate = sample_rate
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
        A, KP, RP = select_major_contribution_pixels(self.model, images_origin, labels, self.sample_rate)

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
                 sample_rate=0.1):
        super().__init__("Limited_PGDL2", model)
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.eps_for_division = eps_for_division
        self.sample_rate = sample_rate
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
        A, KP, RP = select_major_contribution_pixels(self.model, images_origin, labels, self.sample_rate)

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

    def __init__(self, model, c=1, kappa=0, steps=1000, lr=0.01, binary_search=5, sample_rate=0.1):
        super().__init__("Limited_CW2", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.sample_rate = sample_rate
        self.binary_search_steps = binary_search
        self._supported_mode = ['default', 'targeted']

    def forward(self, images, labels):
        # 本方法还是主要讨论无定向攻击

        img = images.clone().detach()
        pre_label = labels.clone().detach()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 定向
        if self._targeted:
            pass
        # 无定向攻击
        else:
            # 攻击目标标签 必须使用one hot编码
            tlab = F.one_hot(pre_label, num_classes=10)
            # Variable(torch.from_numpy(np.eye(num_labels)[pre_label]).to(device).float())
        # boxmin, boxmax = self.model.bounds()
        # print("boxmin={}, boxmax={}".format(boxmin, boxmax))
        # logging.info("boxmin={}, boxmax={}".format(boxmin, boxmax))
        # print(tlab)

        # c的初始化边界
        lower_bound = 0
        c = self.c
        upper_bound = 1e10
        # the best l2, score, and image attack
        o_bestl2 = 1e10
        o_bestscore = -1
        # o_bestattack = None
        # the resulting image, tanh'd to keep bounded from boxmin to boxmax
        original_shape = images.shape
        A, KP, RP = select_major_contribution_pixels(self.model, img, labels, self.sample_rate)

        adv_KP = KP.clone().detach()
        adv_KP[adv_KP == 0.0] = 1e-5
        adv_KP[adv_KP == 1.0] = 1. - 1e-5

        def reconstruct_image(adv_KP):
            adv_images = (A.mm(adv_KP) + RP).reshape(original_shape)
            return adv_images

        # 把原始图像转换成图像数据和扰动的形态
        w = self.box2inf(adv_KP.detach())
        w.requires_grad = True
        # 定义优化器 仅优化 w
        optimizer = torch.optim.Adam([w], lr=self.lr)

        for outer_step in range(self.binary_search_steps):
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
                    # 更新l2范数，和 adv_img
                    if (l2 < o_bestl2) and (pred != pre_label):
                        # print("attack success l2={} target_label={} pro={}".format(l2,target_label,pro))
                        # print("attack success l2={} label={}".format(l2,pred))
                        # logging.info("attack success l2={} label={}".format(l2, pred))
                        o_bestl2 = l2
                        o_bestscore = pred
                        # o_bestattack = new_img.data.cpu().numpy()

            confidence_old = -1

            if self._targeted:
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

        # if o_bestscore != -1:
        #     """
        #         If adversarial_label the target label that we are finding.
        #         The adversarial_example and adversarial_label will be accepted and
        #         True will be returned.
        #         :return: bool
        #     """
        #     if adversary.try_accept_the_example(o_bestattack, o_bestscore):
        #         return adversary
        adv_image = reconstruct_image(self.inf2box(w.detach()))
        return adv_image

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

    def __init__(self, model, c=1e-4, kappa=0, steps=1000, lr=0.01, sample_rate=0.1):
        super().__init__("Limited_CW", model)
        self.c = c
        self.kappa = kappa
        self.steps = steps
        self.lr = lr
        self.sample_rate = sample_rate
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
        A, KP, RP = select_major_contribution_pixels(self.model, images_origin, labels, self.sample_rate)

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
