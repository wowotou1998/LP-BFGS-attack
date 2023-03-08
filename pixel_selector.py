import torch
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)


def tanh_space(x):
    return 1 / 2 * (torch.tanh(x) + 1)


def inf2box(x):
    return 0.5 * (torch.tanh(x) + 1)


def inverse_tanh_space(x):
    # torch.atanh is only for torch >= 1.7.0
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    return atanh(x * 2 - 1)


def box2inf(x):
    return torch.atanh(2.0 * x - 1.)


def pixel_attribution_sort(model, images, labels, pixel_k, FIND_MAX=True):
    # Only output the
    n = images.numel()
    k = pixel_k
    if images.shape[0] > 1:
        raise Exception('the batch size of images must be 1')
    if k > n:
        print('pixel_k: ', pixel_k, 'images.numel() ', n)
        raise Exception('the pixel_k is more than the number of pixels in images')
    baseline = torch.zeros_like(images)
    ig = IntegratedGradients(model)
    # attributions 表明每一个贡献点对最终决策（正确的标签）的重要性，正值代表正贡献， 负值代表负贡献，绝对值越大则像素点的值对最终决策的印象程度越高
    attributions, delta = ig.attribute(images.detach().clone(), baseline,
                                       target=labels[0].item(),
                                       return_convergence_delta=True)

    attributions_abs = torch.abs(attributions)
    attributions_abs_flat = attributions_abs.flatten()
    if FIND_MAX:
        v, idx = attributions_abs_flat.sort(descending=True)
    else:
        v, idx = attributions_abs_flat.sort(descending=False)
    idx = idx[0:k]
    return idx, attributions_abs


def pixel_saliency_sort(model, images, labels, pixel_k, num_class=10):
    # Only output the
    attributions_abs = torch.zeros_like(images, device=images.device)
    # num_class = 10
    n = images.numel()
    k = pixel_k
    if k > n:
        raise Exception('the pixel_k is more than the number of pixels in images')
    baseline = torch.zeros_like(images)
    ig = IntegratedGradients(model)
    # attributions 表明每一个贡献点对最终决策（正确的标签）的重要性，正值代表正贡献， 负值代表负贡献，绝对值越大则像素点的值对最终决策的印象程度越高
    for i in range(num_class):
        attributions, delta = ig.attribute(images.detach().clone(), baseline,
                                           target=i,
                                           return_convergence_delta=True)
        attributions_abs += torch.abs(attributions)

    attributions_abs_flat = attributions_abs.flatten()
    v, idx = attributions_abs_flat.sort(descending=True)
    idx = idx[0:k]
    return idx, attributions_abs


def pixel_selector_by_attribution(model, images, labels, pixel_k):
    """
    Inputs
        model,
        images X: denote a image as a vector X with length n
        pixel_k: set the number of major attribution pixel
    Outputs
        the matrix A: we denote the major attributions K Pixels as a vector KP with length k,
        so we can get a matrix A that satisfies  X = A*KP, where the size of A is n*k, the size of X is n*1,
        and then we can get the KP whose size is k*1.
        we denote the Rest unchanged Pixels as RP, RP = X-A*KP
    Steps
        1. we must decide the number of the pixel that will be modified
        2. according to the attributions value to decide the position in which pixel will be changed.
            2.1 so fist the pixel position must be in conjunction with the attributions value.
            2.2 find the top-k pixel position where pixels have higher attributions value.
    """
    n = images.numel()
    k = pixel_k
    if k > n:
        raise Exception('the pixel_k is more than the number of pixels in images')
    if images.shape[0] > 1:
        raise Exception('the batch size of images must be 1')

    A = torch.zeros(size=(n, k), device=images.device, dtype=torch.float)
    # KP = torch.zeros(k, device=images.device, dtype=torch.float)
    # 找到矩阵A, 满足 image = A*KP+RP, A:n*k; KP:k*1; C:n*1
    idx, attributions_abs = pixel_attribution_sort(model, images, labels, pixel_k)
    # idx, attributions_abs = pixel_saliency_sort(model, images, labels, pixel_k, num_class=10)

    KP = images.detach().clone().flatten()[idx].view(-1, 1)

    for i in range(k):
        # 第 idx[i] 行第 i列 的元素置为 1
        # idx保存了对最终决策有重要作用的像素点的下标，
        A[idx[i].item()][i] = 1
    A_KP = A.mm(KP)
    RP = images.detach().clone().flatten().view(-1, 1) - A_KP

    # -------- plot --------
    # import matplotlib.pyplot as plt
    # shape = images.shape
    # attr_min, attr_max = attributions_abs.min().item(), attributions_abs.max().item()
    # attributions_abs_img = (attributions_abs - attr_min) / (
    #         attr_max - attr_min)
    #
    # fig, axes = plt.subplots(1, 4, figsize=(2 * 4, 2))
    # for i in range(4):
    #     axes[i].set_xticks([])
    #     axes[i].set_yticks([])
    #
    # image = images[0].cpu().detach().numpy().transpose(1, 2, 0)
    # axes[0].imshow(image, cmap='gray')
    # axes[0].set_title('origin')
    #
    # image = attributions_abs_img[0].cpu().detach().numpy().transpose(1, 2, 0)
    # axes[1].imshow(image, cmap='coolwarm')
    # axes[1].set_title('attribution heatmap')
    # # add color bar
    # s_cmap_std = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=attr_min, vmax=attr_max))
    # fig.colorbar(s_cmap_std, ax=axes[1], ticks=[attr_min, 0.5 * (attr_max - attr_min), attr_max])
    #
    # image = A_KP.reshape(shape)[0].cpu().detach().numpy().transpose(1, 2, 0)
    # axes[2].imshow(image, cmap='gray')
    # axes[2].set_title('important k pixels')
    #
    # image = RP.reshape(shape)[0].cpu().detach().numpy().transpose(1, 2, 0)
    # axes[2].imshow(image, cmap='gray')
    # axes[2].set_title('the rest pixels')
    # plt.show()
    # fig.savefig('pixel_selecor.pdf')
    # -------- plot --------

    # A is a matrix, size is n*k
    # KP is a matrix, size is k*1
    # RP is a matrix, size is n*1
    return idx, A, KP, RP
