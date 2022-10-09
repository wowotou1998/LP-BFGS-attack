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


def select_major_contribution_pixels(model, images, labels, rate=0.1):
    """
    Inputs
        model,
        images X: denote a image as a vector X with length n
        rate: set the number of major attribution pixel
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
    shape = images.shape
    n = images.numel()
    k = int(n * rate)
    A = torch.zeros(size=(n, k), device=images.device, dtype=torch.float)
    # KP = torch.zeros(k, device=images.device, dtype=torch.float)
    # 找到矩阵A, 满足 image = A*KP+RP, A:n*k; KP:k*1; C:n*1

    baseline = torch.zeros_like(images)
    ig = IntegratedGradients(model)
    # attributions 表明每一个贡献点对最终决策（正确的标签）的重要性，正值代表正贡献， 负值代表负贡献，绝对值越大则像素点的值对最终决策的印象程度越高
    attributions, delta = ig.attribute(images, baseline,
                                       target=labels[0].item(),
                                       return_convergence_delta=True)

    attributions_abs = torch.abs(attributions)
    attributions_abs_flat = attributions_abs.flatten()
    v, idx = attributions_abs_flat.sort(descending=True)
    idx = idx[0:k]

    KP = images.detach().clone().flatten()[idx].view(-1, 1)

    for i in range(k):
        # 第 idx[i] 行第 i列 的元素置为 1
        # idx保存了对最终决策有重要作用的像素点的下标，
        A[idx[i].item()][i] = 1
    A_KP = A.mm(KP)
    RP = images.detach().clone().flatten().view(-1, 1) - A_KP
    #     attributions_abs_img = (attributions_abs - attributions_abs.min()) / (
    #             attributions_abs.max() - attributions_abs.min())
    # images_show = torch.cat([images, attributions_abs_img, AKP.reshape(shape), RP.reshape(shape)], dim=0)
    # show_two_image(images_show,
    #                titles=['origin', 'attribution heatmap', 'major contribution pixels', 'the rest pixels'],
    #                cmaps=['gray', 'rainbow', 'gray', 'gray'])
    # A is a matrix, size is n*k
    # KP is a matrix, size is k*1
    # RP is a matrix, size is n*1
    return A, KP, RP
