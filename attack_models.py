import math

import torch
import torchattacks
import torchvision.models
import torchvision.models as models
from captum.attr import IntegratedGradients
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import save_model_results
from MNIST_models import lenet5, FC_256_128
import matplotlib.pyplot as plt
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn.functional as F
from minimize import minimize
from pixel_selector import inf2box, box2inf
from pixel_selector import pixel_selector_by_attribution
from attack_method_self_defined import Limited_FGSM, Limited_PGD, Limited_CW3, JSMA
from torchattacks import SparseFool
from prefetch_generator import BackgroundGenerator
# from advertorch.attacks import JSMA

import pickle

import numpy as np

import torch
import torch.nn as nn
import time


# from advertorch.attacks import LBFGSAttack


def show_one_image(images, title):
    plt.figure()
    print(images.shape)
    images = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(images.detach().numpy()[0].shape)
    plt.imshow(images)
    plt.title(title)
    plt.show()


def show_images(images, titles, ):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # plt.figure()
    print(images.shape)
    B = images.shape[0]
    C = images.shape[1]

    # image = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.title(title)
    # plt.show()

    fig, axes = plt.subplots(1, B, figsize=(2 * B, 2))
    for i in range(B):
        image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        if C == 1:
            axes[i].imshow(image, cmap='gray')
        else:
            axes[i].imshow(image)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(titles[i])
    plt.show(block=True)
    # fig.savefig('pixel_selecor.pdf')


def random_target(y_original, num_classes):
    import random
    """
    Draw random target labels all of which are different and not the original label.
    Args:
        original_label(int): Original label.
    Return:
        target_labels(list): random target labels
    """
    y_adv = torch.zeros_like(y_original, device=y_original.device, dtype=torch.long)
    batch = y_original.numel()
    # sample(list, k) 返回一个长度为k新列表，新列表存放list所产生k个随机唯一的元素
    for i in range(batch):
        # 随机标签列表
        y_o_i = y_original[i].item()
        label_set = list(range(num_classes))
        label_set_2 = label_set[0:y_o_i] + label_set[y_o_i + 1:num_classes]
        y_adv_i = random.sample(label_set_2, 1)
        y_adv[i] = y_adv_i[0]

    return y_adv


# def generate_attack_method(attack_name, model, epsilon, Iterations, Momentum):
#     if attack_name == 'FGSM':
#         atk = torchattacks.FGSM(model, eps=epsilon)
#     elif attack_name == 'I_FGSM':
#         atk = torchattacks.BIM(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations, )
#     elif attack_name == 'PGD':
#         atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations,
#                                random_start=True)
#     elif attack_name == 'MI_FGSM':
#         atk = torchattacks.MIFGSM(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations, decay=Momentum)
#     else:
#         atk = None
#     return atk


def load_model_args(model_name):
    import os
    assert os.path.isdir('./Checkpoint'), 'Error: no checkpoint directory found!'
    if model_name == 'LeNet5':
        model = lenet5()
    elif model_name == 'FC_256_128':
        model = FC_256_128()
    # ------ CIFAR10-------
    elif model_name == 'Res20_CIFAR10':
        # resnet20_cifar10
        model = ptcv_get_model("resnet20_cifar10", pretrained=True, root='./Checkpoint')
        return model, 88.
    # --------SVHN--------
    elif model_name == 'Res20_SVHN':
        model = ptcv_get_model("resnet20_svhn", pretrained=True, root='./Checkpoint')
        return model, 88.
    # --------ImageNet--------
    elif model_name == 'VGG16_ImageNet':
        model = ptcv_get_model("vgg16", pretrained=True, root='../Checkpoint')
        return model, 76.130
    elif model_name == 'VGG19_ImageNet':
        model = ptcv_get_model("vgg19", pretrained=True, root='../Checkpoint')
        # model = torchvision.models.vgg19(pretrained=True, )
        return model, 76.130
    elif model_name == 'ResNet34_ImageNet':
        model = ptcv_get_model("resnet34", pretrained=True, root='../Checkpoint')
        return model, 100 - 26.94
    elif model_name == 'ResNet18_ImageNet':
        model = ptcv_get_model("resnet18", pretrained=True, root='../Checkpoint')
        return model, 76.130
    elif model_name == 'ResNet101_ImageNet':
        model = ptcv_get_model("resnet101", pretrained=True, root='../Checkpoint')
        return model, 76.130
    elif model_name == 'DenseNet121_ImageNet':
        model = ptcv_get_model("densenet121", pretrained=True, root='../Checkpoint')
        # model = torchvision.models.densenet121(pretrained=True)
        return model, 100 - 21.91
    else:
        raise Exception('Unknown model!!!')
    check_point = torch.load('./Checkpoint/%s.pth' % (model_name), map_location='cuda:0')
    model.load_state_dict(check_point['model'])
    print(model_name, 'has been load！', check_point['test_acc'])
    return model, check_point['test_acc']


def load_dataset(dataset, batch_size, is_shuffle=False, pin=True):
    class DataLoaderX(DataLoader):

        def __iter__(self):
            return BackgroundGenerator(super().__iter__())

    data_tf = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    data_tf_imagenet = transforms.Compose(
        [

            transforms.Resize(size=(224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485, 0.456, 0.406],
            #                      std=[0.229, 0.224, 0.225])

        ]
    )
    test_dataset = None
    if dataset == 'MNIST':
        test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, transform=data_tf, download=True)
        test_dataset_size = 10000
        num_classes = 10

    elif dataset == 'ImageNet':
        test_dataset = datasets.ImageNet(root='./DataSet/ImageNet', split='val', transform=data_tf_imagenet)
        test_dataset_size = 50000
        num_classes = 1000

    elif dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, transform=data_tf, download=True)
        test_dataset_size = 10000
        num_classes = 10

    elif dataset == 'SVHN':
        test_dataset = datasets.SVHN(root='./DataSet/SVHN', split='test', transform=data_tf, download=True)
        test_dataset_size = 10000
        num_classes = 10
    else:
        raise Exception('Unknown dataset')
    test_loader = DataLoaderX(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=2)

    return test_loader, test_dataset_size, num_classes


import time


def torch_time_ticker(func):
    def inner(*args, **kwargs):
        torch.cuda.synchronize()
        start_time = time.time()
        res = func(*args, **kwargs)
        torch.cuda.synchronize()
        end_time = time.time()
        result = end_time - start_time
        return res, result

    return inner


# @torch_time_ticker
# def fun():
#     return 2

# print(fun())


def attack_by_k_pixels(attack_name, model, images, num_classes, labels, eps, trade_off_c, A, KP, RP):
    if attack_name == 'limited_FGSM':
        atk = Limited_FGSM(model, A, KP, RP, eps=eps)

        torch.cuda.synchronize()
        start = time.perf_counter()
        adv_images = atk(images, labels)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return adv_images, end - start

    if attack_name == 'limited_PGD':
        atk = Limited_PGD(model, A, KP, RP, eps=eps, alpha=(1.5 * eps) / 200, steps=200)
        torch.cuda.synchronize()
        start = time.perf_counter()
        adv_images = atk(images, labels)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return adv_images, end - start
    if attack_name == 'SparseFool':
        atk = SparseFool(model, steps=200, lam=KP.numel() * (4. / 100.))

        torch.cuda.synchronize()
        start = time.perf_counter()
        adv_images = atk(images, labels)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return adv_images, end - start

    if attack_name == 'JSMA':
        atk = JSMA(model, num_classes,
                   theta=1., gamma=1.0 * KP.numel() / RP.numel(),
                   clip_min=0.0, clip_max=1.0,
                   )

        torch.cuda.synchronize()
        start = time.perf_counter()
        adv_images = atk(images, random_target(labels, num_classes=10))
        torch.cuda.synchronize()
        end = time.perf_counter()

        return adv_images, end - start

    if attack_name == 'limited_CW':
        atk = Limited_CW3(model, A, KP, RP, c=trade_off_c, steps=200)
        # atk = torchattacks.CW(model, c=1)
        torch.cuda.synchronize()
        start = time.perf_counter()
        adv_images = atk(images, labels)
        torch.cuda.synchronize()
        end = time.perf_counter()

        return adv_images, end - start
    if 'BFGS' in attack_name:

        x0 = images.detach().clone()[0:1]
        labels = labels[0:1]
        original_shape = x0.shape
        A, KP_box, C0 = A.detach().clone(), KP.detach().clone(), RP.detach().clone()
        KP_box_min = 1e-3  # math.atanh(-1. + 10./255)
        KP_box_max = 1 - 1e-3  # math.atanh(1. - 10./255)

        KP_box[KP_box == 0.0] = 1e-3
        KP_box[KP_box == 1.0] = 1 - 1e-3
        w = box2inf(KP_box)

        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        # B_inf_min = -1e4  # math.atanh(-1. + 10./255)
        # B_inf_max = +1e4  # math.atanh(1. - 10./255)

        def ce_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            c = trade_off_c
            # 这个函数这么多printf是为了防止nan出现
            # 虽然 B_inf 的取值范围是-∞到+∞，但还是要对 B_inf 的最大值最小值做限制，不然的话就会出现 Nan
            # B_inf = torch.clamp(B_inf, min=B_inf_min, max=B_inf_max)
            # print('B_inf', B_inf)
            KP_box = inf2box(B_inf)
            KP_box = torch.clamp(KP_box, min=KP_box_min, max=KP_box_max)
            # print('KP_box', KP_box)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()
            # print('adv_images',adv_images)
            outputs = model(adv_images)
            # print('outputs',outputs)
            # print('labels',labels)

            cost = L2_loss - c * CELoss(outputs, labels)
            return cost

        def cw_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = trade_off_c
            KP_box = inf2box(B_inf)
            KP_box = torch.clamp(KP_box, min=KP_box_min, max=KP_box_max)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            labels = labels.cpu()
            # print(labels.device)
            # print(outputs.device)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
            out1 = torch.clamp((j - i), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def cw_log_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            labels = labels.cpu()
            c = trade_off_c
            KP_box = inf2box(B_inf)
            KP_box = torch.clamp(KP_box, min=KP_box_min, max=KP_box_max)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()
            outputs = F.softmax(model(adv_images), dim=1)
            # outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs,
                             dim=1)  # get the second largest logit 其实这里得到的是非正确标签中的最大概率，i
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit, 其实这里是得到正确标签对应的概率，j
            # 对于无目标攻击， 我们总是希望真标签对应的概率较小，而不是正确的标签的概率较大， 即 (i-j)越大越好， (j-i)越小越好
            out1 = torch.clamp((torch.log(j) - torch.log(i)), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        if attack_name == 'limited_BFGS_CW':
            torch.cuda.synchronize()
            start = time.perf_counter()

            res1 = minimize(cw_loss, w.detach().clone(), method='bfgs', max_iter=200, tol=1e-5, disp=False)
            KP_box = inf2box(res1.x)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)
            # res1 = minimize(ce_loss, w.detach().clone(), method='bfgs', max_iter=200, tol=1e-5, disp=False)
            # res1 = minimize(cw_log_loss, w.detach().clone(), method='newton-exact',
            #                 # options={'handle_npd': 'cauchy'},
            #                 max_iter=10, tol=1e-5,
            #                 disp=False)
            # print('res1', res1)
            # adversary = LBFGSAttack(predict=model, initial_const=100000, num_classes=10)
            # adv_image = adversary.perturb(images.detach().clone(), y=labels.detach().clone())
            # print(torch.sum((adv_image == images)))
            torch.cuda.synchronize()
            end = time.perf_counter()

            return adv_images, end - start
        elif attack_name == 'limited_BFGS_CW_LOG':
            torch.cuda.synchronize()
            start = time.perf_counter()

            res1 = minimize(cw_log_loss, w.detach().clone(), method='bfgs', max_iter=200, tol=1e-5, disp=False)
            KP_box = inf2box(res1.x)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            torch.cuda.synchronize()
            end = time.perf_counter()

            return adv_images, end - start
        elif attack_name == 'limited_BFGS_CE':
            torch.cuda.synchronize()
            start = time.perf_counter()

            res1 = minimize(ce_loss, w.detach().clone(), method='bfgs', max_iter=200, tol=1e-4, disp=False)
            KP_box = inf2box(res1.x)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            torch.cuda.synchronize()
            end = time.perf_counter()

            return adv_images, end - start
        else:
            raise Exception('unknown BFGS attack')

    raise Exception('Unknown attack method')


def attack_one_model(model, test_loader, test_loader_size, num_classes, attack_set, N, eps, trade_off_c, pixel_k,
                     attri_method,
                     SHOW=False):
    cifar_label = {0: "airplane", 1: "car", 2: "bird", 3: "cat", 4: "deer",
                   5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"}
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    sample_num = 0.
    epoch_num = 0

    acc_num_before_attack = 0.

    attack_success_num = torch.zeros(len(attack_set), dtype=torch.float, device=device)

    time_total = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    confidence_total = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm0_total = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm1_total = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm2_total = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm_inf_total = torch.zeros(len(attack_set), dtype=torch.float, device=device)

    time_ave = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    confidence_ave = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm0_ave = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm1_ave = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm2_ave = torch.zeros(len(attack_set), dtype=torch.float, device=device)
    noise_norm_inf_ave = torch.zeros(len(attack_set), dtype=torch.float, device=device)

    # every epoch has 64 images ,every image has 1 channel and the channel size is 28*28
    pbar = tqdm(total=test_loader_size)
    model.to(device)
    model.eval()

    for data in test_loader:
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad_(True)

        batch_num = labels.shape[0]
        epoch_num += 1
        sample_num += batch_num
        pbar.update(batch_num)
        # if epoch_num < 299:
        #     continue

        # if batch_num > 10000:
        #     break
        # labels.requires_grad_(True)

        _, predict = torch.max(model(images), 1)
        # 选择预测正确的images和labels，剔除预测不正确的images和labels
        predict_answer = (labels == predict)
        # print('predict_answer', predict_answer)
        # 我们要确保predict correct是一个一维向量,因此使用flatten
        predict_correct_index = torch.flatten(torch.nonzero(predict_answer))
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(images, 0, predict_correct_index)
        labels = torch.index_select(labels, 0, predict_correct_index)

        # ------------------------------

        if min(images.shape) == 0:
            # print('\nNo images correctly classified in this batch')
            # 为了保证不越界，全部分类不正确时要及时退出，避免下面的计算
            continue

        # pixel_idx, attribution_abs = pixel_attribution_sort(model, images, labels, pixel_k)
        # pixel_idx, attribution_abs = pixel_saliency_sort(model, images, labels, pixel_k)
        pixel_idx, A, KP, RP = pixel_selector_by_attribution(model, images, labels, pixel_k, attri_method)
        acc_num_before_attack += predict_answer.sum().item()
        # 统计神经网络分类正确的样本的个数总和
        # valid_attack_num += labels.shape[0]
        if SHOW:
            plot_images = images.detach().clone()
            plot_titles = ['original: ' + str(labels[0].item())]
        for idx, attack_i in enumerate(attack_set):
            images_under_attack, time_i = attack_by_k_pixels(attack_i, model, images, num_classes, labels, eps,
                                                             trade_off_c, A, KP, RP)
            b = images_under_attack.shape[0]
            time_i = torch.as_tensor([time_i] * b, device=device).view(b, -1)
            confidence, predict = torch.max(F.softmax(model(images_under_attack), dim=1), dim=1)

            noise = images_under_attack.detach().clone().view(images.shape[0], -1) - \
                    images.detach().clone().view(images.shape[0], -1)
            if attack_i not in ['SparseFool', 'JSMA']:
                noise = torch.index_select(noise, 1, pixel_idx)
            noise_norm0 = torch.linalg.norm(noise, ord=0, dim=1)
            noise_norm1 = torch.linalg.norm(noise, ord=1, dim=1)
            noise_norm2 = torch.linalg.norm(noise, ord=2, dim=1)
            noise_norm_inf = torch.linalg.norm(noise, ord=float('inf'), dim=1)

            # 记录每一个攻击方法在每一个批次的攻击成功个数
            attack_success_num[idx] += (labels != predict).sum().item()
            # 记录误分类置信度
            # 攻击成功的对抗样本的置信度
            # 选择攻击成功的images的confidences
            # misclassification = ()
            # print('predict_answer', predict_answer)
            # 我们要确保 predict correct 是一个一维向量,因此使用 flatten
            attack_success_index = torch.flatten(torch.nonzero(labels != predict))
            # print('predict_correct_index', predict_correct_index)
            valid_time = torch.index_select(time_i, 0, attack_success_index)
            valid_confidence = torch.index_select(confidence, 0, attack_success_index)
            valid_noise_norm0 = torch.index_select(noise_norm0, 0, attack_success_index)
            valid_noise_norm1 = torch.index_select(noise_norm1, 0, attack_success_index)
            valid_noise_norm2 = torch.index_select(noise_norm2, 0, attack_success_index)
            valid_noise_norm_inf = torch.index_select(noise_norm_inf, 0, attack_success_index)

            time_total[idx] = valid_time.sum().item()
            confidence_total[idx] += valid_confidence.sum().item()
            noise_norm0_total[idx] += valid_noise_norm0.sum().item()
            noise_norm1_total[idx] += valid_noise_norm1.sum().item()
            noise_norm2_total[idx] += valid_noise_norm2.sum().item()
            noise_norm_inf_total[idx] += valid_noise_norm_inf.sum().item()
            if SHOW:
                plot_images = torch.cat([plot_images, images_under_attack.clone().detach()], dim=0)
                plot_titles += [attack_i + ': ' + str(predict[0].item())]
            if acc_num_before_attack == 1:
                pass

                # -------- plot attribution score--------
                # shape = images.shape
                # A = torch.zeros(size=(images.numel(), pixel_k), device=images.device, dtype=torch.float)
                # # KP = torch.zeros(k, device=images.device, dtype=torch.float)
                # # 找到矩阵A, 满足 image = A*KP+RP, A:n*k; KP:k*1; C:n*1
                # idx, attributions_abs = (model, images, labels, pixel_k)
                # attr_min, attr_max = attributions_abs.min().item(), attributions_abs.max().item()
                # attributions_abs_img = (attributions_abs - attr_min) / \
                #                        (attr_max - attr_min)
                #
                # KP = images.detach().clone().flatten()[idx].view(-1, 1)
                #
                # for i in range(pixel_k):
                #     # 第 idx[i] 行第 i列 的元素置为 1
                #     # idx保存了对最终决策有重要作用的像素点的下标，
                #     A[idx[i].item()][i] = 1
                # A_KP = A.mm(KP)
                # RP = images.detach().clone().flatten().view(-1, 1) - A_KP
                #
                # fig, axes = plt.subplots(1, 6, figsize=(2 * 6, 2))
                # for i in range(6):
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
                # A_KP_img_0 = A_KP.reshape(shape)[0].cpu().detach().numpy().transpose(1, 2, 0)
                # axes[2].imshow(A_KP_img_0, cmap='gray')
                # axes[2].set_title('important k pixels')
                #
                # RP_img = RP.reshape(shape)
                # RP_img_0 = RP_img[0].cpu().detach().numpy().transpose(1, 2, 0)
                # axes[3].imshow(RP_img_0, cmap='gray')
                # axes[3].set_title('the rest pixels')
                #
                # adv_KP = images_under_attack-RP_img
                # image = adv_KP[0].cpu().detach().numpy().transpose(1, 2, 0)
                # axes[4].imshow(image, cmap='gray')
                # axes[4].set_title('adv k pixels')
                #
                #
                # image = images_under_attack[0].cpu().detach().numpy().transpose(1, 2, 0)
                # axes[5].imshow(image, cmap='gray')
                # axes[5].set_title('attacked image')
                # plt.show(block=True)
                # fig.savefig('pixel_selecor2.pdf')
                # -------- plot attribution score --------

            # brea
        if SHOW:
            show_images(plot_images, plot_titles)
        if epoch_num >= N:
            break
    print('attack_success_num', attack_success_num)
    # print('confidence_total', confidence_total)
    attack_success_rate = (attack_success_num / acc_num_before_attack) * 100
    # attack_success_num[attack_success_num == 0] = float('inf'), 防止出现除 0 溢出 inf
    no_zero_index = attack_success_num != 0
    time_ave[no_zero_index] = (time_total[no_zero_index] / attack_success_num[no_zero_index]) * 1000.
    confidence_ave[no_zero_index] = (confidence_total[no_zero_index] / attack_success_num[no_zero_index])
    noise_norm0_ave[no_zero_index] = (noise_norm0_total[no_zero_index] / attack_success_num[no_zero_index])
    noise_norm1_ave[no_zero_index] = (noise_norm1_total[no_zero_index] / attack_success_num[no_zero_index])
    noise_norm2_ave[no_zero_index] = (noise_norm2_total[no_zero_index] / attack_success_num[no_zero_index])
    noise_norm_inf_ave[no_zero_index] = (noise_norm_inf_total[no_zero_index] / attack_success_num[no_zero_index])
    pbar.close()
    print('model acc %.2f' % (acc_num_before_attack / sample_num))
    for i in range(len(attack_set)):
        print(
            'eps=%.2f, pixel_k=%d, %s ASR=%.2f%%,time=%.2f(ms) confidence=%.2f, norm(0)=%.2f,norm(1)=%.2f,norm(2)=%.2f, norm(inf)=%.2f' % (
                eps, pixel_k,
                attack_set[i],
                attack_success_rate[i],
                time_ave[i],
                confidence_ave[i],
                noise_norm0_ave[i],
                noise_norm1_ave[i],
                noise_norm2_ave[i],
                noise_norm_inf_ave[i]))
    return attack_success_rate, time_ave, confidence_ave, noise_norm0_ave, noise_norm1_ave, noise_norm2_ave, noise_norm_inf_ave


def attack_many_model(job_name, dataset, model_name_set, attack_N, attack_set, batch_size, eps_set, trade_off_c,
                      pixel_k_set, attri_method_set):
    import datetime
    # datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    res_data = [
        ['attri_method_set', 'dataset', 'mode_name', 'attack_method', 'attack_num', 'constant c', 'eps_i', 'pixel_k',
         'attack_success', 'confidence', 'noise_norm0', 'noise_norm1', 'noise_norm2',
         'noise_norm_inf', 'time(ms)']]
    for attri_i in attri_method_set:
        for set_i, dataset_i in enumerate(dataset):
            test_loader, test_dataset_size, num_classes = load_dataset(dataset_i, batch_size, is_shuffle=True)
            for mode_name in model_name_set[set_i]:
                model, model_acc = load_model_args(mode_name)
                for eps_i in eps_set:
                    for pixel_k in pixel_k_set:
                        success_rate_list, time_list, confidence_list, noise_norm0_list, noise_norm1_list, noise_norm2_list, noise_norm_inf_list \
                            = attack_one_model(
                            model=model,
                            test_loader=test_loader,
                            test_loader_size=test_dataset_size,
                            num_classes=num_classes,
                            attack_set=attack_set,
                            N=attack_N,
                            eps=eps_i,
                            trade_off_c=trade_off_c,
                            pixel_k=pixel_k,
                            attri_method=attri_i
                        )
                        success_rate, time, confidence, norm0, norm1, norm2, norm_inf = success_rate_list.cpu().numpy().tolist(), \
                            time_list.cpu().numpy().tolist(), \
                            confidence_list.cpu().numpy().tolist(), \
                            noise_norm0_list.cpu().numpy().tolist(), \
                            noise_norm1_list.cpu().numpy().tolist(), \
                            noise_norm2_list.cpu().numpy().tolist(), \
                            noise_norm_inf_list.cpu().numpy().tolist()
                        for i in range(len(success_rate_list)):
                            res_data.append(
                                [attri_i, dataset_i, mode_name, attack_set[i], attack_N, trade_off_c, eps_i, pixel_k,
                                 success_rate[i], confidence[i], norm0[i], norm1[i], norm2[i], norm_inf[i], time[i]])

            current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            with open('./Checkpoint/%s_%s_%s.pkl' % (job_name, dataset_i, current_time), 'wb') as f:
                pickle.dump(res_data, f)
            import pandas as pd
            csv = pd.DataFrame(columns=res_data[0], data=res_data[1:])
            csv.to_csv('./Checkpoint/%s_%s_%s.csv' % (job_name, dataset_i, current_time))
            print(csv)

    # with open('%s.pkl' % ('pkl'), 'rb') as f:
    #     basic_info = pickle.load(f)


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    from pylab import mpl
    import random

    # matplotlib.use('agg')
    # matplotlib.get_backend()
    # mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    # mpl.rcParams['font.sans-serif'] = ['Arial']
    # mpl.rcParams['backend'] = 'agg'
    # mpl.rcParams["font.size"] = 12
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    # if torch.cuda.is_available():
    #     torch.backends.cudnn.benchmark = True

    # 生成随机数，以便固定后续随机数，方便复现代码
    random.seed(123)
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(123)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(123)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(123)

    batch_size = 1
    attack_N = 1
    # pixel_k_set = [20]
    # pixel_k_set = [5, 10, 15]
    # pixel_k_set = [10]
    # 'FGSM', 'I_FGSM', 'PGD', 'MI_FGSM', 'Adam_FGSM','Adam_FGSM_incomplete'
    mnist_model_name_set = ['FC_256_128']  # 'LeNet5', 'FC_256_128'
    cifar10_model_name_set = ['Res20_CIFAR10', ]  # 'VGG19', 'ResNet34', 'ResNet101', 'DenseNet121'
    svhn_model_name_set = ['Res20_SVHN']
    # imagenet_model_name_set = ['ResNet18_ImageNet']
    imagenet_model_name_set = [
        # 'VGG19_ImageNet',
        'ResNet34_ImageNet',
    ]
    # 'DenseNet161_ImageNet','ResNet34_ImageNet', 'DenseNet121_ImageNet VGG19_ImageNet

    # job_name = 'cifar_%d_diff_loss_20pixel_1e3' % attack_N

    # job_name = 'mni_cifar_iter200_%d_100acc_20pixel_1e3' % attack_N
    job_name = 'random2_iter200_%d_100acc_20pixel_1e3' % attack_N
    from torch import autograd

    # with autograd.detect_anomaly():
    # with torch.autograd.profiler.profile(enabled=True) as prof:
    attack_many_model(job_name,
                      [
                          # 'MNIST',
                          'CIFAR10',
                          # 'ImageNet'
                      ],

                      [
                          # mnist_model_name_set,
                          cifar10_model_name_set,
                          # imagenet_model_name_set
                      ],

                      attack_N=1000,
                      attack_set=[
                          'limited_BFGS_CW',
                          'limited_BFGS_CE',
                          'limited_BFGS_CW_LOG',
                          'limited_FGSM',
                          'limited_CW',
                          # 'SparseFool',
                          # 'JSMA'
                      ],
                      batch_size=1,
                      eps_set=[1.0],
                      trade_off_c=1e3,
                      # pixel_k_set=[40]
                      pixel_k_set=[
                          # 20,
                          40,
                          # 60,
                          # 80,
                          100,
                          # 200,
                          # 300,
                          400,
                          # 700,
                          1000
                      ],
                      attri_method_set=['IG', 'DeepLIFT', 'Random']
                      )
    # print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    # attack_many_model(job_name,
    #                   [
    #                       # 'MNIST',
    #                       'CIFAR10',
    #                       # 'ImageNet'
    #                   ],
    #
    #                   [
    #                       # mnist_model_name_set,
    #                       cifar10_model_name_set,
    #                       # imagenet_model_name_set
    #                   ],
    #
    #                   attack_N=1000,
    #                   attack_set=[
    #                       # 'limited_BFGS_CW',
    #                       'limited_BFGS_CE',
    #                       # 'limited_BFGS_CW_LOG',
    #                       # 'limited_FGSM',
    #                       # 'limited_CW',
    #                       # 'SparseFool',
    #                       # 'JSMA'
    #                   ],
    #                   batch_size=1,
    #                   eps_set=[1.0],
    #                   trade_off_c=1e3,
    #                   # pixel_k_set=[40]
    #                   pixel_k_set=[
    #                       # 20,
    #                       # 40, 60, 80,
    #                       100
    #                   ]
    #                   )

    # job_name = 'imagenet_%d_100acc_20pixel_1e3' % attack_N
    # attack_many_model(job_name,
    #                   ['ImageNet'],
    #                   # [cifar10_model_name_set, svhn_model_name_set],
    #                   [imagenet_model_name_set],
    #                   attack_N=1000,
    #                   attack_set=[
    #                       'limited_BFGS_CW',
    #                       'limited_BFGS_CE',
    #                       'limited_BFGS_CW_LOG',
    #                       'limited_FGSM',
    #                       'limited_CW',
    #                       # 'SparseFool'
    #                   ],
    #                   batch_size=1,
    #                   eps_set=[1.0],
    #                   trade_off_c=1e3,
    #                   pixel_k_set=[5, 10, 15, 20]
    #                   # pixel_k_set=[20]
    #                   )

    # attack_many_model(job_name,
    #                   # ['MNIST', 'CIFAR10', 'SVHN'],
    #                   ['CIFAR10'],
    #                   # [mnist_model_name_set, cifar10_model_name_set, svhn_model_name_set],
    #                   [cifar10_model_name_set],
    #                   attack_N,
    #                   attack_set,
    #                   batch_size=1,
    #                   eps_set=[1.0],
    #                   trade_off_c=1e3,
    #                   pixel_k_set=[20, 40, 60, 80, 100]
    #                   )

    # attack_many_model('CIFAR10',
    #                   cifar10_model_name_set,
    #                   attack_N,
    #                   attack_set,
    #                   batch_size,
    #                   eps_set=[1.0],
    #                   pixel_k_set=pixel_k_set
    #                   )

    print("ALL WORK HAVE BEEN DONE!!!")
