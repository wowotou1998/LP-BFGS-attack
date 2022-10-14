import torch
import torchattacks
import torchvision.models
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from utils import save_model_results
from MNIST_models import lenet5, FC_256_128
import matplotlib.pyplot as plt
from pytorchcv.model_provider import get_model as ptcv_get_model
import torch.nn.functional as F
from minimize import minimize
from pixel_selector import (inverse_tanh_space, tanh_space,
                            inf2box, box2inf,
                            select_major_contribution_pixels, major_contribution_pixels_idx)
from attack_method_self_defined import Limited_FGSM, Limited_PGD, Limited_PGDL2, Limited_CW, Limited_CW2
# prepare your pytorch model as "model"
# prepare a batch of data and label as "cln_data" and "true_label"
# ...
import pickle

import numpy as np

import torch
import torch.nn as nn

from advertorch.attacks import LBFGSAttack


def show_one_image(images, title):
    plt.figure()
    print(images.shape)
    images = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(images.detach().numpy()[0].shape)
    plt.imshow(images)
    plt.title(title)
    plt.show()


def show_two_image(images, titles, ):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # plt.figure()
    print(images.shape)
    N = images.shape[0]
    C = images.shape[1]

    # image = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.title(title)
    # plt.show()

    fig, axes = plt.subplots(1, N, figsize=(2 * N, 2))
    for i in range(N):
        image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        if C == 1:
            axes[i].imshow(image, cmap='gray')
        else:
            axes[i].imshow(image)
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(titles[i])
    plt.show()


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
    assert os.path.isdir('./Checkpoint'), 'Error: no checkpoint directory found!'
    if model_name == 'LeNet5':
        model = lenet5()
    elif model_name == 'FC_256_128':
        model = FC_256_128()
    #     ------ CIFAR10-------
    elif model_name == 'Res20_CIFAR10':
        # resnet20_cifar10
        model = ptcv_get_model("resnet20_cifar10", pretrained=True, root='./Checkpoint')
        return model, 88.
    elif model_name == 'VGG16_ImageNet':
        model = ptcv_get_model("vgg16", pretrained=True, root='../Checkpoint')
        return model, 76.130
    elif model_name == 'VGG19_ImageNet':
        model = ptcv_get_model("vgg19", pretrained=True, root='../Checkpoint')
        # model = torchvision.models.vgg19(pretrained=True, )
        return model, 76.130
    elif model_name == 'ResNet18_ImageNet':
        model = ptcv_get_model("resnet18", pretrained=True, root='../Checkpoint')
        return model, 100 - 26.94
    elif model_name == 'ResNet50_ImageNet':
        model = ptcv_get_model("resnet50", pretrained=True, root='../Checkpoint')
        return model, 76.130
    elif model_name == 'ResNet101_ImageNet':
        model = ptcv_get_model("resnet101", pretrained=True, root='../Checkpoint')
        return model, 76.130
    elif model_name == 'DenseNet121_ImageNet':
        model = ptcv_get_model("densenet121", pretrained=True, root='../Checkpoint')
        # model = torchvision.models.densenet121(pretrained=True)
        return model, 100 - 21.91
    else:
        model = models.vgg16(num_classes=10)
    check_point = torch.load('./Checkpoint/%s.pth' % (model_name), map_location='cuda:0')
    model.load_state_dict(check_point['model'])
    print(model_name, 'has been load！', check_point['test_acc'])
    return model, check_point['test_acc']


def load_dataset(dataset, batch_size, is_shuffle=False):
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

    if dataset == 'MNIST':
        test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, transform=data_tf, download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_dataset_size = 10000
    elif dataset == 'ImageNet':
        test_dataset = datasets.ImageNet(root='../DataSet/ImageNet', split='val', transform=data_tf_imagenet)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_dataset_size = 50000
    elif dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, transform=data_tf, download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_dataset_size = 10000
    else:
        test_loader, test_dataset_size = None, 0
    return test_loader, test_dataset_size


def generate_adv_images_by_k_pixels(attack_name, model, images, labels, eps, pixel_k):
    if attack_name == 'limited_FGSM':
        atk = Limited_FGSM(model, eps=eps, pixel_k=pixel_k)
        adv_images = atk(images, labels)
        return adv_images
    if attack_name == 'limited_PGD':
        atk = Limited_PGD(model, eps=eps, alpha=(1.5 * eps) / 200, steps=200, pixel_k=pixel_k)
        adv_images = atk(images, labels)
        return adv_images
    if attack_name == 'limited_PGDL2':
        atk = Limited_PGDL2(model, eps=eps, alpha=0.27, steps=20, pixel_k=pixel_k)
        adv_images = atk(images, labels)
        return adv_images
    if attack_name == 'limited_CW':
        atk = Limited_CW2(model, c=1e5, pixel_k=pixel_k)
        # atk = torchattacks.CW(model, c=1)
        adv_images = atk(images, labels)
        return adv_images
    if attack_name == 'limited_BFGS':
        x0 = images.detach().clone()[0:1]
        labels = labels[0:1]
        original_shape = x0.shape
        A, KP_box, C0 = select_major_contribution_pixels(model, x0, labels, pixel_k)

        KP_box[KP_box == 0.0] = 1e-4
        KP_box[KP_box == 1.0] = 1 - 1e-4
        w = box2inf(KP_box)

        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        def cw_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 1e5
            KP_box = inf2box(B_inf)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
            out1 = torch.clamp((j - i), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def cw_log_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 1e5
            KP_box = inf2box(B_inf)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs,
                             dim=1)  # get the second largest logit 其实这里得到的是非正确标签中的最大概率，i
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit, 其实这里是得到正确标签对应的概率，j
            # 对于无目标攻击， 我们总是希望真标签对应的概率较小，而不是正确的标签的概率较大， 即 (i-j)越大越好， (j-i)越小越好
            out1 = torch.clamp((torch.log(j) - torch.log(i)), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def pure_lbfgs_attack_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            c = 1e5
            KP_box = inf2box(B_inf)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)

            cost = L2_loss + c * CELoss(outputs, labels)
            return -cost

        # res1 = minimize(cw_loss, w.detach().clone(), method='bfgs', max_iter=200, tol=1e-5, disp=False)
        res1 = minimize(pure_lbfgs_attack_loss, w.detach().clone(), method='bfgs', max_iter=200, tol=1e-5, disp=False)
        # res1 = minimize(cw_log_loss, w.detach().clone(), method='newton-exact',
        #                 # options={'handle_npd': 'cauchy'},
        #                 max_iter=10, tol=1e-5,
        #                 disp=False)
        # print('res1', res1)

        KP_box = inf2box(res1.x)
        adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

        # adversary = LBFGSAttack(predict=model, initial_const=100000, num_classes=10)
        # adv_image = adversary.perturb(images.detach().clone(), y=labels.detach().clone())
        # print(torch.sum((adv_image == images)))
        return adv_images
    raise RuntimeError('Unknown attack method')


def attack_one_model(model, test_loader, test_loader_size, attack_method_set, N, eps, pixel_k):
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    sample_num = 0.
    epoch_num = 0

    acc_num_before_attack = 0.

    attack_success_num = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)

    confidence_total = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)
    noise_norm1_total = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)
    noise_norm2_total = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)
    noise_norm_inf_total = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)

    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
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
            print('\nNo images correctly classified in this batch')
            # 为了保证不越界，全部分类不正确时要及时退出，避免下面的计算
            continue

        pixel_idx = major_contribution_pixels_idx(model, images, labels, pixel_k)

        acc_num_before_attack += predict_answer.sum().item()
        # 统计神经网络分类正确的样本的个数总和
        # valid_attack_num += labels.shape[0]

        for idx, attack_i in enumerate(attack_method_set):

            images_under_attack = generate_adv_images_by_k_pixels(attack_i, model, images, labels, eps, pixel_k)

            confidence, predict = torch.max(F.softmax(model(images_under_attack), dim=1), dim=1)
            noise = images_under_attack.detach().clone().view(images.shape[0], -1) - \
                    images.detach().clone().view(images.shape[0], -1)
            noise = torch.index_select(noise, 1, pixel_idx)
            noise_norm1 = torch.linalg.norm(noise, ord=2, dim=1)
            noise_norm2 = torch.linalg.norm(noise, ord=2, dim=1)
            noise_norm_inf = torch.linalg.norm(noise, ord=float('inf'), dim=1)

            # 记录每一个攻击方法在每一批次的攻击成功个数
            attack_success_num[idx] += (labels != predict).sum().item()
            # 记录误分类置信度
            # 攻击成功的对抗样本的置信度
            # 选择攻击成功的images的confidences
            # misclassification = ()
            # print('predict_answer', predict_answer)
            # 我们要确保 predict correct 是一个一维向量,因此使用 flatten
            selector = labels != predict
            attack_success_index = torch.flatten(torch.nonzero(labels != predict))
            # print('predict_correct_index', predict_correct_index)
            valid_confidence = torch.index_select(confidence, 0, attack_success_index)
            valid_noise_norm1 = torch.index_select(noise_norm1, 0, attack_success_index)
            valid_noise_norm2 = torch.index_select(noise_norm2, 0, attack_success_index)
            # if valid_noise_norm2==
            valid_noise_norm_inf = torch.index_select(noise_norm_inf, 0, attack_success_index)
            a = valid_confidence.sum().item()
            confidence_total[idx] += valid_confidence.sum().item()
            noise_norm1_total[idx] += valid_noise_norm1.sum().item()
            noise_norm2_total[idx] += valid_noise_norm2.sum().item()
            noise_norm_inf_total[idx] += valid_noise_norm_inf.sum().item()

            if epoch_num == 1:
                print('predict_correct_element_num: ', predict_correct_index.nelement())
                titles_1 = (str(labels[0].item()), str(predict[0].item()))
                # show_one_image(images, 'image_after_' + attack_i)
                # show_two_image(torch.cat([images, images_under_attack], dim=0), titles_1)

                # ------ IntegratedGradient ------
                # select_major_contribution_pixels(model, images, labels, pixel_k=1)
                # baseline = torch.zeros_like(images)
                # ig = IntegratedGradients(model)
                # titles_2 = (str(labels[0].item()), str(predict[0].item()), 'attributions')
                # # attributions 表明每一个贡献点对最终决策的重要性，正值代表正贡献， 负值代表负贡献，绝对值越大则像素点的值对最终决策的印象程度越高
                # attributions, delta = ig.attribute(images, baseline, target=labels[0].item(),
                #                                    return_convergence_delta=True)
                # attributions = torch.abs(attributions)
                # attributions = (attributions - torch.min(attributions)) / (
                #         torch.max(attributions) - torch.min(attributions))
                # show_two_image(torch.cat([images, images_under_attack, attributions], dim=0), titles_2)
                # print('IG Attributions:', attributions)
                # print('Convergence Delta:', delta)

                # break

        if acc_num_before_attack > N:
            break
    print(attack_success_num)
    attack_success_rate = (attack_success_num / acc_num_before_attack) * 100
    # attack_success_num[attack_success_num == 0] = float('inf'), 防止出现除 0 溢出 inf
    confidence_ave = (confidence_total / attack_success_num)
    noise_norm1_ave = (noise_norm1_total / attack_success_num)
    noise_norm2_ave = (noise_norm2_total / attack_success_num)
    noise_norm_inf_ave = (noise_norm_inf_total / attack_success_num)

    for i in range(len(attack_method_set)):
        print('eps=%.2f, pixel_k=%d, %s ASR=%.2f%%, confidence=%.2f, norm(1)=%.2f,norm(2)=%.2f, norm(inf)=%.2f' % (
            eps, pixel_k,
            attack_method_set[i],
            attack_success_rate[i],
            confidence_ave[i],
            noise_norm1_ave[i],
            noise_norm2_ave[i],
            noise_norm_inf_ave[i]))

    pbar.close()
    print('model acc %.2f' % (acc_num_before_attack / sample_num))
    return attack_success_rate, confidence_ave, noise_norm1_ave, noise_norm2_ave, noise_norm_inf_ave


def attack_many_model(dataset, model_name_set, attack_N, attack_method_set, batch_size, eps_set, pixel_k_set):
    import datetime
    # datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    test_loader, test_dataset_size = load_dataset(dataset, batch_size, is_shuffle=True)
    res_data = [['dataset', 'mode_name', 'attack_method', 'attack_num', 'eps_i', 'pixel_k',
                 'attack_success', 'confidence', 'noise_norm1', 'noise_norm2', 'noise_norm_inf']]

    for mode_name in model_name_set:
        model, model_acc = load_model_args(mode_name)
        for eps_i in eps_set:
            for pixel_k in pixel_k_set:
                attack_success_list, confidence_list, noise_norm1_list, noise_norm2_list, noise_norm_inf_list = attack_one_model(
                    model=model,
                    test_loader=test_loader,
                    test_loader_size=test_dataset_size,
                    attack_method_set=attack_method_set,
                    N=attack_N,
                    eps=eps_i,
                    pixel_k=pixel_k)
                success_rate, confidence, norm1, norm2, norm_inf = attack_success_list.cpu().numpy().tolist(), \
                                                                   confidence_list.cpu().numpy().tolist(), \
                                                                   noise_norm1_list.cpu().numpy().tolist(), \
                                                                   noise_norm2_list.cpu().numpy().tolist(), \
                                                                   noise_norm_inf_list.cpu().numpy().tolist()
                for i in range(len(attack_success_list)):
                    res_data.append([dataset, mode_name, attack_method_set[i], attack_N, eps_i, pixel_k,
                                     success_rate[i], confidence[i], norm1[i], norm2[i], norm_inf[i]])
    with open('./Checkpoint/%s_%s.pkl' % ('data', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")), 'wb') as f:
        pickle.dump(res_data, f)
    import pandas as pd
    test = pd.DataFrame(columns=res_data[0], data=res_data[1:])
    test.to_csv('./Checkpoint/%s_%s.csv' % ('data', datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")))
    print(test)

    # with open('%s.pkl' % ('pkl'), 'rb') as f:
    #     basic_info = pickle.load(f)


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from pylab import mpl
    import random

    # matplotlib.use('agg')
    # matplotlib.get_backend()
    # mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    # mpl.rcParams['font.sans-serif'] = ['Arial']
    mpl.rcParams['backend'] = 'agg'
    # mpl.rcParams["font.size"] = 12
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # mpl.rcParams['savefig.dpi'] = 400  # 保存图片分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 生成随机数，以便固定后续随机数，方便复现代码
    random.seed(123)
    # 没有使用GPU的时候设置的固定生成的随机数
    np.random.seed(123)
    # 为CPU设置种子用于生成随机数，以使得结果是确定的
    torch.manual_seed(123)
    # torch.cuda.manual_seed()为当前GPU设置随机种子
    torch.cuda.manual_seed(123)

    batch_size = 1
    attack_N = 500
    pixel_k_set = [5, 10, 15, 20]
    # pixel_k_set = [20]
    attack_method_set = [
        'limited_BFGS',
        'limited_FGSM',
        'limited_PGD',
        'limited_CW',
    ]  # 'FGSM', 'I_FGSM', 'PGD', 'MI_FGSM', 'Adam_FGSM','Adam_FGSM_incomplete'
    mnist_model_name_set = ['FC_256_128']  # 'LeNet5', 'FC_256_128'
    cifar10_model_name_set = ['Res20_CIFAR10', ]  # 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121'
    # imagenet_model_name_set = ['ResNet50_ImageNet']
    # 'DenseNet161_ImageNet','ResNet50_ImageNet', 'DenseNet121_ImageNet VGG19_ImageNet

    attack_many_model('MNIST',
                      mnist_model_name_set,
                      attack_N,
                      attack_method_set,
                      batch_size,
                      eps_set=[1.0],
                      pixel_k_set=pixel_k_set
                      )

    attack_many_model('CIFAR10',
                      cifar10_model_name_set,
                      attack_N,
                      attack_method_set,
                      batch_size,
                      eps_set=[1.0],
                      pixel_k_set=pixel_k_set
                      )

    print("ALL WORK HAVE BEEN DONE!!!")

'''
def generate_adv_images(attack_name, model, images, labels, options):
    if attack_name == 'second-order':
        x0 = images[0:1]
        x0[x0 == 0.0] = 1. / 255 * 0.01
        x0[x0 == 1.0] = 1. - 1. / 255 * 0.01
        w = inverse_tanh_space(x0)

        # def func(image, label=labels[0].detach().clone()):
        #     logits = model(tanh_space(image))
        #     return -logits[0][label]

        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        def cw_loss(w, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 1
            adv_images = tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
            out1 = torch.clamp((j - i), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def cw_log_loss(w, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 10000
            adv_images = tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs,
                             dim=1)  # get the second largest logit 其实这里得到的是非正确标签中的最大概率，i
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit, 其实这里是得到正确标签对应的概率，j
            # 对于无目标攻击， 我们总是希望真标签对应的概率较小，而不是正确的标签的概率较大， 即 (i-j)越大越好， (j-i)越小越好
            out1 = torch.clamp((torch.log(j) - torch.log(i)), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def pure_lbfgs_attack_loss(w, labels=labels.detach().clone(), init_images=images.detach().clone()):
            c = 1
            adv_images = tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)

            cost = L2_loss + c * CELoss(outputs, labels)
            return -cost

        res1 = minimize(cw_log_loss, w.detach().clone(), method='bfgs', max_iter=100, tol=1e-5, disp=False)
        # res1 = minimize(cw_log_loss, w.detach().clone(), method='newton-exact',
        #                 # options={'handle_npd': 'cauchy'},
        #                 max_iter=10, tol=1e-5,
        #                 disp=False)
        # adv_image = tanh_space(res1.x)
        adversary = LBFGSAttack(predict=model, initial_const=100000, num_classes=10)

        adv_image = adversary.perturb(images.detach().clone(), y=labels.detach().clone())
        print(torch.sum((adv_image == images)))
        return adv_image

    if attack_name == 'limited-second-order':
        x0 = images[0:1]
        labels = labels[0:1]
        original_shape = x0.shape
        A, KP_box, C0 = select_major_contribution_pixels(model, x0, labels, pixel_k=1)

        KP_box[KP_box == 0.0] = 1. / 255 * 0.1
        KP_box[KP_box == 1.0] = 1. - 1. / 255 * 0.1
        w = box2inf(KP_box)

        # def func(image, label=labels[0].detach().clone()):
        #     logits = model(tanh_space(image))
        #     return -logits[0][label]

        CELoss = nn.CrossEntropyLoss()
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        def cw_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 1
            KP_box = inf2box(B_inf)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
            out1 = torch.clamp((j - i), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def cw_log_loss(B_inf, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 1000000
            KP_box = inf2box(B_inf)
            adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)
            one_hot_labels = torch.eye(len(outputs[0]))[labels].to(images.device)
            i, _ = torch.max((1 - one_hot_labels) * outputs,
                             dim=1)  # get the second largest logit 其实这里得到的是非正确标签中的最大概率，i
            j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit, 其实这里是得到正确标签对应的概率，j
            # 对于无目标攻击， 我们总是希望真标签对应的概率较小，而不是正确的标签的概率较大， 即 (i-j)越大越好， (j-i)越小越好
            out1 = torch.clamp((torch.log(j) - torch.log(i)), min=-kappa).sum()
            cost = L2_loss + c * out1
            return cost

        def pure_lbfgs_attack_loss(w, labels=labels.detach().clone(), init_images=images.detach().clone()):
            c = 1
            adv_images = tanh_space(w)

            current_L2 = MSELoss(Flatten(adv_images),
                                 Flatten(init_images)).sum(dim=1)
            L2_loss = current_L2.sum()

            outputs = model(adv_images)

            cost = L2_loss + c * CELoss(outputs, labels)
            return -cost

        res1 = minimize(cw_log_loss, w.detach().clone(), method='bfgs', max_iter=100, tol=1e-5, disp=False)
        # res1 = minimize(cw_log_loss, w.detach().clone(), method='newton-exact',
        #                 # options={'handle_npd': 'cauchy'},
        #                 max_iter=10, tol=1e-5,
        #                 disp=False)

        KP_box = inf2box(res1.x)
        adv_images = (A.mm(KP_box) + C0).reshape(original_shape)

        # adversary = LBFGSAttack(predict=model, initial_const=100000, num_classes=10)
        # adv_image = adversary.perturb(images.detach().clone(), y=labels.detach().clone())
        # print(torch.sum((adv_image == images)))
        return adv_images
'''
