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
                            select_major_contribution_pixels)
from attack_method_self_defined import Limted_FGSM, Limited_PGD
# prepare your pytorch model as "model"
# prepare a batch of data and label as "cln_data" and "true_label"
# ...

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


def show_two_image(images, titles, cmaps=None):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # plt.figure()
    print(images.shape)
    N = images.shape[0]

    # image = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.title(title)
    # plt.show()

    fig, axes = plt.subplots(1, N, figsize=(2 * N, 2))
    for i in range(N):
        image = images[i].cpu().detach().numpy().transpose(1, 2, 0)
        axes[i].imshow(image, cmap='gray' if cmaps is None else cmaps[i])
        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].set_title(titles[i])
    plt.show()


def generate_attack_method(attack_name, model, epsilon, Iterations, Momentum):
    if attack_name == 'FGSM':
        atk = torchattacks.FGSM(model, eps=epsilon)
    elif attack_name == 'I_FGSM':
        atk = torchattacks.BIM(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations, )
    elif attack_name == 'PGD':
        atk = torchattacks.PGD(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations,
                               random_start=True)
    elif attack_name == 'MI_FGSM':
        atk = torchattacks.MIFGSM(model, eps=epsilon, alpha=epsilon / Iterations, steps=Iterations, decay=Momentum)
    else:
        atk = None
    return atk


def load_model_args(model_name):
    assert os.path.isdir('./Checkpoint'), 'Error: no checkpoint directory found!'

    if model_name == 'LeNet5':
        model = lenet5()
    elif model_name == 'FC_256_128':
        model = FC_256_128()
    elif model_name == 'VGG19':
        model = models.vgg19(num_classes=10)
    elif model_name == 'ResNet50':
        model = models.resnet50(num_classes=10)
    elif model_name == 'ResNet101':
        model = models.resnet101(num_classes=10)
    elif model_name == 'DenseNet121':
        model = models.densenet121(num_classes=10)

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
        test_dataset = datasets.MNIST(root='../DataSet/MNIST', train=False, transform=data_tf, download=True)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_dataset_size = 10000
    elif dataset == 'ImageNet':
        test_dataset = datasets.ImageNet(root='../DataSet/ImageNet', split='val', transform=data_tf_imagenet)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_dataset_size = 50000
    elif dataset == 'CIFAR10':
        test_dataset = datasets.CIFAR10(root='../DataSet/CIFAR10', train=False, transform=data_tf)
        test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=is_shuffle)
        test_dataset_size = 10000
    else:
        test_loader, test_dataset_size = None, 0
    return test_loader, test_dataset_size





def generate_adv_images_by_k_pixels(attack_name, model, images, labels, options):
    if attack_name == 'limited_FGSM':
        atk = Limted_FGSM(model, eps=0.1)
        adv_images = atk(images, labels)
        return adv_images
    if attack_name == 'limited_PGD':
        atk = Limited_PGD(model, eps=0.9)
        adv_images = atk(images, labels)
        return adv_images
    if attack_name == 'limited-second-order':
        x0 = images.detach().clone()[0:1]
        labels = labels[0:1]
        original_shape = x0.shape
        A, KP_box, C0 = select_major_contribution_pixels(model, x0, labels, )

        KP_box[KP_box == 0.0] = 1. / 255 * 0.1
        KP_box[KP_box == 1.0] = 1. - 1. / 255 * 0.1
        w = box2inf(KP_box)

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


def attack_one_model(model, test_loader, test_loader_size, attack_method_set, options):
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    sample_num = 0.
    epoch_num = 0

    acc_num_before_attack = 0.

    attack_success_num = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)

    confidence_total = torch.zeros(len(attack_method_set), dtype=torch.float, device=device)
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
            continue

        acc_num_before_attack += predict_answer.sum().item()
        # 统计神经网络分类正确的样本的个数总和
        # valid_attack_num += labels.shape[0]

        for idx, attack_i in enumerate(attack_method_set):
            images_under_attack = generate_adv_images_by_k_pixels(attack_i, model, images, labels, options)

            confidence, predict = torch.max(F.softmax(model(images_under_attack), dim=1), dim=1)
            noise = images_under_attack.detach().clone().view(images.shape[0], -1) - \
                    images.detach().clone().view(images.shape[0], -1)
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

            predict_incorrect_index = torch.flatten(torch.nonzero(labels != predict))
            # print('predict_correct_index', predict_correct_index)
            valid_confidence = torch.index_select(confidence, 0, predict_incorrect_index)
            valid_noise_norm2 = torch.index_select(noise_norm2, 0, predict_incorrect_index)
            valid_noise_norm_inf = torch.index_select(noise_norm_inf, 0, predict_incorrect_index)

            confidence_total[idx] += valid_confidence.sum().item()
            noise_norm2_total[idx] += valid_noise_norm2.sum().item()
            noise_norm_inf_total[idx] += valid_noise_norm_inf.sum().item()

            if epoch_num == 1:
                print('predict_correct_element_num: ', predict_correct_index.nelement())
                titles_1 = (str(labels[0].item()), str(predict[0].item()))
                # show_one_image(images, 'image_after_' + attack_i)
                show_two_image(torch.cat([images, images_under_attack], dim=0), titles_1)

                # ------ IntegratedGradient ------
                # select_major_contribution_pixels(model, images, labels, rate=1.0 / 28)
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

        if attack_success_num > 100:
            break
    print(attack_success_num)
    attack_success_rate = (attack_success_num / acc_num_before_attack) * 100
    # attack_success_num[attack_success_num == 0] = float('inf'), 防止出现除 0 溢出 inf
    confidence_ave = (confidence_total / attack_success_num)
    noise_norm2_ave = (noise_norm2_total / attack_success_num)
    noise_norm_inf_ave = (noise_norm_inf_total / attack_success_num)

    for i in range(len(attack_method_set)):
        print('%s ASR = %.2f%%, confidence %.2f, norm(2) = %.2f, norm(inf) = %.2f' % (
        attack_method_set[i],
        attack_success_rate[i],
        confidence_ave[i],
        noise_norm2_ave[i],
        noise_norm_inf_ave))

    pbar.close()
    print('model acc %.2f' % (acc_num_before_attack / sample_num))
    return attack_success_rate, confidence_ave, noise_norm2_ave


def attack_many_model(dataset, model_name_set, attack_method_set, batch_size, work_name, option_set):
    test_loader, test_dataset_size = load_dataset(dataset, batch_size, is_shuffle=True)
    for mode_name in model_name_set:
        model, model_acc = load_model_args(mode_name)
        for option_i in option_set:
            attack_method_succ_list, confidence_list, perturbation_list = attack_one_model(model=model,
                                                                                           test_loader=test_loader,
                                                                                           test_loader_size=test_dataset_size,
                                                                                           attack_method_set=attack_method_set,
                                                                                           options=option_i)


if __name__ == '__main__':
    import os

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    from pylab import mpl

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

    batch_size = 1
    attack_method_set = [
        'limited-second-order',
        # 'limited_FGSM'
        # 'limited_PGD'
    ]  # 'FGSM', 'I_FGSM', 'PGD', 'MI_FGSM', 'Adam_FGSM','Adam_FGSM_incomplete'
    mnist_model_name_set = ['FC_256_128']  # 'LeNet5', 'FC_256_128'
    # cifar10_model_name_set = ['VGG16', ]  # 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121'
    # imagenet_model_name_set = ['ResNet50_ImageNet']
    # 'DenseNet161_ImageNet','ResNet50_ImageNet', 'DenseNet121_ImageNet VGG19_ImageNet

    attack_many_model('MNIST',
                      mnist_model_name_set, attack_method_set,
                      batch_size,
                      work_name='second-order-attack',
                      option_set=[None])
    #
    # attack_many_model('CIFAR10',
    #                   cifar10_model_name_set, attack_method_set,
    #                   batch_size,
    #                   work_name='adam_fgsm_cifar',
    #                   Epsilon_set=[5],
    #                   Iterations_set=[10],
    #                   Momentum=1.0)

    # attack_many_model('ImageNet',
    #                   imagenet_model_name_set, attack_method_set,
    #                   batch_size,
    #                   work_name='adam_fgsm_imagenet',
    #                   Epsilon_set=[3],
    #                   Iterations_set=[10],
    #                   Momentum=1.0)

    # VGG16 pao yi xia
    # attack_many_model('ImageNet',
    #                   imagenet_model_name_set, attack_method_set,
    #                   batch_size,
    #                   work_name='imagenet_iteration_compare',
    #                   Epsilon_set=[5],
    #                   Iterations_set=[1, 4, 8, 12, 16, 20],
    #                   Momentum=1.0)
    #
    # attack_many_model(model_name_set, attack_method_set,
    #                   batch_size,
    #                   work_name='iterations_compare',
    #                   Epsilon_set=[5],
    #                   Iterations_set=[1],
    #                   Momentum=0.9)
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
        A, KP_box, C0 = select_major_contribution_pixels(model, x0, labels, rate=10. / 28)

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
