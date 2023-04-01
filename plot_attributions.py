# coding = UTF-8
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy
from pylab import mpl
import torch.nn.functional as F
from attack_models import load_model_args, load_dataset
from pixel_selector import pixel_attribution_sort
import pickle
from matplotlib import ticker
from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz


# 相似性范围从-1到1：
# -1意味着两个向量指向的方向正好截然相反，1表示它们的指向是完全相同的，
# 0通常表示它们之间是独立的，而在这之间的值则表示中间的相似性或相异性。

def show_one_image(subplot, images, title, color):
    # C*H*W-->H*W*C
    c, h, w = images[0].shape
    image = numpy.transpose(images[0].cpu().detach().numpy(), (1, 2, 0))
    if c == 1:
        subplot.imshow(image, 'gray')
    else:
        subplot.imshow(images)
    # subplot.axis('off')  # 关掉坐标轴为 off
    # 显示坐标轴但是无刻度
    subplot.set_xticks([])
    subplot.set_yticks([])
    # 设定图片边框粗细
    subplot.spines['top'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['bottom'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['left'].set_linewidth('2.0')  # 设置边框线宽为2.0
    subplot.spines['right'].set_linewidth('2.0')  # 设置边框线宽为2.0
    # 设定边框颜色
    subplot.spines['top'].set_color(color)
    subplot.spines['bottom'].set_color(color)
    subplot.spines['left'].set_color(color)
    subplot.spines['right'].set_color(color)
    # subplot.set_title(title, y=-0.25, color=color, fontsize=8)  # 图像题目


def attribute_image_features(net, algorithm, input, label, **kwargs):
    net.zero_grad()
    tensor_attributions = algorithm.attribute(input,
                                              target=label,
                                              **kwargs
                                              )

    return tensor_attributions


def plot_attributions(net, image, label):
    saliency = Saliency(net)
    NT = NoiseTunnel(saliency)
    IG = IntegratedGradients(net)
    s1 = attribute_image_features(net, saliency, image, label)  # labels[0].item()
    s2 = attribute_image_features(net, NT, image, label, nt_type='smoothgrad', nt_samples=100, stdevs=0.2)
    s3 = attribute_image_features(net, IG, image, label, baselines=images * 0, )
    grads = np.transpose(s1.squeeze().cpu().detach().numpy(), (1, 2, 0))


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['Times New Roman']
    mpl.rcParams['mathtext.fontset'] = 'stix'
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
    mpl.rcParams['figure.dpi'] = 200  # 分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    # mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.05, 0.99
    # mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.07, 0.99
    # mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1005, 0.1005
    # plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    # plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    # mpl.rcParams['figure.constrained_layout.use'] = True

    model_name_set = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121']
    #
    # select_a_sample_to_plot('MNIST',
    #                         'ResNet18_ImageNet',
    #                         Epsilon=5 / 255,
    #                         Iterations=10,
    #                         Momentum=1.0)

    # select_a_sample_to_plot(
    #     'CIFAR10',
    #     'Res20_CIFAR10'
    # )
    test_loader, _, _ = load_dataset('ImageNet', batch_size=1, is_shuffle=False)

    model, model_acc = load_model_args('Res34_ImageNet')
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
    # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
    for data in test_loader:
        original_images, original_labels = data
        original_images = original_images.to(device)
        original_labels = original_labels.to(device)
        _, predict = torch.max(F.softmax(model(original_images), dim=1), 1)
        # 选择预测正确的original_images和original_labels，剔除预测不正确的original_images和original_labels
        # predict_answer为一维向量，大小为batch_size
        predict_answer = (original_labels == predict)
        # torch.nonzero会返回一个二维矩阵，大小为（nozero的个数）*（1）
        no_zero_predict_answer = torch.nonzero(predict_answer)
        # 我们要确保 predict_correct_index 是一个一维向量,因此使用flatten,其中的元素内容为下标
        predict_correct_index = torch.flatten(no_zero_predict_answer)
        # print('predict_correct_index', predict_correct_index)
        images = torch.index_select(original_images, 0, predict_correct_index)
        labels = torch.index_select(original_labels, 0, predict_correct_index)
        if images.shape[0] > 0:
            print('correct prediction')
            plot_attributions()
            break
        else:
            print('wrong prediction')

    # select_a_sample_to_plot(
    #     'ImageNet',
    #     'ResNet34_ImageNet'
    # )

    print("----ALL WORK HAVE BEEN DONE!!!----")

'''
    # sharex 和 sharey 表示坐标轴的属性是否相同，可选的参数：True，False，row，col，默认值均为False，表示画布中的四个ax是相互独立的；
    # True 表示所有子图的x轴（或者y轴）标签是相同的，
    # row 表示每一行之间的子图的x轴（或者y轴）标签是相同的（不同行的子图的轴标签可以不同），
    # col表示每一列之间的子图的x轴（或者y轴）标签是相同的（不同列的子图的轴标签可以不同）
'''
