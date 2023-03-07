# coding = UTF-8
import numpy as np
import torch
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os

import matplotlib.pyplot as plt
import numpy
from pylab import mpl
import torch.nn.functional as F
from MNIST_models import lenet5, FC_256_128
from pytorchcv.model_provider import get_model as ptcv_get_model
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from torch import nn
from attack_models import load_model_args, load_dataset
from pixel_selector import pixel_attribution_sort


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


def obtain_a_loss_value(sample, label, model):
    from torch import nn
    criterion = nn.CrossEntropyLoss()
    loss = criterion(model(sample), label).sum().item()
    return loss


def obtain_loss_matrix(u, v, sample_a, sample_b, sample_offset, label, model):
    """
    计算高度的函数
    :param x: 向量
    :param y: 向量
    :return: dim(x) * dim(y)维的矩阵
    """
    result = np.zeros_like(u)
    r, c = u.shape
    for i in range(r):
        for j in range(c):
            sample = u[i][j] * sample_a + v[i][j] * sample_b + sample_offset
            # sample = v[i][j] * sample_b + sample_offset
            # sample = u[i][j] * sample_b + sample_offset
            result[i][j] = obtain_a_loss_value(sample, label, model)
    return result


def obtain_a_predict_label(image_size, vectors, factors, model):
    n_vector = len(vectors)
    batch, channel, h, w = image_size
    sample = np.zeros((batch * channel * h * w), dtype=float)
    for i in range(n_vector):
        sample += vectors[i] * factors[i]
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    sample_ = torch.from_numpy(sample).view(batch, channel, h, w).type(torch.FloatTensor).to(device)
    _, predict_label = torch.max(F.softmax(model(sample_), dim=1), 1)
    result = predict_label[0].item()
    return result


def obtain_label_matrix(u, v, sample_a, sample_b, label, model):
    """
        计算高度的函数
        :param x: 向量
        :param y: 向量
        :return: dim(x)*dim(y)维的矩阵
    """
    from torch import nn
    criterion = nn.CrossEntropyLoss()

    result = np.zeros_like(u)
    r, c = u.shape
    for i in range(r):
        for j in range(c):
            sample = u[i][j] * sample_a + v[i][j] * sample_b
            _, predict = torch.max(F.softmax(model(sample), dim=1), 1)
            result[i][j] = predict[0].item()
    return result


def plot_loss_3d(sample_original, label, model):
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # 进行颜色填充
    # plt.contourf(ii, jj, kk, 8, cmap='rainbow')
    # plt.contourf(ii, jj, kk, 8, cmap='coolwarm')
    # 进行等高线绘制
    # contour = plt.contour(ii, jj, obtain_loss_matrix(ii, jj, sample_a, sample_b, label, model), 8, colors='black')
    # # 线条标注的绘制
    # plt.clabel(c, inline=True, fontsize=10)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    # -----------------------------分别找到两对像素---------
    pixel_k = 2
    idx, attributions_abs = pixel_attribution_sort(model, sample_original, label, pixel_k, FIND_MAX=False)

    # --------------------------准备基向量,确定坐标轴的大致形状---------------------------------

    x_i = np.linspace(0, 1, 50)
    y_i = np.linspace(0, 1, 50)
    ii, jj = np.meshgrid(x_i, y_i)  # 获得网格坐标矩阵

    sample_a = torch.zeros_like(sample_original, dtype=torch.float)
    sample_b = torch.zeros_like(sample_original, dtype=torch.float)
    sample_offset = sample_original.detach().clone()  # torch.rand(size=sample_original.shape)
    #
    sample_a = sample_a.flatten()
    sample_a[idx[0]] = 1.
    sample_a = sample_a.reshape(sample_original.shape)
    #
    sample_b = sample_b.flatten()
    sample_b[idx[1]] = 1.
    sample_b = sample_b.reshape(sample_original.shape)

    sample_a = sample_a.to(device)
    sample_b = sample_b.to(device)
    sample_offset = sample_offset.to(device)

    # --------------------------绘制loss平面---------------------------------
    kk = obtain_loss_matrix(ii, jj, sample_a, sample_b, sample_offset, label, model)
    # kk = obtain_label_matrix(ii, jj, sample_a, sample_b, label, model)

    # 绘制曲面
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Plot the 3D surface
    surf = ax.plot_surface(ii, jj, kk, cmap=cm.rainbow)

    # Plot projections of the contours for each dimension.
    ax.contour(ii, jj, kk, zdir='z', offset=kk.min(), cmap=cm.rainbow)

    ax.set(xlabel='Direction 1', ylabel='Direction 2', zlabel='Loss')
    fig.colorbar(surf,
                 fraction=0.023, pad=0.04
                 # shrink=0.5, aspect=5
                 )
    # plt.tight_layout()
    plt.show()


def select_a_sample_to_plot(dataset, mode_name):
    test_loader, _ = load_dataset(dataset, batch_size=1, is_shuffle=False)

    model, model_acc = load_model_args(mode_name)
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
    # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
    original_images, original_labels = next(iter(test_loader))
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

    plot_loss_3d(original_images, original_labels, model)
    # plot_loss_3d(images, labels, model)
    # pca_contour_3d(sample_a, sample_b, labels, model, many_adv_images_list)


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['Arial']
    # 解决保存图像是负号'-'显示为方块的问题
    mpl.rcParams['axes.unicode_minus'] = False
    mpl.rcParams['savefig.dpi'] = 200  # 保存图片分辨率
    mpl.rcParams['figure.dpi'] = 200  # 分辨率
    mpl.rcParams['figure.constrained_layout.use'] = True
    # mpl.rcParams["figure.subplot.left"], mpl.rcParams["figure.subplot.right"] = 0.05, 0.99
    # mpl.rcParams["figure.subplot.bottom"], mpl.rcParams["figure.subplot.top"] = 0.07, 0.99
    # mpl.rcParams["figure.subplot.wspace"], mpl.rcParams["figure.subplot.hspace"] = 0.1005, 0.1005
    plt.rcParams['xtick.direction'] = 'in'  # 将x周的刻度线方向设置向内
    plt.rcParams['ytick.direction'] = 'in'  # 将y轴的刻度方向设置向内
    # mpl.rcParams['figure.constrained_layout.use'] = True

    model_name_set = ['VGG16', 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121']


    # select_a_sample_to_plot('MNIST',
    #                         'ResNet18_ImageNet',
    #                         Epsilon=5 / 255,
    #                         Iterations=10,
    #                         Momentum=1.0)

    # select_a_sample_to_plot('MNIST',
    #                         'FC_256_128')
    def example_plot(ax, fontsize=12, hide_labels=False):
        pc = ax.pcolormesh(np.random.randn(30, 30), vmin=-2.5, vmax=2.5)
        if not hide_labels:
            ax.set_xlabel('x-label', fontsize=fontsize)
            ax.set_ylabel('y-label', fontsize=fontsize)
            ax.set_title('Title', fontsize=fontsize)
        return pc


    fig = plt.figure(layout='constrained', figsize=(10, 4))
    subfigs = fig.subfigures(1, 2, wspace=0.07, width_ratios=[2, 1])

    # -------------------sub-figure-1
    # sharex 和 sharey 表示坐标轴的属性是否相同，可选的参数：True，False，row，col，默认值均为False，表示画布中的四个ax是相互独立的；
    # True 表示所有子图的x轴（或者y轴）标签是相同的，
    # row 表示每一行之间的子图的x轴（或者y轴）标签是相同的（不同行的子图的轴标签可以不同），
    # col表示每一列之间的子图的x轴（或者y轴）标签是相同的（不同列的子图的轴标签可以不同）
    axsLeft = subfigs[0].subplots(1, 2, sharey=True)
    subfigs[0].set_facecolor('0.75')
    for ax in axsLeft:
        pc = example_plot(ax)
    subfigs[0].suptitle('Left plots', fontsize='x-large')
    subfigs[0].colorbar(pc, shrink=0.6, ax=axsLeft, location='bottom')

    # ----------------sub-figure-2
    axsRight = subfigs[1].subplots(3, 1, sharex=True)
    for nn, ax in enumerate(axsRight):
        pc = example_plot(ax, hide_labels=True)
        if nn == 2:
            ax.set_xlabel('xlabel')
        if nn == 1:
            ax.set_ylabel('ylabel')

    subfigs[1].set_facecolor('0.85')
    subfigs[1].colorbar(pc, shrink=0.6, ax=axsRight)
    subfigs[1].suptitle('Right plots', fontsize='x-large')

    fig.suptitle('Figure suptitle', fontsize='xx-large')

    plt.show()

    print()
    print("----ALL WORK HAVE BEEN DONE!!!----")
