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

    result = np.zeros_like(u)
    r, c = u.shape
    for i in range(r):
        for j in range(c):
            sample = u[i][j] * sample_a + v[i][j] * sample_b
            _, predict = torch.max(F.softmax(model(sample), dim=1), 1)
            result[i][j] = predict[0].item()
    return result


def plot_loss_3d(image, label, model):
    # plt.show()
    # plt.figure(figsize=(8, 8))
    # 进行颜色填充
    # plt.contourf(X, Y, Z, 8, cmap='rainbow')
    # plt.contourf(X, Y, Z, 8, cmap=cm.coolwarm)
    # 进行等高线绘制
    # contour = plt.contour(X, Y, obtain_loss_matrix(X, Y, sample_a, sample_b, label, model), 8, colors='black')
    # # 线条标注的绘制
    # plt.clabel(c, inline=True, fontsize=10)
    def find_pixel_loc(shape, pixel_idx):
        b, c, h, w = shape
        if b > 1:
            raise Exception('batch size should be 1')
        a = torch.zeros(shape, dtype=torch.int)
        a = a.flatten()
        for i in range(pixel_idx.numel()):
            a[pixel_idx[i].item()] = 1
        a = a.reshape(shape)
        locs = torch.nonzero(a)
        return locs

    def get_plot_data(FIND_MAX=True):
        device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")

        # -----------------------------分别找到两对像素---------
        pixel_k = 2
        pixel_idx, attributions_abs = pixel_attribution_sort(model, image, label, pixel_k, FIND_MAX)

        # --------------------------准备基向量,确定坐标轴的大致形状---------------------------------
        x_i = np.linspace(0, 1, 50)
        y_i = np.linspace(0, 1, 50)
        X, Y = np.meshgrid(x_i, y_i)  # 获得网格坐标矩阵

        sample_a = torch.zeros_like(image, dtype=torch.float)
        sample_b = torch.zeros_like(image, dtype=torch.float)
        sample_offset = image.detach().clone()  # torch.rand(size=image.shape)
        #
        sample_a = sample_a.flatten()
        sample_a[pixel_idx[0]] = 1.
        sample_a = sample_a.reshape(image.shape)
        #
        sample_b = sample_b.flatten()
        sample_b[pixel_idx[1]] = 1.
        sample_b = sample_b.reshape(image.shape)

        sample_a = sample_a.to(device)
        sample_b = sample_b.to(device)
        sample_offset = sample_offset.to(device)
        Z = obtain_loss_matrix(X, Y, sample_a, sample_b, sample_offset, label, model)
        return X, Y, Z, pixel_idx, attributions_abs

    # --------------------------绘制loss平面---------------------------------
    # 获得三维数据
    # X, Y, Z, pixel_idx, attri_score = get_plot_data(FIND_MAX=True)
    # X2, Y2, Z2, pixel_idx2, attri_score2 = get_plot_data(FIND_MAX=False)
    # plot_data = [X, Y, Z, pixel_idx, attri_score, X2, Y2, Z2, pixel_idx2, attri_score2, image]

    with open('Checkpoint/plot_data_50_50.pkl', 'rb') as f:
        plot_data = pickle.load(f)
        X, Y, Z, pixel_idx, attri_score, X2, Y2, Z2, pixel_idx2, attri_score2, image = plot_data

    y_min = min(Z.min(), Z2.min())
    y_max = max(Z.max(), Z2.max())
    y_min = y_min - 0.3 * (y_max - y_min)
    y_norm = plt.Normalize(vmin=y_min, vmax=y_max)

    attri_min = torch.min(attri_score).item()
    attri_max = torch.max(attri_score).item()
    attri_norm = plt.Normalize(vmin=attri_min, vmax=attri_max)

    sm = plt.cm.ScalarMappable(cmap=cm.coolwarm, norm=y_norm)
    sm2 = plt.cm.ScalarMappable(cmap=cm.Blues, norm=attri_norm)
    # Z = obtain_label_matrix(X, Y, sample_a, sample_b, label, model)

    # -------------------------绘制曲面-----------------
    fig = plt.figure(layout='constrained', figsize=(9 * 0.9, 4 * 0.9))
    subfigs = fig.subfigures(1, 2, width_ratios=[2, 1])
    subfigs[0].suptitle('Loss surface in different space')

    # -------------------sub-figure-0------------------

    ax1 = subfigs[0].add_subplot(1, 2, 1, projection='3d')
    ax2 = subfigs[0].add_subplot(1, 2, 2, projection='3d')
    ax1.plot_surface(X, Y, Z, cmap=cm.coolwarm, norm=y_norm, linewidth=0, alpha=0.9)
    ax1.contourf(X, Y, Z, zdir='z', offset=y_min, cmap=cm.coolwarm, norm=y_norm)
    ax1.set(xlabel=r'$x_1$', ylabel=r'$x_2$', zlabel='')

    # ax1.xaxis.set_major_locator(ticker.LinearLocator(5))  # X轴显示5个刻度值
    # ax1.yaxis.set_major_locator(ticker.LinearLocator(5))  # X轴显示5个刻度值
    ax1.set_zlim((y_min, y_max))
    ax1.set_title(r'Space spanned by pixel $x_1,x_2$')

    ax1.view_init(30, 45)
    ax2.plot_surface(X2, Y2, Z2, cmap=cm.coolwarm, norm=y_norm, linewidth=0, alpha=0.9)
    ax2.contourf(X2, Y2, Z2, zdir='z', offset=y_min, cmap=cm.coolwarm, norm=y_norm)
    ax2.set(xlabel=r'$x_3$', ylabel=r'$x_4$', zlabel='')
    ax2.set_zlim((y_min, y_max))
    ax2.view_init(30, 45)
    ax2.set_title(r'Space spanned by pixel $x_3,x_4$ ')
    cbar = subfigs[0].colorbar(sm, shrink=0.45, ax=[ax1, ax2], label='CE loss', location='bottom')
    cbar.formatter.set_powerlimits((0, 0))  # 设置colorbar为科学计数法
    cbar.update_ticks()

    # ----------------sub-figure-1----------------
    axsRight = subfigs[1].subplots(2, 2, sharex=True, sharey=True).flatten()
    attri_image = attri_score.reshape(image.shape)

    titles = ['R channel', 'G channel', 'B channel', 'Image']
    for i, ax in enumerate(axsRight):
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_title(titles[i])
        if i < 3:
            ax.imshow(attri_image[0][i].cpu().detach().numpy(),
                      cmap=cm.Greens,
                      norm=attri_norm
                      )

        else:
            ax.imshow(image[0].cpu().detach().numpy().transpose(1, 2, 0))

    subfigs[1].colorbar(sm2, shrink=0.8, ax=axsRight, label='Attribution Score', location='bottom')
    subfigs[1].suptitle('Attribution score in different channel')
    # ---------------------annotate the perturbed pixels
    x1_x2 = find_pixel_loc(shape=image.shape, pixel_idx=pixel_idx)
    print(x1_x2)
    x3_x4 = find_pixel_loc(shape=image.shape, pixel_idx=pixel_idx2)
    print(x3_x4)

    for i in range(pixel_idx.numel()):
        channel = x1_x2[i][1].item()
        h, w = x1_x2[i][2].item(), x1_x2[i][3].item()
        axsRight[channel].plot(w, h, 's', markerfacecolor='none', markersize=3,
                               color='r', alpha=0.6)
        axsRight[channel].text(w + 1, h + 1, r'$x_{%d}$' % (i + 1),
                               family='Times New Roman',  # 标注文本字体
                               fontsize=14,  # 文本大小
                               # fontweight='bold',  # 字体粗细
                               color='r'  # 文本颜色
                               )
        # axsRight[channel].annotate(r'$x_{%d}$' % i,
        #                            xy=(w, h),
        #                            xytext=(h + 5, w + 5),
        #                            arrowprops=dict(alpha=0.6,
        #                                            arrowstyle='->',
        #                                            connectionstyle='arc3,rad=0.5',  # 有多个参数可选
        #                                            color='r',
        #                                            ),
        #                            )

        channel = x3_x4[i][1].item()
        h, w = x3_x4[i][2].item(), x3_x4[i][3].item()
        axsRight[channel].plot(w, h, 's', markerfacecolor='none', markersize=3,
                               color='b', alpha=0.6)
        axsRight[channel].text(w + 1, h + 1, r'$x_{%d}$' % (i + 3),
                               family='Times New Roman',  # 标注文本字体
                               fontsize=14,  # 文本大小
                               # fontweight='bold',  # 字体粗细
                               color='b'  # 文本颜色
                               )

        # axsRight[channel].annotate(r'$x_{%d}$' % (i + 2),
        #                            xy=(w, h),
        #                            xytext=(h + 5, w + 5),
        #                            arrowprops=dict(alpha=0.6,
        #                                            arrowstyle='->',
        #                                            connectionstyle='arc3,rad=0.5',  # 有多个参数可选
        #                                            color='b',
        #                                            ),
        #                            )

    # print(find_pixel_loc(shape=image.shape, pixel_idx=pixel_idx2))

    # subfigs[1].suptitle('Right plots')
    # fig.suptitle('Figure suptitle')

    plt.show(block=True)
    import datetime
    current_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    # fig.savefig('%s_%s.pdf' % ('attribution_effect_opt', current_time))
    with open('./Checkpoint/plot_data_50_50.pkl', 'wb') as f:
        pickle.dump(plot_data, f)


def select_a_sample_to_plot(dataset, mode_name):
    test_loader, _, _ = load_dataset(dataset, batch_size=1, is_shuffle=False)

    model, model_acc = load_model_args(mode_name)
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
            plot_loss_3d(images, labels, model)
            break
        else:
            print('wrong prediction')

    # plot_loss_3d(images, labels, model)
    # pca_contour_3d(sample_a, sample_b, labels, model, many_adv_images_list)


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
    plot_loss_3d(None, None, None)

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
