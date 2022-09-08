# ----------------------train-------------------------------------
import os

import torch
from torch import optim, nn


def evaluate_accuracy(test_data_loader, model, device):
    test_acc_sum, n = 0.0, 0
    model = model.to(device)
    model.eval()
    for sample_data, sample_true_label in test_data_loader:
        # data moved to GPU or CPU
        sample_data = sample_data.to(device)
        sample_true_label = sample_true_label.to(device)
        sample_predicted_probability_label = model(sample_data)
        _, predicted_label = torch.max(sample_predicted_probability_label.data, 1)
        test_acc_sum += predicted_label.eq(sample_true_label.data).cpu().sum().item()
        # test_acc_sum += (sample_predicted_probability_label.argmax(dim=1) == sample_true_label).sum().item()
        n += sample_data.shape[0]
    return (test_acc_sum / n) * 100.0


# this training function is only for classification task
def training(model,
             train_data_loader, test_data_loader,
             epochs, criterion, optimizer,
             enable_cuda, gpu_id,
             load_model_args,
             model_name='MNIST'):
    loss_record, train_accuracy_record, test_accuracy_record = [], [], []
    # ---------------------------------------------------------------------
    if enable_cuda:
        device = torch.device("cuda:%d" % (gpu_id) if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    if optimizer is None:
        optimizer = optim.SGD(model.parameters(), lr=0.01)

    if criterion is None:
        # 直接计算batch size中的每一个样本的loss，然后再求平均值
        criterion = nn.CrossEntropyLoss()

    # Load checkpoint.
    print('--> %s is training...' % (model_name))
    try:
        print('--> Resuming from checkpoint..')
        # assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('../Checkpoint/%s.pth' % (model_name))
        if load_model_args:
            print('--> Loading model state dict..')
            model.load_state_dict(checkpoint['model'])
        # data moved to GPU
        model = model.to(device)
        # 必须先将模型进行迁移,才能再装载optimizer,不然会出现数据在不同设备上的错误
        # optimizer.load_state_dict(checkpoint['optimizer'])
        best_test_acc = checkpoint['test_acc']
        start_epoch = checkpoint['epoch']
        print('--> Load checkpoint successfully! ')
    except Exception as e:
        print('--> %s\' checkpoint is not found ! ' % (model_name))
        best_test_acc = 0
        start_epoch = 0

    model = model.to(device)
    model.train()
    # train_data_loader is a iterator object, which contains data and label
    # sample_data is a tensor,the size is batch_size * sample_size
    # sample_true_label is the same, which is 1 dim tensor, and the length is batch_size, and each sample
    # has a scalar type value
    for epoch in range(start_epoch, start_epoch + epochs):
        train_loss_sum, train_acc_sum, sample_sum = 0.0, 0.0, 0
        for sample_data, sample_true_label in train_data_loader:

            # data moved to GPU
            sample_data = sample_data.to(device)
            sample_true_label = sample_true_label.to(device)
            sample_predicted_probability_label = model(sample_data)
            if epoch == 0 and sample_sum == 0:
                print(device)
                print(sample_data.shape, sample_true_label.shape, sample_predicted_probability_label.shape)
                # print(sample_true_label, sample_predicted_probability_label)

            # loss = criterion(sample_predicted_probability_label, sample_true_label).sum()
            loss = criterion(sample_predicted_probability_label, sample_true_label)

            # zero the gradient cache
            optimizer.zero_grad()
            # backpropagation
            loss.backward()
            # update weights and bias
            optimizer.step()

            train_loss_sum += loss.item()
            # argmax(dim=1) 中dim的不同值表示不同维度，argmax(dim=1) 返回列中最大值的下标
            # 特别的在dim=0表示二维中的行，dim=1在二维矩阵中表示列
            # train_acc_sum 表示本轮,本批次中预测正确的个数
            _, predicted_label = torch.max(sample_predicted_probability_label.data, 1)
            train_acc_sum += predicted_label.eq(sample_true_label.data).cpu().sum().item()
            # train_acc_sum += (sample_predicted_probability_label.argmax(dim=1) == sample_true_label).sum().item()
            # sample_data.shape[0] 为本次训练中样本的个数,一般大小为batch size
            # 如果总样本个数不能被 batch size整除的情况下，最后一轮的sample_data.shape[0]比batch size 要小
            # n 实际上为 len(train_data_loader)
            sample_sum += sample_data.shape[0]
            # if sample_sum % 30000 == 0:
            #     print('sample_sum %d' % (sample_sum))
            if epochs == 1:
                print('GPU Memory was locked!')
                while True:
                    pass

        # 每一轮都要干的事
        train_acc = (train_acc_sum / sample_sum) * 100.0
        test_acc = evaluate_accuracy(test_data_loader, model, device)

        # Save checkpoint.
        if test_acc > best_test_acc:
            print('Saving.. test_acc %.2f%% > best_test_acc %.2f%%' % (test_acc, best_test_acc))
            state = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'test_acc': test_acc,
                'epoch': epoch,
            }
            if not os.path.isdir('../Checkpoint'):
                os.mkdir('../Checkpoint')
            torch.save(state, '../Checkpoint/{}.pth'.format(model_name))
            best_test_acc = test_acc
        else:
            print('Not save.. test_acc %.2f%% < best_test_acc %.2f%%' % (test_acc, best_test_acc))
        # 记录每一轮的训练集准确度，损失，测试集准确度
        loss_record.append(train_loss_sum)
        train_accuracy_record.append(train_acc)
        test_accuracy_record.append(test_acc)

        print('epoch %d, train loss %.4f, train acc %.4f%%, test acc %.4f%%'
              % (epoch + 1, train_loss_sum, train_acc, test_acc))

    return [loss_record, train_accuracy_record, test_accuracy_record], best_test_acc


# 设置横纵坐标的名称以及对应字体格式
font_12 = {'family': 'Arial',
           'weight': 'normal',
           'size': 12,
           }

font_16 = {'family': 'Arial',
           'weight': 'normal',
           'size': 16,
           }
font_18 = {'family': 'Arial',
           'weight': 'normal',
           'size': 18,
           }
# 设置横纵坐标的名称以及对应字体格式
title_font = {'family': 'Arial',
              'weight': 'normal',
              'size': 18,
              }

legend_font = {
    'family': 'Arial',
    'weight': 'normal',
    'size': 16,
}

# import cupy as cp
import numpy as np


def show_img(img, title='image'):
    plt.figure("Image")  # 图像窗口名称
    # C*H*W-->H*W*C
    img = np.transpose(img, (1, 2, 0))
    # np.savetxt('img01.txt', img[:, :, 0], fmt='%0.4f')
    plt.imshow(img)
    plt.axis('off')  # 关掉坐标轴为 off
    plt.title(title, title_font)  # 图像题目
    plt.show()


def load_model_args(model, name):
    assert os.path.isdir('../Checkpoint'), 'Error: no checkpoint directory found!'
    model_name = name
    checkpoint = torch.load('../Checkpoint/%s.pth' % (model_name,), map_location={'cuda:0': 'cpu'})
    model.load_state_dict(checkpoint['model'])
    # print('%s(%.2f%%)' % (model_name, checkpoint['test_acc']))
    # cudnn.benchmark = True
    return model, model_name, checkpoint['test_acc']


def save_model_accuracy_rate(dict, model_name):
    filename = open('../Checkpoint/%s_accuracy_rate.txt' % (model_name), 'w')  # dict转txt
    for k, v in dict.items():
        filename.write(k + ':    ' + str(v))
        filename.write('\n')
    filename.close()


def save_model_results(file_name, head, rows):
    import csv, datetime
    path = '../Checkpoint/%s_%s.csv' % (file_name, datetime.datetime.now().strftime("%Y_%m_%d_%H_%M"))
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        # 写入一行数据
        writer.writerow(head)
        # 写入多行数据
        writer.writerows(rows)

    # import datetime
    # filename = open(
    #     '../Checkpoint/lab_res_%s.txt' % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").replace(':', '')),
    #     'w')  # dict转txt
    # for k, v in dict.items():
    #     filename.write(k + ':' + str(v))
    #     filename.write('\n')
    # filename.close()


import matplotlib.pyplot as plt


def show_model_performance(model_data):
    if len(model_data) != 3:
        print('the data list is wrong, some data need to be added or thrown!')
    else:
        plt.figure()
        # show two accuracy rate at the same figure
        # 想要绘制线条的画需要记号中带有‘-’
        plt.plot(model_data[0], 'go-')
        plt.title("the loss of model ", title_font)
        plt.legend(['loss trend'], prop=legend_font)
        plt.show()
        plt.figure()
        plt.plot(model_data[1], 'b+-', model_data[2], 'r*-')
        plt.title("the acc of model ", title_font)
        plt.legend(['train acc', 'test acc'], prop=legend_font)
        plt.show()
