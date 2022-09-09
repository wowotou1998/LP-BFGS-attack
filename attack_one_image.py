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
import os
import torch.nn.functional as F
import os
from minimize import minimize
import torch.nn as nn
import os


def show_one_image(images, title):
    plt.figure()
    print(images.shape)
    images = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # print(images.detach().numpy()[0].shape)
    plt.imshow(images)
    plt.title(title)
    plt.show()


def show_two_image(images, adv_images, titles):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # plt.figure()
    print(images.shape)

    # image = images.cpu().detach().numpy()[0].transpose(1, 2, 0)
    # plt.imshow(image)
    # plt.title(title)
    # plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    for i, x in enumerate((images, adv_images)):
        image = x.cpu().detach().numpy()[0].transpose(1, 2, 0)
        axes[i].imshow(image, cmap='gray')
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


def tanh_space(x):
    return 1 / 2 * (torch.tanh(x) + 1)


def inverse_tanh_space(x):
    # torch.atanh is only for torch >= 1.7.0
    def atanh(x):
        return 0.5 * torch.log((1 + x) / (1 - x))

    return atanh(x * 2 - 1)


def generate_adv_images(attack_name, model, images, labels, epsilon, Iterations, Momentum):
    if attack_name == 'second-order':
        x0 = images[0:1]
        x0[x0 == 0.0] = 1. / 255 * 0.1
        x0[x0 == 1.0] = 1. - 1. / 255 * 0.1
        w = inverse_tanh_space(x0)

        # def func(image, label=labels[0].detach().clone()):
        #     logits = model(tanh_space(image))
        #     return -logits[0][label]
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()

        def func(w, labels=labels.detach().clone(), init_images=images.detach().clone()):
            kappa = 0
            c = 10
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

        res1 = minimize(func, w.detach().clone(), method='bfgs', max_iter=100, tol=1e-5, disp=True)
        adv_image = tanh_space(res1.x)
        return adv_image
    atk = generate_attack_method(attack_name, model, epsilon, Iterations, Momentum)
    images_under_attack = atk(images, labels)
    return images_under_attack


def attack_one_model(model, test_loader, test_loader_size, attack_method_set, Epsilon, Iterations, Momentum, ):
    test_count = 0.
    epoch_num = 0
    acc_before_attack = 0.
    sample_attacked = 0
    attack_success_num = torch.zeros(len(attack_method_set), dtype=torch.float)
    attack_success_confidence = torch.zeros(len(attack_method_set), dtype=torch.float)
    # acc_after_FGSM, acc_after_PGD, acc_after_MI_FGSM, \
    # acc_after_I_FGSM, acc_after_Adam_FGSM, acc_after_Adam_FGSM2 = 0, 0, 0, 0, 0, 0
    device = torch.device("cuda:%d" % (0) if torch.cuda.is_available() else "cpu")

    # every epoch has 64 images ,every images has 1 channel and the channel size is 28*28
    pbar = tqdm(total=test_loader_size)
    model.to(device)
    model.eval()

    # Norm_p = 1
    # Epsilon = 10
    epsilon = Epsilon / 255.
    # Iterations = 10
    # Momentum = 0.9
    # print('len(test_loader)', len(test_loader))

    for data in test_loader:
        # train_loader is a class, DataSet is a list(length is 2,2 tensors) ,images is a tensor,labels is a tensor
        # images is consisted by 64 tensor, so we will get the 64 * 10 matrix. labels is a 64*1 matrix, like a vector.
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        images.requires_grad_(True)
        epoch_size = labels.shape[0]
        epoch_num += 1
        test_count += epoch_size
        pbar.update(epoch_size)
        # if test_count > 10000:
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
            print('empty tensor')
            continue
        if min(labels.shape) == 0:
            print('empty tensor')
            continue

        acc_before_attack += predict_answer.sum().item()
        """
        这里重新写一下, 对图片的处理, 攻击方法的设置
        """

        for idx, attack_name in enumerate(attack_method_set):
            images_under_attack = generate_adv_images(attack_name, model, images, labels, epsilon, Iterations, Momentum)
            confidence, predict = torch.max(F.softmax(model(images_under_attack), dim=1), dim=1)
            print(predict[0] == labels[0])
            # 记录每一个攻击方法在每一批次的成功个数
            attack_success_num[idx] += (labels != predict).sum().item()

            # 记录误分类置信度
            # 选出攻击成功的对抗样本的置信度
            # 选择攻击成功的images的confidences
            misclassification = ()
            # print('predict_answer', predict_answer)
            # 我们要确保predict correct是一个一维向量,因此使用flatten
            predict_incorrect_index = torch.flatten(torch.nonzero(labels != predict))
            # print('predict_correct_index', predict_correct_index)
            confidence_value = torch.index_select(confidence, 0, predict_incorrect_index)
            attack_success_confidence[idx] += confidence_value.sum().item()

            if epoch_num == 1:
                print('predict_correct_element_num: ', predict_correct_index.nelement())
                titles = (str(labels[0].item()), str(predict[0].item()))
                # show_one_image(images, 'image_after_' + attack_name)
                show_two_image(images, images_under_attack, titles)
                break
        sample_attacked += labels.shape[0]

        break

    attack_success_rate = (attack_success_num / sample_attacked) * 100.
    attack_success_confidence_ave = (attack_success_confidence / attack_success_num)
    print(attack_success_confidence_ave)

    print('epsilon %d/255 \nIterations %d \nMomentum %.2f' % (Epsilon, Iterations, Momentum))
    for i in range(len(attack_method_set)):
        print('%s_succ_rate = %.2f%%' % (attack_method_set[i], attack_success_rate[i]))

    pbar.close()
    print('model acc %.2f' % (acc_before_attack / test_count))
    return attack_success_rate, attack_success_confidence_ave


def attack_many_model(dataset, model_name_set, attack_method_set, batch_size, work_name, Epsilon_set, Iterations_set,
                      Momentum):
    lab_result_head = ['model', 'model acc', 'Epsilon', 'Iterations', 'Momentum'] + attack_method_set
    lab_result_content = []
    test_loader, test_dataset_size = load_dataset(dataset, batch_size, is_shuffle=True)
    for mode_name in model_name_set:
        model, model_acc = load_model_args(mode_name)
        for Epsilon in Epsilon_set:
            for Iterations in Iterations_set:
                # FGSM, I_FGSM, PGD, MI_FGSM, Adam_FGSM_acc, Adam_FGSM2_acc
                attack_method_succ_list, confidence_list = attack_one_model(model=model,
                                                                            test_loader=test_loader,
                                                                            test_loader_size=test_dataset_size,
                                                                            attack_method_set=attack_method_set,
                                                                            Epsilon=Epsilon,
                                                                            Iterations=Iterations,
                                                                            Momentum=Momentum)
                tmp_list = [mode_name, model_acc, '%d/255' % Epsilon, Iterations, Momentum] + \
                           attack_method_succ_list.numpy().tolist() + \
                           confidence_list.numpy().tolist()
                lab_result_content.append(tmp_list)
    save_model_results(work_name, lab_result_head, lab_result_content)


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    batch_size = 1
    attack_method_set = ['second-order', ]  # 'FGSM', 'I_FGSM', 'PGD', 'MI_FGSM', 'Adam_FGSM','Adam_FGSM_incomplete'
    mnist_model_name_set = ['LeNet5']  # 'LeNet5', 'FC_256_128'
    # cifar10_model_name_set = ['VGG16', ]  # 'VGG19', 'ResNet50', 'ResNet101', 'DenseNet121'
    # imagenet_model_name_set = ['ResNet50_ImageNet']
    # 'DenseNet161_ImageNet','ResNet50_ImageNet', 'DenseNet121_ImageNet VGG19_ImageNet

    attack_many_model('MNIST',
                      mnist_model_name_set, attack_method_set,
                      batch_size,
                      work_name='adam_fgsm_mnist',
                      Epsilon_set=[45],
                      Iterations_set=[3],
                      Momentum=1.0)
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
