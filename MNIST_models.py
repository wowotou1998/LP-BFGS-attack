import torch.nn as nn
import torch.nn.functional as F


class lenet5(nn.Module):
    def __init__(self):
        '''构造函数，定义网络的结构'''
        super().__init__()
        # 定义卷积层，1个输入通道，6个输出通道，5*5的卷积filter，外层补上了两圈0,因为输入的是32*32
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 第二个卷积层，6个输入，16个输出，5*5的卷积filter
        self.conv2 = nn.Conv2d(6, 16, 5)

        # 最后是三个全连接层
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # print(x.shape)
        '''前向传播函数'''
        # 先卷积，然后调用relu激活函数，再最大值池化操作
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # 第二次卷积+池化操作
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        # 重新塑形,将多维数据重新塑造为二维数据，16*4*4
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print('size', x.size())
        # 第一个全连接
        # print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class FC_256_128(nn.Module):
    # 500个隐藏层,学习率0.1,扰动40左右还是比较高的
    # def __init__(self):
    #     super().__init__()
    #     self.conv1 = nn.Sequential(  # 第1个卷积过程的设置
    #         # feature map's size = (image's size
    #         nn.Conv2d(
    #             in_channels=1,  # 图片只有1个通道，为灰度图
    #             out_channels=10,  # 输出的通道为10,相当于提取10个特征图
    #             kernel_size=28,  # 卷积核的大小为28*28
    #             stride=1,  # 步长为1
    #             padding=0,  # 图片周边填充为0 pixel
    #         ),  # 输出的特征图为（1*1*10）
    #         nn.Sigmoid(),  # 激活函数ReLU
    #     )  # 结果为（1,1,10）
    #     # 最后的全连接层，前面有10个神经元，本层有10个神经元
    #     self.out = nn.Linear(1 * 1 * 10, 10)  # 这里设定了最后的全连接的输入输出参数有多少个
    #
    #     # 全连接层会得到上一卷积层运算出的结果，将卷积层的运算结果展开为一维数组，则这个一维数组有7*7*32个元素
    #     # 全连接层的输出是0-9这10个数，因此第二个参数设定为10
    #
    # def forward(self, x):
    #     # x = torch.tensor(x, dtype=torch.float32)
    #     x = self.conv1(x)
    #     x = x.view(x.size(0), -1)  # flatten操作
    #     # forward() 函数中,input首先经过卷积层，此时的输出x是包含batch size维度为4的tensor，即(batch size，channels，height，width)，
    #     # x.size(0) 指batch size的值,x = x.view(x.size(0), -1)简化x = x.view(batchsize, -1)。
    #     return self.out(x)
    def __init__(self):
        super().__init__()
        # 最后是三个全连接层
        self.fc1 = nn.Linear(1 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
