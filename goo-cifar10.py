import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import argparse
from googlenet import googlenet
from torch.optim import lr_scheduler
#from torchvision.models import googlenet

import os
# 定义是否使用GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 参数设置,使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints') #输出结果保存路径
args = parser.parse_args()

# 超参数设置
EPOCH = 100   #遍历数据集次数
#pre_epoch = 0  # 定义已经遍历数据集的次数
BATCH_SIZE = 512     #批处理尺寸(batch_size)
LR = 0.01        #学习率

# 准备数据集并预处理
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),  #先四周填充0，在吧图像随机裁剪成32*32
    transforms.RandomHorizontalFlip(),  #图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), #R,G,B每层的归一化用到的均值和方差
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/home/houpuyue_2020/code/resnet/data', train=True, download=True, transform=transform_train) #训练数据集
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)   #生成一个个batch进行批训练，组成batch的时候顺序打乱取

testset = torchvision.datasets.CIFAR10(root='/home/houpuyue_2020/code/resnet/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)
# Cifar-10的标签
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# 模型定义-ResNet
os.environ['CUDA_VISIBLE_DEVICES']='0'

#checkpoint = torch.load('./model/model_mulstep.pth')
net = googlenet()
net = nn.DataParallel(net)

#net.load_state_dict(checkpoint)
#net.load_state_dict(torch.load('./model/model_mulstep.pth'))
#net = torch.load('./model/model_mulstep.pth')
#net = nn.DataParallel(net)
net.cuda()
# 定义损失函数和优化方式
criterion = nn.CrossEntropyLoss()  #损失函数为交叉熵，多用于多分类问题
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4) #优化方式为mini-batch momentum-SGD，并采用L2正则化（权重衰减）
lr_schedule = lr_scheduler.MultiStepLR(optimizer,[60,80],0.1)
#lr_schedule = lr_scheduler.ExponentialLR(optimizer,gamma=1)

# 训练
if __name__ == "__main__":
	#if not os.path.exists(args.outf):
		#os.makedirs(args.outf)
    best_acc = 0  #2 初始化best test accuracy
    print("Start Training, googlenet!")  # 定义遍历数据集的次数
    with open("ci-acc16.txt", "w") as f:
        with open("ci-log16.txt", "w")as f2:
            for epoch in range(0, EPOCH):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0

                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    #inputs, labels = inputs.to(device), labels.to(device)
                    inputs,labels = inputs.cuda(),labels.cuda()
                    optimizer.zero_grad()

                    # forward + backward
                    outputs,aux2,aux1 = net(inputs)
                    loss1 = criterion(outputs, labels)
                    loss2 = criterion(aux2,labels)
                    loss3 = criterion(aux1,labels)

                    loss = loss1+0.3*loss2+0.3*loss3
                    loss.backward()
                    optimizer.step()

                    # 每训练1个batch打印一次loss和准确率
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                lr_schedule.step()
                # 每训练完一个epoch测试一下准确率
                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0
                    total = 0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.cuda(), labels.cuda()
                        outputs = net(images)
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    print('测试分类准确率为：%.3f%%' % (100 * correct / total))
                    acc = 100. * correct / total
                    # 将每次测试结果实时写入acc.txt文件中
                    #print('Saving model......')

                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("ci-best_acc16.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            print("Training Finished, TotalEPOCH=%d" % EPOCH)
            torch.save(net.state_dict(), './model/model_aux.pth')


