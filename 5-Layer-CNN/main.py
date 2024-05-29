import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import random

from utils import progress_bar
import resnet18k
import mcnn

# RUN WITH - HSA_OVERRIDE_GFX_VERSION=10.3.0 
# - (ROCm Compability Parameter)
# 
# HSA_OVERRIDE_GFX_VERSION=10.3.0 /bin/python3 /home/potatosalad/project/project_ML2/5-Layer-CNN/main.py

##개인의 실험 목적에 맞게 이 파라미터들을 설정하세요.
arg_lr = 0.0001
arg_resume = False
arg_batch_size = 128
arg_model = "mcnn"
arg_optimizer = "adam"
arg_noise = 0.0
arg_data_size = 1.0
arg_data = "cifar10"
arg_w_param = 4
arg_epoch = 1001

main_path = f'./5-Layer-CNN/{arg_model}_{arg_data}_{arg_optimizer}/noise-{arg_noise}_datasize-{arg_data_size}_w_param-{arg_w_param}'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
if arg_data == 'cifar10':
  num_classes = 10
elif arg_data == 'cifar100':
  num_classes = 100

def prepare_dataset():
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    if arg_data == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root='./5-Layer-CNN/data', train=True, download=True, transform=transform_train)
        new_len = int(arg_data_size*len(trainset))
        if arg_data_size < 1.0:
            trainset, _ = torch.utils.data.random_split(trainset, [new_len, len(trainset)-new_len])
        if arg_noise > 0.0:
            noise_len = int(len(trainset) * arg_noise)
            index = torch.randperm(len(trainset))[:noise_len]
            for i in index:
                noise_label = (trainset.targets[i] - 9)*(-1)
                trainset.targets[i] = noise_label

        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=arg_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./5-Layer-CNN/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    elif arg_data == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root='./5-Layer-CNN/data', train=True, download=True, transform=transform_train)
        new_len = int(arg_data_size*len(trainset))
        if arg_data_size < 1.0:
            trainset, _ = torch.utils.data.random_split(trainset, [new_len, len(trainset)-new_len])
        if arg_noise > 0.0:
            noise_len = int(len(trainset) * arg_noise)
            index = torch.randperm(len(trainset))[:noise_len]
            for i in index:
                noise_label = (trainset.targets[i] - 9)*(-1)
                trainset.targets[i] = noise_label
                
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=arg_batch_size, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./5-Layer-CNN/data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=100, shuffle=False, num_workers=2)

    return trainloader, testloader

prepare_dataset()

# Model
print('==> Building model..')
if arg_model == "resnet18k":
    net = resnet18k.make_resnet18k(c = arg_w_param, num_classes = num_classes)
    net = net.to(device)
elif arg_model == "mcnn":
    net = mcnn.make_cnn(c = arg_w_param, num_classes = num_classes)
    net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if arg_resume==True:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(os.path.join(main_path, 'checkpoint')), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(os.path.join(main_path,'./checkpoint/ckpt.pth'))
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
if arg_optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=0.1, weight_decay=5e-4)
    epoch = arg_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
elif arg_optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=arg_lr)
    epoch = arg_epoch

train_loss_history = []
train_accuracy_history = []
test_loss_history = []
test_accuracy_history = []

os.makedirs(main_path, exist_ok=True)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_accuracy = 100.*correct/total

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    train_loss_history.append(train_loss)
    train_accuracy_history.append(train_accuracy)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            test_accuracy = 100.*correct/total

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_loss_history.append(test_loss)
        test_accuracy_history.append(test_accuracy)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir(os.path.join(main_path,'checkpoint')):
            os.mkdir(os.path.join(main_path,'checkpoint'))
        torch.save(state, os.path.join(main_path,'./checkpoint/ckpt.pth'))
        best_acc = acc

if __name__ == "__main__":
    trainloader, testloader = prepare_dataset()
    for epoch in range(start_epoch, start_epoch+epoch):
        train(epoch)
        test(epoch)
        if arg_optimizer=='sgd':
            scheduler.step()

    torch.save({
    'train_loss_history': train_loss_history,
    'train_accuracy_history': train_accuracy_history,
    'test_loss_history': test_loss_history,
    'test_accuracy_history': test_accuracy_history,
    }, os.path.join(main_path,'history.pth'))
