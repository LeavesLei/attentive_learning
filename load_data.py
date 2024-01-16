import sys
import numpy as np
import torch
from torch.utils.data import random_split
import torchvision
import random
from PIL import Image
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

data_path = '/home/slei5230/Leaves/datasets'

transforms_mnist = transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (1)),
    ])


transforms_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (1, 1, 1)),
    ])


def load_mnist(batch_size=128):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')
    

    trainset = datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms_mnist)
    testset = datasets.MNIST(data_path, train=False,
                       transform=transforms_mnist)
    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader, num_classes


def load_fashionmnist(batch_size=128):

    # Data Uplaod
    print('\n[Phase 1] : Data Preparation')

    trainset = datasets.FashionMNIST(data_path, train=True, download=True,
                       transform=transforms_mnist)
    testset = datasets.FashionMNIST(data_path, train=False,
                       transform=transforms_mnist)
    num_classes = 10

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader, num_classes


def load_cifar(dataset='cifar10', batch_size=128):
    # Data Uplaod

    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 100

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader, num_classes


def load_half_cifar(dataset='cifar10', batch_size=128, trainset_mode='former', former_sample_ratio=1., latter_sample_ratio=1.):
    # Data Uplaod
    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 100

    trainset.targets = np.array(trainset.targets)
    testset.targets = np.array(testset.targets)

    former_data = trainset.data[trainset.targets<(num_classes//2)]
    former_target = trainset.targets[trainset.targets<(num_classes//2)]
    latter_data = trainset.data[trainset.targets>(num_classes//2 - 1)]
    latter_target = trainset.targets = trainset.targets[trainset.targets>(num_classes//2 - 1)]

    former_sample_size = int(former_data.shape[0] * former_sample_ratio)
    latter_sample_size = int(latter_data.shape[0] * latter_sample_ratio)

    former_data = former_data[:former_sample_size]
    former_target = former_target[:former_sample_size]
    latter_data = latter_data[:latter_sample_size]
    latter_target = latter_target[:latter_sample_size]
    
    if trainset_mode == 'former':
        trainset.data = former_data
        testset.data = testset.data[testset.targets<(num_classes//2)]
        trainset.targets = former_target
        testset.targets = testset.targets[testset.targets<(num_classes//2)]
        num_classes = num_classes // 2

    elif trainset_mode == 'latter':
        trainset.data = latter_data
        testset.data = testset.data[testset.targets>(num_classes//2 - 1)]
        trainset.targets = latter_target - num_classes//2 
        testset.targets = testset.targets[testset.targets>(num_classes//2 - 1)] - num_classes//2
        num_classes = num_classes // 2

    elif trainset_mode == 'joint':
        trainset.data = np.vstack((former_data, latter_data))
        trainset.targets = np.hstack((former_target, latter_target))
        trainset.targets = trainset.targets  % (num_classes // 2)
        testset.targets = testset.targets % (num_classes // 2)
        num_classes = num_classes // 2

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader, num_classes


def load_shift_cifar(dataset='cifar10', batch_size=128, mode='red', alpha='1.'):
    # Data Uplaod
    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 100

    trainset.targets = np.array(trainset.targets)
    testset.targets = np.array(testset.targets)

    if mode == 'red':
        trainset.data[-15000:, :, :, 1] = trainset.data[-15000:, :, :, 1] * alpha
        trainset.data[-15000:, :, :, 2] = trainset.data[-15000:, :, :, 2] * alpha
    elif mode == 'green':
        trainset.data[-15000:, :, :, 0] = trainset.data[-15000:, :, :, 0] * alpha
        trainset.data[-15000:, :, :, 2] = trainset.data[-15000:, :, :, 2] * alpha
    elif mode == 'gray':
        trainset.data[-15000:, :, :, 0] = trainset.data[-15000:, :, :, 0] * alpha + (1-alpha) * trainset.data[-15000:, :, :, 2]
        trainset.data[-15000:, :, :, 1] = trainset.data[-15000:, :, :, 1] * alpha + (1-alpha) * trainset.data[-15000:, :, :, 2]
    elif mode == 'sub':
        trainset.data = trainset.data[:-15000]
        trainset.targets = trainset.targets[:-15000]

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader, num_classes


def load_noise_half_cifar(dataset='cifar10', batch_size=128, noise_set='former', label_noise_ratio=0.):
    # Data Uplaod
    if dataset == 'cifar10':
        print("| Preparing CIFAR-10 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 10
    elif dataset == 'cifar100':
        print("| Preparing CIFAR-100 dataset...")
        sys.stdout.write("| ")
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms_cifar)
        testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transforms_cifar)
        num_classes = 100

    trainset.targets = np.array(trainset.targets)
    testset.targets = np.array(testset.targets)

    former_data = trainset.data[trainset.targets<(num_classes//2)]
    former_target = trainset.targets[trainset.targets<(num_classes//2)]
    latter_data = trainset.data[trainset.targets>(num_classes//2 - 1)]
    latter_target = trainset.targets = trainset.targets[trainset.targets>(num_classes//2 - 1)]

    if noise_set == 'former':
        random.seed(9)
        random_label_num = int(label_noise_ratio * len(former_target))
        random_label_list = [random.randint(0, num_classes//2 - 1) for i in range(random_label_num)]
        former_target = [random_label_list[index] if index < len(random_label_list)  else label for index, label in enumerate(former_target)]
    elif noise_set == 'latter':
        random.seed(9)
        random_label_num = int(label_noise_ratio * len(latter_target))
        random_label_list = [random.randint(num_classes//2, num_classes - 1) for i in range(random_label_num)]
        latter_target = [random_label_list[index] if index < len(random_label_list)  else label for index, label in enumerate(latter_target)]
    
    trainset.data = np.vstack((former_data, latter_data))
    trainset.targets = np.hstack((former_target, latter_target))
    trainset.targets = trainset.targets  % (num_classes // 2)
    testset.targets = testset.targets % (num_classes // 2)
    num_classes = num_classes // 2
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)
    return trainloader, testloader, num_classes


def load_mnist_train(batch_size=128):
    trainset = datasets.MNIST(data_path, train=True, download=True,
                       transform=transforms_mnist)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader


def load_fashionmnist_train(batch_size=128):
    trainset = datasets.FashionMNIST(data_path, train=True, download=True,
                       transform=transforms_mnist)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)
    return trainloader


