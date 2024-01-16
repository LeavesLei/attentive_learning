import sys
import math
import numpy as np
import torch
import time
import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
softmax = nn.Softmax(dim=1)
num_epochs = 100


def learning_rate(init, epoch):
    optim_factor = 0
    if(epoch > 90):
        optim_factor = 3
    elif(epoch > 60):
        optim_factor = 2
    elif(epoch > 30):
        optim_factor = 1

    return init*math.pow(0.2, optim_factor)

def test(net, testloader, epoch=1):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()

        # Save checkpoint when best model
        acc = 100. * correct / total
        print("\n| Validation Epoch #%d\t\t\tLoss: %.4f Acc@1: %.2f%%" %(epoch, loss.item(), acc))
    return test_loss, (acc/100).detach().cpu().numpy()


def train(net, trainloader, epoch=1, lr=0.1, optimizer_name='sgdm'):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate(lr, epoch), weight_decay=5e-4)
    elif optimizer_name == 'sgdm':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate(lr, epoch), momentum=0.9, weight_decay=5e-4)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate(lr, epoch))#, weight_decay=5e-4)
    elif optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(net.parameters(), lr=learning_rate(lr, epoch), weight_decay=5e-4)


    print('\n=> Training Epoch #%d, LR=%.4f' %(epoch, learning_rate(lr, epoch)))
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda() # GPU settings
        optimizer.zero_grad()
        outputs = net(inputs.float())
        loss = criterion(outputs, targets) # Loss
        loss.backward()  # Backward Propagation
        optimizer.step() # Optimizer update

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        sys.stdout.write('\r')
        sys.stdout.write('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%'
                %(epoch, num_epochs, batch_idx+1,
                    len(trainloader), loss.item(), 100.*correct/total))
        sys.stdout.flush()
    return train_loss


def get_hms(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)

    return h, m, s


def get_output(net, dataloader):
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.type(torch.FloatTensor)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            confidence_score =outputs
            if batch_idx == 0:
                confidence_score_list = confidence_score
            else:
                confidence_score_list = torch.vstack((confidence_score_list, confidence_score))
    return confidence_score_list.detach().cpu().numpy()


def get_margin(confidence_score):
    sorted_list = sorted(confidence_score)
    return sorted_list[-1] - sorted_list[-2]


def get_pred(net, testloader):
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.type(torch.FloatTensor)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            _, predicted = torch.max(outputs.data, 1)
            if batch_idx == 0:
                predict_list = predicted
            else:
                predict_list = torch.hstack((predict_list, predicted))
    return predict_list.detach().cpu().numpy()


def get_confidence_score(net, testloader):
    net.eval()
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.type(torch.FloatTensor)
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            confidence_score = softmax(outputs)
            if batch_idx == 0:
                confidence_score_list = confidence_score
            else:
                confidence_score_list = torch.vstack((confidence_score_list, confidence_score))
    return confidence_score_list.detach().cpu().numpy()


def get_targets(testloader):
    for batch_idx, (inputs, targets) in enumerate(testloader):
        if batch_idx == 0:
            targets_list = targets
        else:
            targets_list = torch.hstack((targets_list, targets))
    return targets_list.detach().cpu().numpy()

def convert_prediction_to_loss(prediction_list, num_classes=10, repeated=5):
    probability_list = []
    for i in range(num_classes):
        probability_list.append(np.count_nonzero(prediction_list == i)/repeated)
    return np.array(probability_list)