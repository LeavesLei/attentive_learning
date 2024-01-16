import torch
from utils import train, test, get_hms, get_pred
from load_data import load_shift_cifar, load_cifar
from models.wide_resnet import WRN28
from models.vgg import Vgg16
from models.resnet import ResNet18
import argparse
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--mode', type=str, default='sub') # red, green, gray, sub, full
parser.add_argument('--alpha', type=float, default=0.)
parser.add_argument('--optimizer_name', type=str, default='sgdm', help='select optimizer')
parser.add_argument('--net_type', type=str, default='vgg16', help='model type')
parser.add_argument('--index', type=str, default='1', help='iteration index')
parser.add_argument('--fix_init', type=str, default='True', help='fix initialization')
parser.add_argument('--save_path', type=str, default='save_folder/dist_shift/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

batch_size = args.batch_size
dataset = args.dataset
mode = args.mode
alpha = args.alpha
optimizer_name = args.optimizer_name
net_type = args.net_type
index = args.index
fix_init = args.fix_init
save_path = args.save_path

if fix_init == 'False':
    fix_init = False
else:
    fix_init = True

# Load data
trainloader, _, num_classes = load_shift_cifar(dataset=dataset, batch_size=batch_size, mode=mode, alpha=alpha)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
if net_type == 'vgg16':
    net = Vgg16(num_classes=num_classes, dataset=dataset).to(device)
elif net_type == 'resnet':
    net = ResNet18(num_classes=num_classes, dataset=dataset).to(device)
elif net_type == 'wrn':
    net = WRN28(num_classes=num_classes, dataset=dataset).to(device)

if fix_init:
    net = torch.load('save_folder/initialized_nets/' + net_type + '_initialization_num_classes_' + str(num_classes)).to(device)

# Training

lr = 0.1

start_epoch = 1
num_epochs = 100

print('\n[Phase 3] : Training model')
print('| Training Epochs = ' + str(num_epochs))
print('| Initial Learning Rate = ' + str(lr))

elapsed_time = 0
for epoch in range(start_epoch, start_epoch + num_epochs):

    start_time = time.time()
    train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, optimizer_name=optimizer_name)
    # Save model
    #torch.save(net, save_path + net_type + '_' + trainset_mode + '_' + dataset + '_epoch_' + str(epoch) + '_index_' + index)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))

save_path = save_path + net_type + '/'
if mode == 'sub' or mode == 'full':
    torch.save(net, save_path + net_type + '_' + dataset + '_' + mode + '_index_' + index)
else:
    torch.save(net, save_path + net_type + '_' + dataset + '_' + mode  + '_alpha_' + str(alpha) + '_index_' + index)