import torch
from utils import train, test, get_hms
from load_data import load_noise_half_cifar
from models.wide_resnet import WRN28
from models.vgg import Vgg16
from models.resnet import ResNet18
import argparse
import time
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "2"

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=256, help='batch size')
parser.add_argument('--noise_set', type=str, default='former') # former, latter, joint
parser.add_argument('--label_noise_ratio', type=float, default='0.0')
parser.add_argument('--net_type', type=str, default='vgg16', help='model type')
parser.add_argument('--index', type=str, default='1', help='iteration index')
parser.add_argument('--fix_init', type=str, default='True', help='fix initialization')
parser.add_argument('--save_path', type=str, default='save_folder/label_noise/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

batch_size = args.batch_size
dataset = args.dataset
noise_set = args.noise_set
label_noise_ratio = args.label_noise_ratio
net_type = args.net_type
index = args.index
fix_init = args.fix_init
save_path = args.save_path

if fix_init == 'False':
    fix_init = False
else:
    fix_init = True

trainloader, _, num_classes = load_noise_half_cifar(dataset=dataset, batch_size=batch_size, noise_set=noise_set, label_noise_ratio=label_noise_ratio)

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
    train(net=net, trainloader=trainloader, epoch=epoch, lr=lr, optimizer_name='sgdm')
    # Save model
    #torch.save(net, save_path + net_type + '_' + trainset_mode + '_' + dataset + '_epoch_' + str(epoch) + '_index_' + index)

    epoch_time = time.time() - start_time
    elapsed_time += epoch_time
    print('| Elapsed time : %d:%02d:%02d' % (get_hms(elapsed_time)))


save_path = save_path + net_type + '/' 

torch.save(net, save_path + net_type + '_' + noise_set + '_' + dataset + '_label_noise_ratio_' + str(label_noise_ratio) + '_index_' + index)