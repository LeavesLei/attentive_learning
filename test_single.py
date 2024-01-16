import torch
import numpy as np
import time
from utils import get_pred, get_targets, convert_prediction_to_loss
from load_data import load_half_cifar
from models.cnn import CNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--net_type', type=str, default='vgg16', help='model type')
parser.add_argument('--optimizer_name', type=str, default='sgdm', help='select optimizer')
parser.add_argument('--load_path', type=str, default='save_folder/locally_and_generalization/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

dataset = args.dataset
batch_size = args.batch_size
optimizer_name = args.optimizer_name
net_type = args.net_type
source_load_path = args.load_path


# Load data
_, former_testloader, num_classes = load_half_cifar(dataset=dataset, batch_size=batch_size, trainset_mode='former')
_, latter_testloader, num_classes = load_half_cifar(dataset=dataset, batch_size=batch_size, trainset_mode='latter')

former_targets = get_targets(former_testloader)
latter_targets = get_targets(latter_testloader)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

start_epoch = 1
index_num = 10

for net_type in ['vgg16', 'resnet', 'wrn', 'vit']:
    for optimizer_name in ['sgd', 'sgdm', 'rmsprop', 'adam']:

        former_pred = []
        latter_pred = []
        joint_pred_former = []
        joint_pred_latter = []
        
        load_path = source_load_path + net_type + '/' + optimizer_name + '/'

        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_former_' + dataset + '_sample_ratio_1.0_index_' + str(index)).to(device)
            former_pred.append(get_pred(net, former_testloader))

        
        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_latter_' + dataset + '_sample_ratio_1.0_index_' + str(index)).to(device)
            latter_pred.append(get_pred(net, latter_testloader))

        
        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_joint_' + dataset + '_joint_ratio_1.0x1.0_index_' + str(index)).to(device)
            joint_pred_former.append(get_pred(net, former_testloader))
            joint_pred_latter.append(get_pred(net, latter_testloader))



        print("===========================================================")
        print("Net type: " + net_type)
        print("Optimizer: " + optimizer_name)

        prediction_matrix = np.array(former_pred).transpose()
        former_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        prediction_matrix = np.array(latter_pred).transpose()
        latter_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        prediction_matrix = np.array(joint_pred_former).transpose()
        joint_pred_former = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])


        prediction_matrix = np.array(joint_pred_latter).transpose()
        joint_pred_latter = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        index_list  = [num_classes * i for i in range(former_targets.shape[0])] + former_targets
        test_error = 1 - np.mean(former_pred.reshape(-1)[index_list])
        print("former test error: " + str(test_error))
        print("former stability: " + str(0.5 * np.sum(np.abs(joint_pred_former - former_pred)) / former_targets.shape[0]))

        index_list  = [num_classes * i for i in range(latter_targets.shape[0])] + latter_targets
        test_error = 1 - np.mean(latter_pred.reshape(-1)[index_list])
        print("latter test error: " + str(test_error))
        print("latter stability: " + str(0.5 * np.sum(np.abs(joint_pred_latter - latter_pred)) / latter_targets.shape[0]))
