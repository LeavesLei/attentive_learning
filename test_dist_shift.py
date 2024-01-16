import torch
import numpy as np
from utils import get_targets, train, test, get_hms, get_pred, get_confidence_score, convert_prediction_to_loss
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
parser.add_argument('--net_type', type=str, default='vgg16', help='model type')
parser.add_argument('--load_path', type=str, default='save_folder/dist_shift/', help='iteration index')
parser.add_argument('--save_path', type=str, default='save_folder/results_locally_and_dist_shift/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

batch_size = args.batch_size
dataset = args.dataset
net_type = args.net_type
load_path = args.load_path
save_path = args.save_path

index_num = 10

# Load data
_, testloader, num_classes = load_cifar(dataset=dataset)
targets = get_targets(testloader)

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

for net_type in [net_type]:
    load_path = load_path + net_type + '/'

    sub_pred = []
    full_pred = []
    
    for index in range(1, 1 + index_num):
        net = torch.load(load_path + net_type + '_' + dataset + '_sub_index_' + str(index)).to(device)
        sub_pred.append(get_pred(net, testloader))

    for index in range(1, 1 + index_num):
        net = torch.load(load_path + net_type + '_' + dataset + '_full_index_' + str(index)).to(device)
        full_pred.append(get_pred(net, testloader))


    prediction_matrix = np.array(sub_pred).transpose()
    sub_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

    prediction_matrix = np.array(full_pred).transpose()
    full_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])
    
    for mode in ['red', 'green', 'gray']: #['red','gray']: #['red', 'green', 'gray']:
        stability = []


        stability.append(0.5 * np.sum(np.abs(full_pred - sub_pred)) / targets.shape[0])

        for alpha in [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]:
            print("Mode: " + mode)
            print("Alpha: " + str(alpha))

            joint_pred = []
            for index in range(1, 1 + index_num):
                net = torch.load(load_path + net_type + '_' + dataset + '_' + mode  + '_alpha_' + str(alpha) + '_index_' + str(index)).to(device)
                joint_pred.append(get_pred(net, testloader))

            prediction_matrix = np.array(joint_pred).transpose()
            joint_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])
            
            stability.append(0.5 * np.sum(np.abs(joint_pred - sub_pred)) / targets.shape[0])
        
        np.save(save_path + "result_locally_learning_and_dist_shift_stability_" + dataset + "_" + net_type + "_" + mode + ".npy", np.array(stability))