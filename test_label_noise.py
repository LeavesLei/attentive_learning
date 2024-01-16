import torch
import numpy as np
import time
from utils import get_confidence_score, get_targets, get_hms, get_pred, convert_prediction_to_loss
from load_data import load_half_cifar
from models.cnn import CNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', help='dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--load_path', type=str, default='save_folder/locally_and_generalization/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

dataset = args.dataset
batch_size = args.batch_size
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
#num_epochs = 100
index_num = 10



for net_type in ['vgg16', 'resnet', 'wrn']:
    load_path = source_load_path + net_type + '/sgdm/'
    
    former_pred = []
    latter_pred = []

    for index in range(1, 1 + index_num):
        net = torch.load(load_path + net_type + '_sgdm_former_' + dataset + '_sample_ratio_1.0_index_' + str(index)).to(device)
        former_pred.append(get_pred(net, former_testloader))
    
    for index in range(1, 1 + index_num):
        net = torch.load(load_path + net_type + '_sgdm_latter_' + dataset + '_sample_ratio_1.0_index_' + str(index)).to(device)
        latter_pred.append(get_pred(net, latter_testloader))


    prediction_matrix = np.array(former_pred).transpose()
    former_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

    prediction_matrix = np.array(latter_pred).transpose()
    latter_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

    for label_noise_ratio in [0, 0.05, 0.1, 0.15, 0.2, 0.25]:#['sgd', 'sgdm', 'rmsprop', 'adam']:
        joint_pred_former = []
        joint_pred_latter = []
    
        if label_noise_ratio == 0:
            joint_path = load_path
            for index in range(1, 1 + index_num):
                net = torch.load(joint_path + net_type + '_sgdm_joint_' + dataset + '_joint_ratio_1.0x1.0_index_' + str(index)).to(device)
                joint_pred_former.append(get_pred(net, former_testloader))
                joint_pred_latter.append(get_pred(net, latter_testloader))
        else:
            joint_path = '/public/data1/users/leishiye/learning_locally/save_folder/label_noise/' + net_type + '/'
            for index in range(1, 1 + index_num):
                net_former = torch.load(joint_path + net_type + '_former_' + dataset + '_label_noise_ratio_' + str(label_noise_ratio) +  '_index_' + str(index)).to(device)
                net_latter = torch.load(joint_path + net_type + '_latter_' + dataset + '_label_noise_ratio_' + str(label_noise_ratio) +  '_index_' + str(index)).to(device)
                joint_pred_former.append(get_pred(net_latter, former_testloader))
                joint_pred_latter.append(get_pred(net_former, latter_testloader))


        print("===========================================================")
        print("Net type: " + net_type)
        print("Label noise:" + str(label_noise_ratio))

        prediction_matrix = np.array(joint_pred_former).transpose()
        joint_pred_former = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        prediction_matrix = np.array(joint_pred_latter).transpose()
        joint_pred_latter = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        #print("Former test error: " + str(1 - (sum(former_pred == former_targets) / former_targets.shape[0])))
        #print("Former dataset-perturbation stability: " + str(1 - (sum(former_pred == joint_pred_former) / former_targets.shape[0])))
        #print("Joint test error: " + str(1 - (sum(joint_pred_former == former_targets) / former_targets.shape[0])))
        print("Former dataset-perturbation stability: " + str(0.5 * np.sum(np.abs(joint_pred_former - former_pred)) / former_targets.shape[0]))
        #print("Latter test error: " + str(1 - (sum(latter_pred == latter_targets) / latter_targets.shape[0])))
        #print("Latter dataset-perturbation stability: " + str(1 - (sum(latter_pred == joint_pred_latter) / latter_targets.shape[0])))
        #print("Latter test error: " + str(1 - (sum(joint_pred_latter == latter_targets) / latter_targets.shape[0])))
        print("Latter dataset-perturbation stability: " + str(0.5 * np.sum(np.abs(joint_pred_latter - latter_pred)) / latter_targets.shape[0]))