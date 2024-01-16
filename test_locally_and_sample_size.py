import torch
import numpy as np
import time
from utils import get_confidence_score, get_targets, get_pred, convert_prediction_to_loss
from load_data import load_half_cifar
from models.cnn import CNN
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar10', help='dataset')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--net_type', type=str, default='vgg16', help='model type')
parser.add_argument('--load_path', type=str, default='save_folder/locally_and_generalization/', help='iteration index')
parser.add_argument('--save_path', type=str, default='save_folder/results_locally_and_sample_size/', help='iteration index')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')

dataset = args.dataset
batch_size = args.batch_size
net_type = args.net_type
source_load_path = args.load_path
save_path = args.save_path

# Load data
_, former_testloader, num_classes = load_half_cifar(dataset=dataset, batch_size=batch_size, trainset_mode='former')
_, latter_testloader, num_classes = load_half_cifar(dataset=dataset, batch_size=batch_size, trainset_mode='latter')

former_targets = get_targets(former_testloader)
latter_targets = get_targets(latter_testloader)

former_index_list  = [num_classes * i for i in range(former_targets.shape[0])] + former_targets
latter_index_list  = [num_classes * i for i in range(latter_targets.shape[0])] + latter_targets
# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()

start_epoch = 1
#num_epochs = 100
index_num = 10
optimizer_name = 'sgdm'

for net_type in [net_type]:
    load_path = source_load_path + net_type + '/' + optimizer_name + '/'

    former_baseline_pred = []
    latter_baseline_pred = []

    for index in range(1, 1 + index_num):
        net = torch.load(load_path + net_type + '_' + optimizer_name + '_former_' + dataset + '_sample_ratio_1.0_index_' + str(index)).to(device)
        former_baseline_pred.append(get_pred(net, former_testloader))

    for index in range(1, 1 + index_num):
        net = torch.load(load_path + net_type + '_' + optimizer_name + '_latter_' + dataset + '_sample_ratio_1.0_index_' + str(index)).to(device)
        latter_baseline_pred.append(get_pred(net, latter_testloader))
    
    prediction_matrix = np.array(former_baseline_pred).transpose()
    former_baseline_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

    prediction_matrix = np.array(latter_baseline_pred).transpose()
    latter_baseline_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])
    
    former_test_error = []
    former_m1_stability = []
    former_m2_stability = []
    former_m1_joint_error = []
    former_m2_joint_error = []

    latter_test_error = []
    latter_m1_stability = []
    latter_m2_stability = []
    latter_m1_joint_error = []
    latter_m2_joint_error = []

    for sample_ratio in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        former_pred = []
        latter_pred = []
        joint_m1_pred_former = []
        joint_m1_pred_latter = []
        joint_m2_pred_former = []
        joint_m2_pred_latter = []

        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_former_' + dataset + '_sample_ratio_' + str(sample_ratio) + '_index_' + str(index)).to(device)
            former_pred.append(get_pred(net, former_testloader))


        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_latter_' + dataset + '_sample_ratio_' + str(sample_ratio) + '_index_' + str(index)).to(device)
            latter_pred.append(get_pred(net, latter_testloader))


        prediction_matrix = np.array(former_pred).transpose()
        former_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        prediction_matrix = np.array(latter_pred).transpose()
        latter_pred = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])


        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_joint_' + dataset + '_joint_ratio_1.0x' + str(sample_ratio) + '_index_' + str(index)).to(device)
            joint_m1_pred_former.append(get_pred(net, former_testloader))
            joint_m1_pred_latter.append(get_pred(net, latter_testloader))

        for index in range(1, 1 + index_num):
            net = torch.load(load_path + net_type + '_' + optimizer_name + '_joint_' + dataset + '_joint_ratio_' + str(sample_ratio) + 'x1.0_index_' + str(index)).to(device)
            joint_m2_pred_former.append(get_pred(net, former_testloader))
            joint_m2_pred_latter.append(get_pred(net, latter_testloader))

        prediction_matrix = np.array(joint_m1_pred_former).transpose()
        joint_m1_pred_former = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        prediction_matrix = np.array(joint_m1_pred_latter).transpose()
        joint_m1_pred_latter = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])
        
        prediction_matrix = np.array(joint_m2_pred_former).transpose()
        joint_m2_pred_former = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])

        prediction_matrix = np.array(joint_m2_pred_latter).transpose()
        joint_m2_pred_latter = np.array([convert_prediction_to_loss(prediction_list, num_classes=num_classes, repeated=10) for prediction_list in prediction_matrix])


        print("Net type: " + net_type)
        print("Sample ratio: " + str(sample_ratio))


        former_test_error.append(1 - np.mean(former_pred.reshape(-1)[former_index_list]))
        former_m1_stability.append(0.5 * np.sum(np.abs(joint_m1_pred_former - former_baseline_pred)) / former_targets.shape[0])
        former_m2_stability.append(0.5 * np.sum(np.abs(joint_m2_pred_former - former_pred)) / former_targets.shape[0])
        former_m1_joint_error.append(1 - np.mean(joint_m1_pred_former.reshape(-1)[former_index_list]))
        former_m2_joint_error.append(1 - np.mean(joint_m2_pred_former.reshape(-1)[former_index_list]))

        latter_test_error.append(1 - np.mean(latter_pred.reshape(-1)[latter_index_list]))
        latter_m1_stability.append(0.5 * np.sum(np.abs(joint_m1_pred_latter - latter_pred)) / latter_targets.shape[0])
        latter_m2_stability.append(0.5 * np.sum(np.abs(joint_m2_pred_latter - latter_baseline_pred)) / latter_targets.shape[0])
        latter_m1_joint_error.append(1 - np.mean(joint_m1_pred_latter.reshape(-1)[latter_index_list]))
        latter_m2_joint_error.append(1 - np.mean(joint_m2_pred_latter.reshape(-1)[latter_index_list]))

    np.save(save_path + "result_locally_learning_and_sample_size_former_test_error_" + dataset + "_" + net_type + ".npy", np.array(former_test_error))
    np.save(save_path + "result_locally_learning_and_sample_size_former_varying_perturbation_stability_" + dataset + "_" + net_type + ".npy", np.array(former_m1_stability))
    np.save(save_path + "result_locally_learning_and_sample_size_former_varying_source_stability_" + dataset + "_" + net_type + ".npy", np.array(former_m2_stability))
    np.save(save_path + "result_locally_learning_and_sample_size_former_varying_perturbation_joint_error_" + dataset + "_" + net_type + ".npy", np.array(former_m1_joint_error))
    np.save(save_path + "result_locally_learning_and_sample_size_former_varying_source_joint_error_" + dataset + "_" + net_type + ".npy", np.array(former_m2_joint_error))

    np.save(save_path + "result_locally_learning_and_sample_size_latter_test_error_" + dataset + "_" + net_type + ".npy", np.array(latter_test_error))
    np.save(save_path + "result_locally_learning_and_sample_size_latter_varying_perturbation_stability_" + dataset + "_" + net_type + ".npy", np.array(latter_m2_stability))
    np.save(save_path + "result_locally_learning_and_sample_size_latter_varying_source_stability_" + dataset + "_" + net_type + ".npy", np.array(latter_m1_stability))
    np.save(save_path + "result_locally_learning_and_sample_size_latter_varying_perturbation_joint_error_" + dataset + "_" + net_type + ".npy", np.array(latter_m2_joint_error))
    np.save(save_path + "result_locally_learning_and_sample_size_latter_varying_source_joint_error_" + dataset + "_" + net_type + ".npy", np.array(latter_m1_joint_error))