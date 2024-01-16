import torch
from models.wide_resnet import WRN28
from models.vgg import Vgg16
from models.resnet import ResNet18
from models.vit import vit
import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--net_type', type=str, default='cnn', help='model type')
parser.add_argument('--num_classes', type=int, default=5, help='class number')
parser.add_argument('--input_channel', type=int, default=3, help='input channel number')
parser.add_argument('--save_path', type=str, default='save_folder/initialized_nets/', help='save path')
args = parser.parse_args()

print('--------args----------')
for k in list(vars(args).keys()):
    print('%s: %s' % (k, vars(args)[k]))
print('--------args----------\n')


net_type = args.net_type
num_classes = args.num_classes
input_channel = args.input_channel
save_path = args.save_path

# Model initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
if net_type == 'vgg16':
    net = Vgg16(num_classes=num_classes, input_channel=input_channel).to(device)
elif net_type == 'resnet':
    net = ResNet18(num_classes=num_classes).to(device)
elif net_type == 'wrn':
    net = WRN28(num_classes=num_classes).to(device)
elif net_type == 'vit':
    net = vit(num_classes=num_classes).to(device)
# save initial model
torch.save(net, save_path + net_type + '_initialization_num_classes_' + str(num_classes))