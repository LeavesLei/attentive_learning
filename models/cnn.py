import torch.nn as tnn

def conv_layer(chann_in, chann_out, k_size, s_size, p_size):
    layer = tnn.Sequential(
        tnn.Conv2d(chann_in, chann_out, kernel_size=k_size, stride=s_size, padding=p_size, bias=False),
        #tnn.BatchNorm2d(chann_out),
        tnn.ReLU()
    )
    return layer


def fc_layer(size_in, size_out):
    layer = tnn.Sequential(
        tnn.Linear(size_in, size_out, bias=False),
        #tnn.BatchNorm1d(size_out),
        tnn.ReLU()
    )
    return layer

class CNN(tnn.Module):
    def __init__(self, num_classes=10, input_channel=1):
        super(CNN, self).__init__()

        self.layer1 = conv_layer(input_channel, 16, 3, 2, 1)
        self.layer2 = conv_layer(16, 32, 3, 2, 1)
        self.layer3 = conv_layer(32, 64, 3, 2, 1)
        self.layer4 = conv_layer(64, 128, 3, 2, 1)

        # Final layer
        self.layer5 = tnn.Linear(512, num_classes, bias=False)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out.view(out.size(0), -1)
        out = self.layer5(out)
        return out