## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
from collections import OrderedDict

# *** Conv2d output dimensions ***
# height_out = (height_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# width_out = (width_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# weights_out = height_out * width_out * channels_out
#
# With values: strid = 1, padding = 0, dilation = 1
# height_out = height_in - kernel_size + 1
# width_out = width_in - kernel_size + 1
#
# *** MaxPool2d output dimensions ***
# height_out = (height_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# width_out = (width_in + 2*padding - dilation*(kernel_size - 1) - 1)/stride + 1
# weights_out = height_out * width_out * channels_out
#
# With values: strid = 2, padding = 0, dilation = 1
# height_out = (height_in - kernel_size)/2 + 1
# width_out = (width_in - kernel_size)/2 + 1


class MyNaimishNet(nn.Module):

    def __init__(self, input_channel=1, output_channel=[32,64,128,256],dropout_prob=[0.1,0.2,0.3,0.4,0.5,0.6]):
        super(MyNaimishNet,self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Layer 1
        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1',  nn.Conv2d(input_channel, output_channel[0], 4)),
            ('elu_1', nn.ELU()),
            ('bn1', nn.BatchNorm2d(output_channel[0])),
            ('dropout_1', nn.Dropout2d(dropout_prob[0]))
            ]))

        #LAYER 2
        self.conv2 = nn.Sequential(
            OrderedDict([
            ('conv2',  nn.Conv2d(output_channel[0], output_channel[1], 3)),
            ('elu_2', nn.ELU()),
            ('bn2', nn.BatchNorm2d(output_channel[1])),
            ('dropout_2', nn.Dropout2d(dropout_prob[1]))
            ]))

        #LAYER 3
        self.conv3 = nn.Sequential(
            OrderedDict([
            ('conv3',  nn.Conv2d(output_channel[1], output_channel[2], 2)),
            ('elu_3', nn.ELU()),
            ('bn3', nn.BatchNorm2d(output_channel[2])),
            ('dropout_3', nn.Dropout2d(dropout_prob[2]))
            ]))

        #Layer 4
        self.conv4 = nn.Sequential(
            OrderedDict([
            ('conv4',  nn.Conv2d(output_channel[2], output_channel[3], 1)),
            ('elu_4', nn.ELU()),
            ('bn4' , nn.BatchNorm2d(output_channel[3])),
            ('dropout_4', nn.Dropout2d(dropout_prob[3]))
            ]))

        # Layer 5 Flatten
        self.fc1 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=13*13*256, out_features=1000)),
                ('elu_5' , nn.ELU()),
                ('bn5' , nn.BatchNorm1d(1000)),
                ('dropout_5' , nn.Dropout2d(dropout_prob[4]))
            ]))
        
        self.fc2 = nn.Sequential(
            OrderedDict([
                ('fc2' , nn.Linear(in_features=1000, out_features=500)),
                ('Sig' , nn.Tanh()),
                ('bn6' , nn.BatchNorm1d(500)),
                ('dropout_6' , nn.Dropout2d(dropout_prob[5]))
            ]))
        #Layer 7
        #OUT FKP: (X, Y)
        self.fc3 = nn.Linear(in_features=500, out_features=136)
        
        #Custom weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = I.uniform_(m.weight, a = 0.0, b = 1.0)
            elif isinstance(m, nn.Linear):
                m.weight= I.xavier_uniform_(m.weight, gain=1)
        
    def forward(self, x):
        
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = self.pool(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
    
class Net(nn.Module):

    def __init__(self, input_channel=1, output_channel=[32,64,128],dropout_prob=[0.1,0.2,0.3,0.4,0.5,0.6]):
        
        super(Net,self).__init__()
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        #Layer 1
        self.conv1 = nn.Sequential(
            OrderedDict([
            ('conv1',  nn.Conv2d(input_channel, output_channel[0], 5)),
            ('Relu_1', nn.ReLU()),
            ('dropout_1' , nn.Dropout2d(dropout_prob[0])),
            ]))

        #LAYER 2
        self.conv2 = nn.Sequential(
            OrderedDict([
            ('conv2',  nn.Conv2d(output_channel[0], output_channel[1], 5)),
            ('Relu_2', nn.ReLU()),
            ('bn2' , nn.BatchNorm2d(output_channel[1])),
            ]))

        #LAYER 3
        self.conv3 = nn.Sequential(
            OrderedDict([
            ('conv3',  nn.Conv2d(output_channel[1], output_channel[2], 5)),
            ('Relu_3', nn.ReLU()),
            ]))

        # Layer 4 Flatten
        self.fc1 = nn.Sequential(
            OrderedDict([
                ('fc1', nn.Linear(in_features=24*24*128, out_features=256)),
                ('Relu_4' , nn.ReLU()),
            ]))
        #Layer 5
        #OUT FKP: (X, Y)
        self.fc2 = nn.Linear(in_features=256, out_features=136)
          #Custom weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight = I.uniform_(m.weight, a = 0.0, b = 1.0)
            elif isinstance(m, nn.Linear):
                m.weight= I.xavier_uniform_(m.weight, gain=1)

    def forward(self, x):
        
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = self.pool(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x