# Modified from https://github.com/flower-kyo/Tinysleepnet-pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F

class tiny_CNN(nn.Module):
    def __init__(self, n_inchan = 3, fs = 128):
        super(tiny_CNN, self).__init__()
        self.in_chans = n_inchan
        self.padding_edf = {  # no padding
            'conv1': (0, 0),
            'max_pool1': (0, 0),
            'conv2': (0, 0),
            'max_pool2': (0, 0),
        }
        self.padconv1 = nn.ConstantPad1d(self.padding_edf['conv1'], 0)
        self.conv1 = nn.Conv1d(in_channels=n_inchan, out_channels=128, kernel_size=fs//2, stride=fs//4, bias=False)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.relu1 = nn.ReLU(inplace=True)
        self.padmaxpool1 = nn.ConstantPad1d(self.padding_edf['max_pool1'], 0)  # max p 1
        self.maxpool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.dropout1 = nn.Dropout(p=0.5)
            
        self.padconv2 = nn.ConstantPad1d(self.padding_edf['conv2'], 0)  # conv2
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(num_features=128)
        self.relu2 = nn.ReLU(inplace=True)
        
        self.padconv3 = nn.ConstantPad1d(self.padding_edf['conv2'], 0)  # conv3
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False)
        self.bn3 = nn.BatchNorm1d(num_features=128)
        self.relu3 = nn.ReLU(inplace=True)

        self.padconv4 = nn.ConstantPad1d(self.padding_edf['conv2'], 0)  # conv4
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=8, stride=1, bias=False)
        self.bn4 = nn.BatchNorm1d(num_features=128)
        self.relu4 = nn.ReLU(inplace=True)
        
        self.padmaxpool2 = nn.ConstantPad1d(self.padding_edf['max_pool2'], 0)  # max p 2
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout2 = nn.Dropout(p=0.5)
        
    def forward(self, x):           # (B, C, L)
        B, C, L = x.shape
        # x = x.view((-1, 38400, self.in_chans))        # (bs * 5, 3750,5)
        # x = x.permute(0,2,1)
        
        x = self.padconv1(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.padmaxpool1(x)
        x = self.maxpool1(x)
        x = self.dropout1(x)

        x = self.padconv2(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.padconv3(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        x = self.padconv4(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)

        x = self.padmaxpool2(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        return x

class tiny_RNN(nn.Module):
    def __init__(self, input_channel = 128):
        super(tiny_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size = input_channel, hidden_size = 128, num_layers = 1, 
                            batch_first = True, bidirectional= False)
    def forward(self, x) :
        p, _ = self.lstm(x, None)
        return p

class TinySleepNet(nn.Module):
    def __init__(self, input_channel = 3, output_class = 10):
        super(TinySleepNet, self).__init__()
        self.input_channel = input_channel
        self.cnn = tiny_CNN()
        self.rnn = tiny_RNN()
        self.drop = nn.Dropout(.5)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 1, bias=True)
        self.fc.bias.data = torch.Tensor([50.0])

    def forward(self, x):
        B, C, L = x.shape
        x = self.cnn(x)         #[B, C, L]
        x = x.swapaxes(1,2) 
        x = self.rnn(x)
        x = self.gap(x.swapaxes(1,2)).squeeze(-1)
        x = self.drop(x)
        x = self.fc(x)
        return x, None

if __name__ == "__main__":
    pass

    net = TinySleepNet()
    x = torch.rand([16, 3, 38400])
    print(net(x).shape)