import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class BlockV2(nn.Module):
    expansion = 1
    def __init__(self, input_channels, output_channels, stride = 1, downsample = None):
        super(BlockV2, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, 7, stride, padding=3, bias = False)
        self.bn1 = nn.BatchNorm1d(output_channels)

        self.conv2 = nn.Conv1d(output_channels, output_channels, 7, stride = 1, padding=3, bias = False)
        self.bn2 = nn.BatchNorm1d(output_channels)

        self.downsample = downsample

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)

        out += residual 

        return out

class Backbone(nn.Module):
    def __init__(self, input_channels, input_length=1260, layers=[1, 1, 1, 1], embed_dim=8):
        self.inplanes = 8
        super(Backbone, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(8, momentum=0.01)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(BlockV2, 4, layers[0], stride=4)
        self.layer2 = self._make_layer(BlockV2, 8, layers[1], stride=4)
        self.layer3 = self._make_layer(BlockV2, 16, layers[2], stride=4)

    def _make_layer(self, block, planes, blocks, stride=2):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class HypResnet(nn.Module):
    def __init__(self, input_channels = 5, frame_len = 3750):
        super(HypResnet, self).__init__()
        self.input_channels = input_channels
        self.cnn = Backbone(input_channels)
        self.gru = nn.GRU(input_size=16, hidden_size=5, num_layers=1, batch_first=True, bidirectional = True)

    def forward(self, x):
        # x = torch.swapaxes(x,1,2)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)       
        x, _ = self.gru(x)
        return x, _
         
if __name__ == '__main__':
    RAN = HypResnet()
    hypno = torch.rand([1, 1260, 5])
    pred = RAN(hypno, None)