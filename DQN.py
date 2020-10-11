"""
Created by Edward Li at 10/7/20
Using a CNN structure from this paper (https://www.diva-portal.org/smash/get/diva2:1342302/FULLTEXT01.pdf)
and Google Deep mind
"""
import torch.nn.functional as F
import torch.nn as nn


class DQNCNN(nn.Module):
    def __init__(self, side_length, outputs, in_channels = 1, features=[32, 64, 128]):
        super(DQNCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=features[0], kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(num_features=features[0])
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=features[0], out_channels=features[1], kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=features[1])
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=1)
        # self.conv3 = nn.Conv2d(in_channels=features[1], out_channels=features[2], kernel_size=5, stride=2, padding=2)
        # self.bn3 = nn.BatchNorm2d(num_features=features[2])

        def calc_out_shape(l, k, s, p):
            return (l - k + 2 * p) // s + 1

        # calculate shape
        side_length = calc_out_shape(side_length, 5, 1, 2)  # after first conv
        side_length = calc_out_shape(side_length, 2, 2, 0)  # after first maxpool
        side_length = calc_out_shape(side_length, 3, 1, 1)  # after second conv
        side_length = calc_out_shape(side_length, 2, 1, 0)  # after second maxpool
        # side_length = calc_out_shape(side_length, 5, 2, 2)  # after third conv

        self.linear = nn.Linear(side_length * side_length * features[2], outputs)

    # return [P(stright),P(right),P(left)]
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.mp1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.mp2(x)
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.linear(x.contiguous().view(x.size(0), -1))


class DQNFCN(nn.Module):
    pass