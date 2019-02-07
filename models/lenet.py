import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        return F.relu(self.bn1(self.conv1(x)))

class ResNet3(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet3, self).__init__()
        self.layer1 = BasicBlock(3, 32)
        self.layer2 = BasicBlock(32, 32)
        self.layer3 = BasicBlock(32, 32)
        self.linear = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
