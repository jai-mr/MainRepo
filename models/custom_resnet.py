import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Basic block of the ResNet.
    """

    expansion = 1

    def __init__(self, in_planes, planes, norm_type, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes) if norm_type == "LN" else nn.BatchNorm2d(planes)

    def forward(self, x):
        """
        Forward method.
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = F.relu(out)
        return out


class CustomResNetClass(nn.Module):
    """
    ResNet Architecture.
    """

    def __init__(self, block, norm_type, num_classes=10):
        super().__init__()

        # Prep Layer
        self.prep = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=1),  # RF: 3x3
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # Conv Block
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1),  # RF: 3x3
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # ResNet Block
        self.resblock_1 = block(128, 128, norm_type=norm_type)

        # Conv Block
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        # Conv Block
        self.conv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1, stride=1),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )

        # ResNet Block
        self.resblock_2 = block(512, 512, norm_type=norm_type)

        # MaxPooling with Kernel Size 4
        self.pooling = nn.MaxPool2d(4, 4)

        # Fully Connected Layer
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x):
        """
        Forward method.
        """
        out = self.prep(x)
        out = self.conv_layer_1(out)

        res_out_1 = self.resblock_1(out)

        # Residual Block
        out = out + res_out_1
        out = self.conv_layer_2(out)
        out = self.conv_layer_3(out)

        res_out_2 = self.resblock_2(out)

        # Residual Block
        out = out + res_out_2
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out


def CustomResNet(norm_type="BN"):
    """
    Custom ResNet model.
    """
    return CustomResNetClass(BasicBlock, norm_type)
