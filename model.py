import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_classes=1):
        super().__init__()
        self.n_classes = n_classes
        self.inc = DoubleConv(3, 64)
        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )
        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )
        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )

        self.bottleneck = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.up_conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x5 = self.bottleneck(x4)

        x = self.up1(x5)
        x = self.up_conv1(torch.cat([x, x4], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x3], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x, x2], dim=1))
        x = self.up4(x)
        x = self.up_conv4(torch.cat([x, x1], dim=1))
        x = self.outc(x)
        return x

def initialize_weights(module):
    class_name = module.__class__.__name__
    if class_name.startswith('Conv'):
        print(f'Initializing weights for layer {class_name}')
        _, in_maps, k, _ = module.weight.shape
        n = k * k * in_maps
        std = (2/n) ** 0.5
        nn.init.normal_(module.weight.data, mean=0.0, std=std)
    else:
        print(f'No need to initialize weights for {class_name}')
