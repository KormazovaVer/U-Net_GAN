import torch
import torch.nn as nn
from torchvision import models


# --- Компоненты Self-Attention: Spatial и Channel Attention ---
class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = self.conv(torch.cat([avg_out, max_out], dim=1))
        attn = self.sigmoid(attn)
        return x * attn


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.channel_attn = ChannelAttention(in_channels)
        self.spatial_attn = SpatialAttention(in_channels)

    def forward(self, x):
        x_out = self.channel_attn(x)
        x_out = self.spatial_attn(x_out)
        return x_out


# --- U-Net генератор с Self-Attention (Spatial + Channel) ---
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=64):
        super(UNetGenerator, self).__init__()

        def down_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        def up_block(in_c, out_c, dropout=0.0):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU()
            ]
            if dropout:
                layers.append(nn.Dropout(dropout))
            return nn.Sequential(*layers)

        self.down1 = down_block(in_channels, features, normalize=False)
        self.down2 = down_block(features, features * 2)
        self.down3 = down_block(features * 2, features * 4)
        self.down4 = down_block(features * 4, features * 8)
        self.attn1 = SelfAttention(features * 8)
        self.down5 = down_block(features * 8, features * 8)
        self.down6 = down_block(features * 8, features * 8)
        self.down7 = down_block(features * 8, features * 8)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, 4, 2, 1),
            nn.ReLU()
        )

        self.up1 = up_block(features * 8, features * 8, dropout=0.5)
        self.up2 = up_block(features * 16, features * 8, dropout=0.5)
        self.attn2 = SelfAttention(features * 8)
        self.up3 = up_block(features * 16, features * 8, dropout=0.5)
        self.up4 = up_block(features * 16, features * 8)
        self.up5 = up_block(features * 16, features * 4)
        self.up6 = up_block(features * 8, features * 2)
        self.up7 = up_block(features * 4, features)
        self.final_up = nn.ConvTranspose2d(features * 2, out_channels, 4, 2, 1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d4 = self.attn1(d4)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)

        bottleneck = self.bottleneck(d7)

        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat([up1, d7], 1))
        up2 = self.attn2(up2)
        up3 = self.up3(torch.cat([up2, d6], 1))
        up4 = self.up4(torch.cat([up3, d5], 1))
        up5 = self.up5(torch.cat([up4, d4], 1))
        up6 = self.up6(torch.cat([up5, d3], 1))
        up7 = self.up7(torch.cat([up6, d2], 1))
        final = self.final_up(torch.cat([up7, d1], 1))
        return self.tanh(final)


# --- PatchGAN дискриминатор ---
class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=6, features=64):
        super(PatchDiscriminator, self).__init__()

        def block(in_c, out_c, stride=2, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(in_channels, features, normalize=False),
            block(features, features * 2),
            block(features * 2, features * 4),
            block(features * 4, features * 8, stride=1),
            nn.Conv2d(features * 8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)


# --- Перцептуальная потеря на основе VGG16 ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg16(pretrained=True).features.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.layers = layers if layers else [3, 8, 15, 22]
        self.criterion = nn.L1Loss()

    def forward(self, x, y):
        loss = 0
        x_in = x
        y_in = y
        for i, layer in enumerate(self.vgg):
            x_in = layer(x_in)
            y_in = layer(y_in)
            if i in self.layers:
                loss += self.criterion(x_in, y_in)
        return loss
