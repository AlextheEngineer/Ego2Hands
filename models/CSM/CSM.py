import torch
from torch import nn
import torch.nn.functional as F
#from pairwise import Pairwise

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.downsample is not None:
            shortcut = self.downsample(x)

        out += shortcut
        out = self.relu(out)

        return out

class DeconvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, stride=1, upsample=None):
        super(DeconvBottleneck, self).__init__()
        self.expansion = expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        if stride == 1:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                                   stride=stride, bias=False, padding=1)
        else:
            self.conv2 = nn.ConvTranspose2d(out_channels, out_channels,
                                            kernel_size=3,
                                            stride=stride, bias=False,
                                            padding=1,
                                            output_padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.upsample = upsample

    def forward(self, x):
        shortcut = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)

        if self.upsample is not None:
            shortcut = self.upsample(x)
            
        out += shortcut
        out = self.relu(out)

        return out

class CSM(nn.Module):
    def __init__(self, downblock, upblock, num_layers, n_classes, with_energy, input_edge, n_stages = -1):
        super(CSM, self).__init__()

        self.in_channels = 32
        self.n_classes = n_classes
        self.n_stages = n_stages
        self.with_energy = with_energy
        down_layer_size = 3
        up_layer_size = 3
        num_inchan = 2 if input_edge else 1
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.conv1 = nn.Conv2d(num_inchan, 32, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.dlayer1 = self._make_downlayer(downblock, 32, down_layer_size)
        self.dlayer2 = self._make_downlayer(downblock, 64, down_layer_size, stride=2)
        self.dlayer3 = self._make_downlayer(downblock, 128, down_layer_size, stride=2)
        self.dlayer4 = self._make_downlayer(downblock, 256, down_layer_size, stride=2)

        # stage1
        if self.n_stages >= 1 or self.n_stages == -1:
            self.uplayer1_1 = self._make_up_block(upblock, 256, up_layer_size, stride=2)
            self.uplayer2_1 = self._make_up_block(upblock, 128, up_layer_size, stride=2)
            upsample_1 = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, 32,
                                   kernel_size=1, stride=2,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(32),
            )
            self.uplayer_stage_1 = DeconvBottleneck(self.in_channels, 32, 1, 2, upsample_1)
            self.conv_seg_out_1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, bias=False)
            if self.with_energy:
                self.conv_e_out_1 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, bias=False)

        # stage2
        if self.n_stages >= 2 or self.n_stages == -1:
            self.uplayer1_2 = self._make_up_block(upblock, 64, up_layer_size, stride=2)
            if self.with_energy:
                self.post_cat_2 = nn.Conv2d(134, 128, kernel_size=1, stride=1, bias=False)
            else:
                self.post_cat_2 = nn.Conv2d(131, 128, kernel_size=1, stride=1, bias=False)
            self.bn_2 = nn.BatchNorm2d(128)
            self.uplayer2_2 = self._make_up_block(upblock, 32, up_layer_size)
            upsample_2 = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, 32,
                                   kernel_size=1, stride=2,
                                   bias=False, output_padding=1),
                nn.BatchNorm2d(32),
            )
            self.uplayer_stage_2 = DeconvBottleneck(64, 32, 1, 2, upsample_2)
            self.conv_seg_out_2 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, bias=False)
            if self.with_energy:
                self.conv_e_out_2 = nn.Conv2d(32, n_classes, kernel_size=1, stride=1, bias=False)
        
    def _make_downlayer(self, block, init_channels, num_layer, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != init_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, init_channels*block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(init_channels*block.expansion),
            )
        layers = []
        layers.append(block(self.in_channels, init_channels, stride, downsample))
        self.in_channels = init_channels * block.expansion
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels))

        return nn.Sequential(*layers)

    def _make_up_block(self, block, init_channels, num_layer, stride=1):
        upsample = None
        if stride != 1 or self.in_channels != init_channels * 2:
            if stride == 1:
                output_padding = 0
            else:
                output_padding = 1
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.in_channels, init_channels*2,
                                   kernel_size=1, stride=stride,
                                   bias=False, output_padding=output_padding), #1),
                nn.BatchNorm2d(init_channels*2),
            )
        layers = []
        for i in range(1, num_layer):
            layers.append(block(self.in_channels, init_channels, 4))
        layers.append(block(self.in_channels, init_channels, 2, stride, upsample))
        self.in_channels = init_channels * 2
        return nn.Sequential(*layers)

    def forward(self, x):
        img = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.avgpool(x)

        x = self.dlayer1(x)
        x = self.dlayer2(x)
        x = self.dlayer3(x)
        x = self.dlayer4(x)

        # Mid
        x = self.uplayer1_1(x)
        x_mid = self.uplayer2_1(x)
        
        # Stage 1
        x_stage1 = self.uplayer_stage_1(x_mid)
        x_seg_out1 = self.conv_seg_out_1(x_stage1)
        x_hands1 = x_seg_out1
        if self.with_energy:
            x_e_out1 = self.sigmoid(self.conv_e_out_1(x_stage1))
        
        if self.n_stages == 1:
            if self.with_energy:
                return x_hands1, x_e_out1
            else:
                return x_hands1
                
        # stage2
        x_mid2 = self.uplayer1_2(x_mid)
        if self.with_energy:
            x = torch.cat([x_mid2, x_seg_out1, x_e_out1], dim = 1)
        else:
            x = torch.cat([x_mid2, x_seg_out1], dim = 1)
        x = self.post_cat_2(x)
        x = self.bn_2(x)
        x = self.relu(x)
        x = self.uplayer2_2(x)
        x = self.uplayer_stage_2(x)
        
        x_seg_out2 = self.conv_seg_out_2(x)
        x_hands2 = x_seg_out2
        if self.with_energy:
            x_e_out2 = self.sigmoid(self.conv_e_out_2(x))
        
        if self.n_stages == 2:
            if self.with_energy:
                return x_hands2, x_e_out2
            else:
                return x_hands2
        else:
            if self.with_energy:
                return x_hands1, x_e_out1, x_hands2, x_e_out2
            else:
                return x_hands1, x_hands2

def CSM_baseline(**kwargs):
    return CSM(Bottleneck, DeconvBottleneck, [3, 3, 3, 3], **kwargs)
