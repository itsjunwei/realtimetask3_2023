import torch
import torch.nn as nn
import torch.nn.functional as F
from model_utils import *
from thop import profile, clever_format
from conformer.encoder import ConformerBlock
import torchinfo

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):
        x = input
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ResLayer(nn.Module):
    """Initialize a ResNet layer"""
    def __init__(self, in_channels, out_channels):
        super(ResLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3), stride=(1,1), 
                               padding=(1,1), bias=False)

        self.bn1 = nn.BatchNorm2d(num_features=out_channels)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3,3), stride=(1,1),
                               padding=(1,1), bias=False)

        self.bn2 = nn.BatchNorm2d(num_features=out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)
        nn.init.constant_(self.bn2.weight, 0)        
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu_(out)
        out = F.dropout(out, p=0.1, training=self.training)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + self.shortcut(x)
        out = F.relu_(out)
        return out

class BottleNeckDSC(nn.Module):
    def __init__(self, in_channels, out_channels,
                 stride=None, btn_reduction=4):
        super().__init__()
        
        if btn_reduction is None:
            btn_reduction = 4
        in_width = int(in_channels/btn_reduction)
        
        self.in_btl = nn.Conv2d(in_channels=in_channels,
                               out_channels=in_width,
                               kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn1 = nn.BatchNorm2d(in_width)
        
        self.dpth_conv = nn.Conv2d(in_channels=in_width,
                                   out_channels=in_width,
                                   kernel_size=(3,3), stride=(1,1),
                                   padding=(1,1), groups=in_width, bias=False)
        self.pnt_conv = nn.Conv2d(in_channels=in_width,
                                  out_channels=in_width,
                                  kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn2 = nn.BatchNorm2d(in_width)
        
        self.out_btl = nn.Conv2d(in_channels=in_width,
                                 out_channels=out_channels,
                                 kernel_size=(1,1), stride=(1,1), bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        
        self.stride = stride
        if self.stride == 2:
            self.downsample = nn.Sequential(
                AvgMaxPool(poolsize=(2,2)),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels,
                          out_channels=out_channels,
                          kernel_size=(1,1), stride=(1,1), bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.init_weights()
        
    def init_weights(self):
        init_layer(self.in_btl)
        init_bn(self.bn1)

        init_layer(self.dpth_conv)
        init_layer(self.pnt_conv)
        init_bn(self.bn2)

        init_layer(self.out_btl)
        init_bn(self.bn3)
        nn.init.constant_(self.bn3.weight, 0)

    def forward(self, x):
        identity = x

        if self.stride == 2:
            x = F.avg_pool2d(x, kernel_size=(2, 2))

        out = self.in_btl(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dpth_conv(out)
        out = self.pnt_conv(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = F.dropout(out, p=0.1, training=self.training)

        out = self.out_btl(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out

class RNet14(nn.Module):
    def __init__(self, input_shape, output_shape,
                 resfilters = [32, 64, 128], 
                 p_dropout: float = 0.0, use_conformer=False,
                 use_selayers = False, gru_size = 128, verbose=False,
                 **kwargs):
        super().__init__()
        
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.p_dropout = p_dropout
        self.resfilters = resfilters
        self.use_selayers = use_selayers
        self.gru_size = gru_size
        self.verbose = verbose

        self.use_conformer = use_conformer

        self.conv_block1 = ConvBlock(in_channels=self.input_shape[1], 
                                     out_channels=self.resfilters[0])
        
        self.reslayer1 = ResLayer(in_channels=self.resfilters[0],
                                  out_channels=self.resfilters[0])
        self.reslayer2 = ResLayer(in_channels=self.resfilters[0],
                                  out_channels=self.resfilters[0])
        self.avgmaxpool1 = AvgMaxPool(poolsize=(2,4))
        
        self.reslayer3 = ResLayer(in_channels=self.resfilters[0],
                                  out_channels=self.resfilters[1])
        self.reslayer4 = ResLayer(in_channels=self.resfilters[1],
                                  out_channels=self.resfilters[1])
        self.avgmaxpool2 = AvgMaxPool(poolsize=(2,4))
        
        self.reslayer5 = ResLayer(in_channels=self.resfilters[1],
                                  out_channels=self.resfilters[2])
        self.reslayer6 = ResLayer(in_channels=self.resfilters[2],
                                  out_channels=self.resfilters[2])
        
        if self.use_conformer:
            self.conf1 = ConformerBlock(encoder_dim=self.resfilters[2], conv_kernel_size=31)
            self.conf2 = ConformerBlock(encoder_dim=self.resfilters[2], conv_kernel_size=31)
            self.fc_size = self.resfilters[2]
            self.fc2_size = 128
        else:
            self.gru = nn.GRU(input_size=self.resfilters[2], hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
            self.fc_size = self.gru_size
            self.fc2_size = 128
            
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(self.fc_size, 
                             self.fc2_size, bias=True)
        self.leaky = nn.LeakyReLU()
        
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.fc2_size, 
                             self.output_shape[-1], bias=True)
        
    def forward(self, x):
        """Input: Input x: (batch_size, n_channels, n_timesteps, n_features)"""

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        x = F.dropout(x, p=self.p_dropout, training=self.training, inplace=True)

        x = self.reslayer1(x)
        x = self.reslayer2(x)
        x = self.avgmaxpool1(x)
        
        x = self.reslayer3(x)
        x = self.reslayer4(x)
        x = self.avgmaxpool2(x)
        
        x = self.reslayer5(x)
        x = self.reslayer6(x)

        if self.verbose: print("After Res6 : {}".format(x.shape))

        x1 = torch.mean(x, dim=3)
        (x2, _) = torch.max(x, dim=3)
        x = x1+x2

        x = x.transpose(1,2)

        if self.use_conformer:
            x = self.conf1(x)
            x = self.conf2(x)
        else:
            x, _ = self.gru(x)
            x = torch.tanh(x)
            x = x[:, :, x.shape[-1]//2:] * x[:, :, :x.shape[-1]//2]
            if self.verbose: print("After GRU : {}".format(x.shape))

        x = self.leaky(self.fc1(self.dropout1(x)))
        x = torch.tanh(self.fc2(self.dropout2(x)))

        return x

class ResNet18(nn.Module):
    def __init__(self, input_shape, output_shape,
                 resfilters = [64, 128, 256], 
                 p_dropout: float = 0.0, use_conformer=False,
                 use_selayers = False, gru_size = 128, verbose=False,
                 **kwargs):
        super().__init__()
        """
        :param n_input_channels: Number of input channels.
        :param p_dropout: Dropout probability.
        :param pretrained: If True, load pretrained model.
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        self.p_dropout = p_dropout
        self.resfilters = resfilters
        self.use_selayers = use_selayers
        self.gru_size = gru_size
        self.verbose = verbose

        self.use_conformer = use_conformer

        self.conv_block1 = ConvBlock(in_channels=self.input_shape[1], 
                                     out_channels=32)
        if self.use_selayers:
            self.stemsqex = ChannelSpatialSELayer(num_channels=32)
        self.pool1 = AvgMaxPool((1,2))

        self.resnetlayer1 = BottleNeckDSC(in_channels=32,
                                          out_channels=self.resfilters[0],
                                          stride=None)
        if self.use_selayers:
            self.sqex1 = ChannelSpatialSELayer(num_channels=self.resfilters[0])

        self.resnetlayer2 = BottleNeckDSC(in_channels=self.resfilters[0],
                                          out_channels=self.resfilters[1],
                                          stride=2)
        if self.use_selayers:
            self.sqex2 = ChannelSpatialSELayer(num_channels=self.resfilters[1])
        self.pool2 = AvgMaxPool((1,2))

        self.resnetlayer3 = BottleNeckDSC(in_channels=self.resfilters[1],
                                          out_channels=self.resfilters[2],
                                          stride=2)
        if self.use_selayers:
            self.sqex3 = ChannelSpatialSELayer(num_channels=self.resfilters[2])
        
        if self.use_conformer:
            self.conf1 = ConformerBlock(encoder_dim=self.resfilters[2], conv_kernel_size=31)
            self.conf2 = ConformerBlock(encoder_dim=self.resfilters[2], conv_kernel_size=31)
            self.fc_size = self.resfilters[2]
        else:
            self.gru = nn.GRU(input_size=self.resfilters[2], hidden_size=self.gru_size,
                              num_layers=2, batch_first=True, bidirectional=False, dropout=0.2)
            self.fc_size = self.gru_size
            self.fc2_size = 128
        
        
        self.dropout1 = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(self.fc_size, 
                             self.fc2_size, bias=True)
        # self.leaky = nn.LeakyReLU()
        self.swish = nn.SiLU()
        
        self.dropout2 = nn.Dropout(p=0.2)
        self.fc2 = nn.Linear(self.fc2_size, 
                             self.output_shape[-1], bias=True)


    def forward(self, x):
        """
        Input: Input x: (batch_size, n_channels, n_timesteps, n_features)"""

        x = self.conv_block1(x, pool_size=(2, 2), pool_type="avg")
        if self.use_selayers:
            x = self.stemsqex(x)
        x = F.dropout(x, p=self.p_dropout, training=self.training, inplace=True)

        x = self.resnetlayer1(x)
        if self.use_selayers:
            x = self.sqex1(x)
        x = self.pool1(x)

        x = self.resnetlayer2(x)
        if self.use_selayers:
            x = self.sqex2(x)
        x = self.pool2(x)

        if self.verbose: print("Before Res3 : {}".format(x.shape))
        x = self.resnetlayer3(x)
        if self.use_selayers:
            x = self.sqex3(x)
        if self.verbose: print("After Res3 : {}".format(x.shape))

        x1 = torch.mean(x, dim=3)
        (x2, _) = torch.max(x, dim=3)
        x = x1+x2

        x = x.transpose(1,2)

        if self.use_conformer:
            x = self.conf1(x)
            x = self.conf2(x)
        else:
            x, _ = self.gru(x)
            x = torch.tanh(x)
            if self.verbose: print("After GRU : {}".format(x.shape))

        x = self.swish(self.fc1(self.dropout1(x)))
        x = torch.tanh(self.fc2(self.dropout2(x)))

        return x

if __name__ == "__main__":
    input_feature_shape = (1, 7, 80, 191) # SALSA-Lite input shape
    output_feature_shape = (1, 10, 117)

    model = ResNet18(input_shape=input_feature_shape,
                     output_shape=output_feature_shape,
                    #  use_selayers=True,
                     verbose=True)

    # model = RNet14(input_shape=input_feature_shape,
    #                output_shape=output_feature_shape,
    #             #    resfilters=[64,128,256],
    #                use_conformer=False, verbose=True)
    print(model)

    x = torch.rand((input_feature_shape), device=torch.device("cpu"))
    y = model(x)

    model_profile = torchinfo.summary(model, input_size=input_feature_shape)
    print('MACC:\t \t %.3f' %  (model_profile.total_mult_adds/1e6), 'M')
    print('Memory:\t \t %.3f' %  (model_profile.total_params/1e3), 'K\n')
