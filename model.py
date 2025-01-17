# 3D-UNet model from https://github.com/UdonDa/3D-UNet-PyTorch.

import torch
import torch.nn as nn


def conv_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.Conv3d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def conv_trans_block_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        nn.ConvTranspose3d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.BatchNorm3d(out_dim),
        activation,)


def max_pooling_3d():
    return nn.MaxPool3d(kernel_size=2, stride=2, padding=0)


def conv_block_2_3d(in_dim, out_dim, activation):
    return nn.Sequential(
        conv_block_3d(in_dim, out_dim, activation),
        nn.Conv3d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_dim),)

class UNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_filters):
        super(UNet, self).__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_filters = num_filters
        activation = nn.LeakyReLU(0.2, inplace=True)
        
        # Down sampling
        self.down_1 = conv_block_2_3d(self.in_dim, self.num_filters, activation)
        self.pool_1 = max_pooling_3d()
        self.down_2 = conv_block_2_3d(self.num_filters, self.num_filters * 2, activation)
        self.pool_2 = max_pooling_3d()
        self.down_3 = conv_block_2_3d(self.num_filters * 2, self.num_filters * 4, activation)
        self.pool_3 = max_pooling_3d()
        self.down_4 = conv_block_2_3d(self.num_filters * 4, self.num_filters * 8, activation)
        self.pool_4 = max_pooling_3d()
        self.down_5 = conv_block_2_3d(self.num_filters * 8, self.num_filters * 16, activation)
        self.pool_5 = max_pooling_3d()
        
        # Bridge
        self.bridge = conv_block_2_3d(self.num_filters * 16, self.num_filters * 32, activation)
        
        # Up sampling
        self.trans_1 = conv_trans_block_3d(self.num_filters * 32, self.num_filters * 32, activation)
        self.up_1 = conv_block_2_3d(self.num_filters * 48, self.num_filters * 16, activation)
        self.trans_2 = conv_trans_block_3d(self.num_filters * 16, self.num_filters * 16, activation)
        self.up_2 = conv_block_2_3d(self.num_filters * 24, self.num_filters * 8, activation)
        self.trans_3 = conv_trans_block_3d(self.num_filters * 8, self.num_filters * 8, activation)
        self.up_3 = conv_block_2_3d(self.num_filters * 12, self.num_filters * 4, activation)
        self.trans_4 = conv_trans_block_3d(self.num_filters * 4, self.num_filters * 4, activation)
        self.up_4 = conv_block_2_3d(self.num_filters * 6, self.num_filters * 2, activation)
        self.trans_5 = conv_trans_block_3d(self.num_filters * 2, self.num_filters * 2, activation)
        self.up_5 = conv_block_2_3d(self.num_filters * 3, self.num_filters * 1, activation)
        
        # Output
        self.out = conv_block_3d(self.num_filters, out_dim, nn.Softmax(dim=1))
    
    def forward(self, x):
        x = x.to(self.device)
        # Down sampling
        #print("x", x.shape)
        down_1 = self.down_1(x) # -> [1, 4, 128, 128, 128]
        #print("down_1", down_1.shape)
        pool_1 = self.pool_1(down_1) # -> [1, 4, 64, 64, 64]
        #print("pool_1", pool_1.shape)
        
        down_2 = self.down_2(pool_1) # -> [1, 8, 64, 64, 64]
        #print("down_2", down_2.shape)
        pool_2 = self.pool_2(down_2) # -> [1, 8, 32, 32, 32]
        #print("pool_2", pool_2.shape)
        
        down_3 = self.down_3(pool_2) # -> [1, 16, 32, 32, 32]
        #print("down_3", down_3.shape)
        pool_3 = self.pool_3(down_3) # -> [1, 16, 16, 16, 16]
        #print("pool_3", pool_3.shape)

        down_4 = self.down_4(pool_3) # -> [1, 32, 16, 16, 16]
        #print("down_4", down_4.shape)
        pool_4 = self.pool_4(down_4) # -> [1, 32, 8, 8, 8]
        #print("pool_4", pool_4.shape)

        down_5 = self.down_5(pool_4) # -> [1, 64, 8, 8, 8]
        #print("down_5", down_5.shape)
        pool_5 = self.pool_5(down_5) # -> [1, 64, 4, 4, 4]
        #print("pool_5", pool_5.shape)

        # Bridge
        bridge = self.bridge(pool_5) # -> [1, 128, 4, 4, 4]
        #print("bridge", bridge.shape)

        # Up sampling
        trans_1 = self.trans_1(bridge) # -> [1, 128, 8, 8, 8]
        #print("trans_1", trans_1.shape)
        concat_1 = torch.cat([trans_1, down_5], dim=1) # -> [1, 192, 8, 8, 8]
        up_1 = self.up_1(concat_1) # -> [1, 64, 8, 8, 8]
        #print("up_1", up_1.shape)        

        trans_2 = self.trans_2(up_1) # -> [1, 64, 16, 16, 16]
        #print("trans_2", trans_2.shape)
        concat_2 = torch.cat([trans_2, down_4], dim=1) # -> [1, 96, 16, 16, 16]
        up_2 = self.up_2(concat_2) # -> [1, 32, 16, 16, 16]
        
        trans_3 = self.trans_3(up_2) # -> [1, 32, 32, 32, 32]
        #print("trans_3", trans_3.shape)
        concat_3 = torch.cat([trans_3, down_3], dim=1) # -> [1, 48, 32, 32, 32]
        up_3 = self.up_3(concat_3) # -> [1, 16, 32, 32, 32]
        
        trans_4 = self.trans_4(up_3) # -> [1, 16, 64, 64, 64]
        #print("trans_4", trans_4.shape)
        concat_4 = torch.cat([trans_4, down_2], dim=1) # -> [1, 24, 64, 64, 64]
        up_4 = self.up_4(concat_4) # -> [1, 8, 64, 64, 64]
        
        trans_5 = self.trans_5(up_4) # -> [1, 8, 128, 128, 128]
        #print("trans_5", trans_5.shape)
        concat_5 = torch.cat([trans_5, down_1], dim=1) # -> [1, 12, 128, 128, 128]
        up_5 = self.up_5(concat_5) # -> [1, 4, 128, 128, 128]
        
        # Output
        out = self.out(up_5) # -> [1, 3, 128, 128, 128]
        #print("out", out.shape)
        return out
