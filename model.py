import torch.nn as nn
import torch
import torch.nn.functional as F
import numbers
import math


def fmm(fb_feat, bg_feat):
    assert (fb_feat.size()[:2] == bg_feat.size()[:2])
    size = fb_feat.size()
    _, bg_std = calc_mean_std(bg_feat)
    _, fb_std = calc_mean_std(fb_feat)
    feat = fb_feat * (bg_std.expand(size) / fb_std.expand(size))
    return feat

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

class Self_Attn(nn.Module):
    """ Self attention Layer"""
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input):
        """
            inputs :
                input : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        batchsize, C, width, height = input.size()
        proj_query = self.query_conv(input).view(batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(input).view(batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(input).view(batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batchsize, C, width, height)

        out = self.gamma * out + input
        return out

class FB_FMM(nn.Module):
    def __init__(self, in_dim):
        super(FB_FMM, self).__init__()
        self.SA = Self_Attn(in_dim)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input, mask):
        input = self.SA(input)
        batchsize, C, height, width = input.size()
        _,_,mh,_ = mask.size()
        if height != mh:
            mask = torch.round(F.avg_pool2d(mask,2,stride=mh//height))
        reverse_mask = -1*(mask-1)
        feature_b = reverse_mask * input
        feature_f = mask * input

        proj_query = self.query_conv(feature_f).view(batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(feature_b).view(batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # B X (N) X (N)
        proj_value = self.value_conv(feature_b).view(batchsize, -1, width * height)  # B X C X N

        sw_bg = torch.bmm(proj_value, attention.permute(0, 2, 1))
        sw_bg = sw_bg.view(batchsize, C, height, width)
        
        out = input + self.gamma * fmm(feature_f, sw_bg)
        return out

class ConvBlock(nn.Module):
   def __init__(self, in_planes, out_planes, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
      super(ConvBlock, self).__init__()
      self.conv2d = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding)
      self.bn = nn.BatchNorm2d(out_planes)
   def forward(self, x):
      return F.relu(self.bn(self.conv2d(x)), inplace=False)

class UpsampleConcat(nn.Module):
    def __init__(self, in_planes, out_planes, scale_factor=2):
        super(UpsampleConcat, self).__init__()
        self.scale_factor = scale_factor
        self.conv2d = nn.Conv2d(in_planes, out_planes,kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.bn = nn.BatchNorm2d(out_planes)
    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)
        return F.relu(self.bn(self.conv2d(F.interpolate(x, scale_factor=self.scale_factor, mode="bilinear", align_corners=True))), inplace=False)

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.Down1 = ConvBlock(4, 64, stride=(2, 2))
        self.Down2 = ConvBlock(64, 128)
        self.Down3 = ConvBlock(128, 256, stride=(2, 2))
        self.Down4 = ConvBlock(256, 256)
        self.Down5 = ConvBlock(256, 512, stride=(2, 2))
        self.Down6 = ConvBlock(512, 512)
        self.Down7 = ConvBlock(512, 512)
        self.Down8 = ConvBlock(512, 512)
        self.Down9 = ConvBlock(512, 256)
        self.FB_FMM = FB_FMM(256)
        self.Up1 = UpsampleConcat(512 + 256, 128)
        self.Up2 = UpsampleConcat(256 + 128, 32)
        self.Up3 = UpsampleConcat(128 + 32, 16)
        self.output = nn.Conv2d(16, 3, kernel_size=(3, 3), padding=1)

    def forward(self, image, mask):
        x = torch.cat((image.clone() - 0.4462414, mask.clone()), 1)
        x = self.Down1(x)
        conv1 = self.Down2(x)
        x = self.Down3(conv1)
        conv2 = self.Down4(x)
        x = self.Down5(conv2)
        x = self.Down6(x)
        x = self.Down7(x)
        conv3 = self.Down8(x)
        x = self.Down9(conv3)
        x = self.FB_FMM(x, mask)
        x = self.Up1(x, conv3)
        x = self.Up2(x, conv2)
        x = self.Up3(x, conv1)
        out = self.output(x)

        return (image + torch.tanh(out)).clamp(0, 1)