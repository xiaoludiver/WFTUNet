import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numpy as np
import cv2
from pytorch_wavelets import DWTForward, DWTInverse  # (or import DWT, IDWT)
from HAB import HAB, WithBias_LayerNorm, ChannelTransformerBlock


##########################################################################
# Basic modules
def conv(in_channels, out_channels, kernel_size, bias=False, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer


def default_conv(in_channels, out_channels, kernel_size, stride=1, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), stride=stride, bias=bias)


class ResBlock(nn.Module):
    def __init__(
            self, conv, n_feats, kernel_size,
            bias=True, bn=False, act=nn.PReLU(), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            if i == 0:
                m.append(conv(n_feats, 64, kernel_size, bias=bias))
            else:
                m.append(conv(64, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


##########################################################################

## Compute inter-stage features
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size, bias):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 1, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x1 = x1 + x
        return x1, img


class mergeblock(nn.Module):
    def __init__(self, n_feat, kernel_size, bias, subspace_dim=16):
        super(mergeblock, self).__init__()
        self.conv_block = conv(n_feat * 2, n_feat, kernel_size, bias=bias)
        self.num_subspace = subspace_dim
        self.subnet = conv(n_feat * 2, self.num_subspace, kernel_size, bias=bias)

    def forward(self, x, bridge):
        out = torch.cat([x, bridge], 1)
        b_, c_, h_, w_ = bridge.shape
        sub = self.subnet(out)
        V_t = sub.view(b_, self.num_subspace, h_ * w_)
        V_t = V_t / (1e-6 + torch.abs(V_t).sum(axis=2, keepdims=True))
        V = V_t.permute(0, 2, 1)
        mat = torch.matmul(V_t, V)
        mat_inv = torch.inverse(mat)
        project_mat = torch.matmul(mat_inv, V_t)
        bridge_ = bridge.view(b_, c_, h_ * w_)
        project_feature = torch.matmul(project_mat, bridge_.permute(0, 2, 1))
        bridge = torch.matmul(V, project_feature).permute(0, 2, 1).view(b_, c_, h_, w_)
        out = torch.cat([x, bridge], 1)
        out = self.conv_block(out)
        return out + x




##########################################################################shiyan


class RB(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, ):
        super(RB, self).__init__()
        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)

    def forward(self, x):
        out = self.conv_1(x)
        out_1, out_2 = torch.chunk(out, 2, dim=1)
        out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))
        out += self.identity(x)
        return out

def pad_if_needed(x):
    b, _, h, w = x.shape
    pad = nn.ReplicationPad2d(1)
    if h % 2 == 1 and w % 2 == 1:
        x = pad(x)
        return x[:, :, 0:-1, 0:-1]
    elif h % 2 == 1:
        x = pad(x)
        return x[:, :, 0:-1, :]
    elif w % 2 == 1:
        x = pad(x)
        return x[:, :, :, 0:-1]
    return x


class TwoframeDN(nn.Module):   #第一次小波并后与1/4输入融合 然后输出
    def __init__(self, out_size):
        super(TwoframeDN, self).__init__()
        self.WTF = DWTForward(J=1, mode='zero', wave='db1')
        self.rb = RB(in_size=1, out_size=out_size, relu_slope=0.2)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
    def forward(self, x, x_2):
        x = pad_if_needed(x)
        yl, yh = self.WTF(x)
        yh = yh[0]
        b, c, _, fh, fw = yh.shape
        yh = yh.view(b, -1, fh, fw)
        x_2 = self.rb(x_2)
        stacked = torch.stack([x_2, yl], dim=0)
        yl = torch.mean(stacked, dim=0)
        out = torch.cat((yl, yh), 1)
        return out, yl


class LGFM(nn.Module):   #低频门控融合
    def __init__(self, out_size):
        super(LGFM, self).__init__()
        self.conv = nn.Conv2d(out_size, out_size, kernel_size=3, stride=1, padding=1,
                                groups=1, bias=False)
        self.project_out = nn.Conv2d(out_size, out_size, kernel_size=1, bias=False)
    def forward(self, yl_out, skip):
        yl_out0 = self.conv(yl_out)
        skip = self.conv(skip)
        x = F.gelu(yl_out) * skip
        x = self.project_out(x)
        yl_out = yl_out0 + x

        return yl_out


class TwoframeUP(nn.Module):  # 第一次小波并后与1/4输入融合 然后输出
    def __init__(self, out_size):
        super(TwoframeUP, self).__init__()
        self.WTI = DWTInverse(mode='zero', wave='db1')
        self.LGFM= LGFM(out_size)
    def forward(self, x,skip):
        b, c, fh, fw = x.shape
        yl_out = x[:, :c // 4, :, :]
        yh_out = x[:, c // 4:, :, :].view(b, -1, 3, fh, fw)
        yl_out = self.LGFM(yl_out, skip)
        out = self.WTI((yl_out, [yh_out]))
        out = pad_if_needed(out)
        return out

##########################################################################

class Encoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth,csff):
        super(Encoder, self).__init__()

        self.depth = depth

        self.down1 = UNetConvBlock1(in_size=n_feat, out_size=n_feat, downsample=TwoframeDN(out_size=n_feat), relu_slope=0.2,
                                   use_csff=csff, use_HIN=True ,)
        self.down2 = UNetConvBlock2(in_size=n_feat *4, out_size=n_feat *4,
                                   downsample=TwoframeDN(out_size=n_feat*4), relu_slope=0.2, use_csff=csff, use_HIN=True)

        self.bottom = UNetConvBlock3(in_size=n_feat * 16, out_size=n_feat * 16,
                                    downsample=False, relu_slope=0.2, use_csff=csff, use_HIN=True)

    def forward(self, x,x_2 ,x_4, encoder_outs=None, decoder_outs=None ):
        res = []
        skip = []

        if encoder_outs is not None and decoder_outs is not None:
            x, x_up1, ll1 = self.down1(x,x_2, encoder_outs[0], decoder_outs[1] )
            res.append(x_up1)
            x, x_up2, ll2 = self.down2(x, x_4, encoder_outs[1], decoder_outs[0])
            res.append(x_up2)
            x = self.bottom(x)
            skip.append(ll1)
            skip.append(ll2)
        else:
            x, x_up1, ll1 = self.down1(x, x_2)
            res.append(x_up1)
            x, x_up2,ll2 = self.down2(x, x_4)
            res.append(x_up2)
            skip.append(ll1)
            skip.append(ll2)

            x = self.bottom(x)

        return res, x, skip

class UNetConvBlock1(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock1, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN


    def forward(self, x, x_2 ,enc=None, dec=None,):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down , LL= self.downsample(out,x_2)
            return out_down, out,LL
        else:
            return out
class UNetConvBlock2(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock2, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff
        self.hab = HAB(LayerNorm_type=WithBias_LayerNorm,dim=out_size//2)
        self.conv_1 = nn.Conv2d(in_size, out_size//2, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(out_size//2, out_size, kernel_size=3, padding=1, bias=True)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN


    def forward(self, x, x_2 ,enc=None, dec=None,):
        out = self.conv_1(x)
        out = self.hab(out)
        out = self.conv_2(out)
        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down, LL = self.downsample(out, x_2)
            return out_down, out, LL
        else:
            return out
class UNetConvBlock3(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock3, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size//2, kernel_size=3, padding=1, bias=True)
        self.conv_2 = nn.Conv2d(out_size // 2, out_size, kernel_size=3, padding=1, bias=True)
        self.hab = HAB(LayerNorm_type=WithBias_LayerNorm, dim=out_size//2)
        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)
        out = self.hab(out)
        out = self.conv_2(out)
        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out
class UNetConvBlock4(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock4, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(in_size, out_size, 3, 1, 1)
            self.phi = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.gamma = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size // 2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            skip_ = F.leaky_relu(self.csff_enc(enc) + self.csff_dec(dec), 0.1, inplace=True)
            out = out * torch.sigmoid(self.phi(skip_)) + self.gamma(skip_) + out
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock1(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock1, self).__init__()
        self.up = TwoframeUP(out_size)
        self.conv_block = UNetConvBlock3(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge, skip):
        up = self.up(x, skip)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class UNetUpBlock2(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock2, self).__init__()
        self.up = TwoframeUP(out_size)
        self.conv_block = UNetConvBlock4(out_size * 2, out_size, False, relu_slope)

    def forward(self, x, bridge, skip):
        up = self.up(x, skip)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Decoder(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, scale_unetfeats , depth):
        super(Decoder, self).__init__()

        self.depth = depth


        self.up2 = UNetUpBlock1(in_size=n_feat  * 16, out_size=n_feat *4, relu_slope=0.2)
        self.up1 = UNetUpBlock2(in_size=n_feat*4, out_size=n_feat, relu_slope=0.2)


        self.skip_conv2 = nn.Conv2d(n_feat * 4, n_feat *4, 3, 1, 1)
        self.skip_conv1 = nn.Conv2d(n_feat , n_feat, 3, 1, 1)

    def forward(self, x, bridges, skip):
        res = []

        x1 = self.up2(x, self.skip_conv2(bridges[-1]), skip[-1])
        res.append(x1)
        x = self.up1(x1, self.skip_conv1(bridges[-2]), skip[-2])
        res.append(x)
        return res

# class LLATT(nn.Module):
#     def __init__(self, n_feat, ):
#         super(LLATT, self).__init__()
#
#         self.ll1 = ChannelTransformerBlock(dim=n_feat, num_heads=2, ffn_expansion_factor=2.66, bias=False,
#                                     LayerNorm_type=WithBias_LayerNorm)
#         self.ll2 = ChannelTransformerBlock(dim=n_feat*4, num_heads=2, ffn_expansion_factor=2.66, bias=False,
#                                     LayerNorm_type=WithBias_LayerNorm)
#
#     def forward(self, x):
#         skip = []
#         x1 = self.ll1(x[0])
#         skip.append(x1)
#         x2 = self.ll2(x[1])
#         skip.append(x2)
#         return skip
##########################################################################


##########################################################################

class WFTUNet(nn.Module):
    def __init__(self, in_c=1, out_c=1, n_feat=32, scale_unetfeats=16, scale_orsnetfeats=32, num_cab=8, kernel_size=3,
                 reduction=4, bias=False):
        super(WFTUNet, self).__init__()
        # Extract Shallow Features
        act = nn.PReLU()
        self.shallow_feat1 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.shallow_feat2 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.shallow_feat3 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.shallow_feat4 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.shallow_feat5 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.shallow_feat6 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))
        self.shallow_feat7 = nn.Sequential(conv(in_c, n_feat, kernel_size, bias=bias))

        # Gradient Descent Module (GDM)
        self.r0 = nn.Parameter(torch.Tensor([0.5]))
        self.r1 = nn.Parameter(torch.Tensor([0.5]))
        self.r2 = nn.Parameter(torch.Tensor([0.5]))
        self.r3 = nn.Parameter(torch.Tensor([0.5]))
        self.r4 = nn.Parameter(torch.Tensor([0.5]))
        self.r5 = nn.Parameter(torch.Tensor([0.5]))
        self.r6 = nn.Parameter(torch.Tensor([0.5]))
        self.lam0 = nn.Parameter(torch.Tensor([0.4]))
        self.lam1 = nn.Parameter(torch.Tensor([0.4]))
        self.lam2 = nn.Parameter(torch.Tensor([0.4]))
        self.lam3 = nn.Parameter(torch.Tensor([0.4]))
        self.lam4 = nn.Parameter(torch.Tensor([0.4]))
        self.lam5 = nn.Parameter(torch.Tensor([0.4]))

        # Informative Proximal Mapping Module (IPMM)
        self.stage1_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3, csff=False)
        self.stage1_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.stage2_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3, csff=True)
        self.stage2_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.stage3_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3, csff=True)
        self.stage3_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.stage4_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3, csff=True)
        self.stage4_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.stage5_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3, csff=True)
        self.stage5_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.stage6_encoder = Encoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3, csff=True)
        self.stage6_decoder = Decoder(n_feat, kernel_size, reduction, act, bias, scale_unetfeats, depth=3)

        self.sam12 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge12 = mergeblock(n_feat, 3, True)
        self.sam23 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge23 = mergeblock(n_feat, 3, True)
        self.sam34 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge34 = mergeblock(n_feat, 3, True)
        self.sam45 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge45 = mergeblock(n_feat, 3, True)
        self.sam56 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge56 = mergeblock(n_feat, 3, True)
        self.sam67 = SAM(n_feat, kernel_size=1, bias=bias)
        self.merge67 = mergeblock(n_feat, 3, True)

        self.tail = conv(n_feat, out_c, kernel_size, bias=bias)
        # self.llattention = LLATT(n_feat)

    def forward(self, img):
        ##-------------------------------------------
        ##-------------- Stage 1---------------------
        ##-------------------------------------------
        ## GDM

        x1_img = img
        # PMM
        x1_2 = F.interpolate(x1_img, scale_factor=0.5)
        x1_4 = F.interpolate(x1_2, scale_factor=0.5)
        x1 = self.shallow_feat1(x1_img)
        feat1, feat_fin1, skip = self.stage1_encoder(x1 ,x1_2 ,x1_4)
        # skip = self.llattention(skip)
        res1 = self.stage1_decoder(feat_fin1, feat1,skip)
        x2_samfeats, stage1_img = self.sam12(res1[-1], x1_img)

        ##-------------------------------------------
        ##-------------- Stage 2---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_2 = stage1_img - img
        x2_img = stage1_img - self.r1 * phixsy_2 + self.lam0 * (stage1_img - img)
        ## PMM
        x2_2 = F.interpolate(x2_img, scale_factor=0.5)
        x2_4 = F.interpolate(x2_2, scale_factor=0.5)
        x2 = self.shallow_feat2(x2_img)
        x2_cat = self.merge12(x2, x2_samfeats)  # Features interaction on the largest scale
        feat2, feat_fin2,skip = self.stage2_encoder(x2_cat, x2_2,x2_4, feat1, res1)  # Features interaction within encoder
        # skip = self.llattention(skip)
        res2 = self.stage2_decoder(feat_fin2, feat2, skip)  # Features interaction within decoder
        x3_samfeats, stage2_img = self.sam23(res2[-1], x2_img)

        ##-------------------------------------------
        ##-------------- Stage 3---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_3 = stage2_img - img
        x3_img = stage2_img - self.r2 * phixsy_3 + self.lam1 * (stage2_img - stage1_img)
        ## PMM
        x3_2 = F.interpolate(x3_img, scale_factor=0.5)
        x3_4 = F.interpolate(x3_2, scale_factor=0.5)
        x3 = self.shallow_feat3(x3_img)
        x3_cat = self.merge23(x3, x3_samfeats)
        feat3, feat_fin3, skip = self.stage3_encoder(x3_cat,x3_2,x3_4, feat2, res2)
        # skip = self.llattention(skip)
        res3 = self.stage3_decoder(feat_fin3, feat3, skip)
        x4_samfeats, stage3_img = self.sam34(res3[-1], x3_img)

        ##-------------------------------------------
        ##-------------- Stage 4---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_4 = stage3_img - img
        x4_img = stage3_img - self.r3 * phixsy_4 + self.lam2 * (stage3_img - stage2_img)
        ## PMM
        x4_2 = F.interpolate(x4_img, scale_factor=0.5)
        x4_4 = F.interpolate(x4_2, scale_factor=0.5)
        x4 = self.shallow_feat4(x4_img)
        x4_cat = self.merge34(x4, x4_samfeats)
        feat4, feat_fin4,skip = self.stage4_encoder(x4_cat, x4_2, x4_4, feat3, res3)
        # skip = self.llattention(skip)
        res4 = self.stage4_decoder(feat_fin4, feat4, skip)
        x5_samfeats, stage4_img = self.sam45(res4[-1], x4_img)

        ##-------------------------------------------
        ##-------------- Stage 5---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_5 = stage4_img - img
        x5_img = stage4_img - self.r4 * phixsy_5 + self.lam3 * (stage4_img - stage3_img)
        ## PMM
        x5_2 = F.interpolate(x5_img, scale_factor=0.5)
        x5_4 = F.interpolate(x5_2, scale_factor=0.5)
        x5 = self.shallow_feat5(x5_img)
        x5_cat = self.merge45(x5, x5_samfeats)
        feat5, feat_fin5, skip = self.stage5_encoder(x5_cat,x5_2,x5_4, feat4, res4)
        # skip = self.llattention(skip)
        res5 = self.stage5_decoder(feat_fin5, feat5, skip)
        x6_samfeats, stage5_img = self.sam56(res5[-1], x5_img)

        ##-------------------------------------------
        ##-------------- Stage 6---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_6 = stage5_img - img
        x6_img = stage5_img - self.r5 * phixsy_6 + self.lam4 * (stage5_img - stage4_img)
        ## PMM
        x6_2 = F.interpolate(x6_img, scale_factor=0.5)
        x6_4 = F.interpolate(x6_2, scale_factor=0.5)
        x6 = self.shallow_feat6(x6_img)
        x6_cat = self.merge56(x6, x6_samfeats)
        feat6, feat_fin6, skip = self.stage6_encoder(x6_cat, x6_2, x6_4, feat5, res5)
        # skip = self.llattention(skip)
        res6 = self.stage6_decoder(feat_fin6, feat6, skip)
        x7_samfeats, stage6_img = self.sam67(res6[-1], x6_img)

        ##-------------------------------------------
        ##-------------- Stage 7---------------------
        ##-------------------------------------------
        ## GDM
        phixsy_7 = stage6_img - img
        x7_img = stage6_img - self.r6 * phixsy_7 + self.lam5 * (stage6_img - stage5_img)
        ## PMM
        x7 = self.shallow_feat7(x7_img)
        x7_cat = self.merge67(x7, x7_samfeats)
        stage7_img = self.tail(x7_cat) + img

        return [stage7_img, stage6_img, stage5_img, stage4_img, stage3_img, stage2_img, stage1_img]

from thop import profile
if __name__ == '__main__':
    input = torch.randn(1, 1, 128, 128).cuda()
    m = WFTUNet(in_c=1, out_c=1).cuda()
    out = m(input)
    out = out[0]
    print(out.shape)
    # 创建一个符合模型输入要求的张量，包括批次维度
    input_tensor = torch.randn(1, 1, 128, 128).cuda()  # 这里的 1 表示批次大小为1

    # 使用 thop 来计算 FLOPS
    flops, params = profile(m, inputs=(input_tensor,), verbose=True)  # verbose=False 可以关闭详细输出，只保留结果
    flops=flops/1e9
    params = params/1e6
    print(f"FLOPS: {flops}, Params: {params}")