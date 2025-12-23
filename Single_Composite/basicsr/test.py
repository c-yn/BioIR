import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange


###########################################################
####borrowed from 
# Improving image restoration by revisiting global information aggregation
# only used for deblurring tasks (following previous works) in channel attention unit

#######################################################################


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Feed-forward network
class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):

        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        # self.patch_size = 8

        # self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)

        # self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.down = nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim)
        self.down_conv = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)
        self.down_dw = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features, bias=bias)
    def forward(self, x):

        x1 = self.down(x)
        x1 = self.down_conv(x1)
        x1 = self.down_dw(x1)
        x1 = F.interpolate(x1, size=x.shape[2:], mode='bilinear')

        x2 = self.project_in(x)
        x2 = self.dwconv(x2)
        # x = self.project_in(x)

        # x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x2) * x1
        x = self.project_out(x)

        return x

# class StripConv(nn.Module):
#     def __init__(self, dim, kernel):
#         super().__init__()
#         self.dim = dim
#         self.kernel = kernel
#         self.padding = kernel // 2
        
#         self.conv = nn.Sequential(
#             nn.Conv2d(dim, dim, kernel_size=(1, self.kernel), padding=(0, self.padding), groups=dim),
#             nn.Conv2d(dim, dim, kernel_size=(self.kernel, 1), padding=(self.padding, 0), groups=dim),
#         )
        
#     def forward(self, x):
#         return self.conv(x)


class Dyna(nn.Module):
    def __init__(self, dim, kernel, stride=1, group=1):
        super(Dyna, self).__init__()
        # self.stride = stride
        self.kernel = kernel
        self.group = group

        self.conv = nn.Conv2d(dim, group*kernel**2, kernel_size=1, stride=1, bias=False)
        # self.bn = nn.BatchNorm2d(group*kernel**2)

        self.act = nn.Tanh()
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # self.lamb_l = nn.Parameter(torch.zeros(dim), requires_grad=True)
        # self.lamb_h = nn.Parameter(torch.zeros(dim), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel//2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        shortcut = x
        filter = self.gap(x)
        filter = self.conv(filter)

        n, c, h, w = x.shape  
        x = F.unfold(self.pad(x), kernel_size=self.kernel).reshape(n, self.group, c//self.group, self.kernel**2, h*w)

        n,c1,p,q = filter.shape
        filter = filter.reshape(n, c1//self.kernel**2, self.kernel**2, p*q).unsqueeze(2)
        filter = self.act(filter)

        out = torch.sum(x * filter, dim=3).reshape(n, c, h, w)
        return out + shortcut


class Branch(nn.Module):
    def __init__(self, dim, kernel):
        super(Branch, self).__init__()

        self.dim = dim
        self.kernel = kernel
        self.dyna_dim = 2

        self.process = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.Conv2d(dim, dim, kernel_size=kernel, padding=kernel//2, groups=dim),
            nn.Conv2d(dim, self.dyna_dim, kernel_size=1),
            Dyna(dim=self.dyna_dim, kernel=3),
            nn.Conv2d(self.dyna_dim, dim, kernel_size=1),
        )

    def forward(self, x):
        return x + self.process(x)

##########################################################################
## Star module
class AttModule(nn.Module):
    def __init__(self, dim, bias):
        super(AttModule, self).__init__()

        self.dim = dim
        self.to_hidden = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.down = nn.Conv2d(dim, dim, kernel_size=2, stride=2, groups=dim)
        self.down_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        # self.down_dw = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.dim_branch = dim

        self.branch1 = Branch(dim=dim, kernel=5)
        # self.branch2 = Branch(dim=self.dim_branch, kernel=5)
        # self.branch3 = Branch(dim=self.dim_branch, kernel=5)
        # self.branch4 = Branch(dim=self.dim_branch, kernel=5)

        # self.conv = nn.Conv2d(dim, dim, 1)

    def forward(self, x):

        x1 = self.down(x)
        # x1 = x1)
        # x1 = self.down_dw(x1)

        x1_b1 = self.branch1(x1)
        # x1_b2 = self.branch2(x1_b1 * x1[1])
        # x1_b3 = self.branch3(x1_b2 * x1[2])
        # x1_b4 = self.branch3(x1_b3 * x1[3])

        x1 = self.down_conv(x1_b1)

        x1 = F.interpolate(x1, size=x.shape[2:], mode='bilinear')

        x2 = self.to_hidden(x)
        x2 = self.to_hidden_dw(x2)

        out = x1 * x2

        out = self.project_out(out)

        return out

##########################################################################
class AttBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='WithBias'):
        super(AttBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = AttModule(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x


class Fuse(nn.Module):
    def __init__(self, n_feat):
        super(Fuse, self).__init__()
        self.n_feat = n_feat
        self.conv = nn.Conv2d(n_feat * 2, n_feat, 1, 1, 0)

    def forward(self, enc, dnc):
        x = self.conv(torch.cat((enc, dnc), dim=1))
        return x


##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat * 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, n_feat // 2, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)


##########################################################################
##---------- StarIR -----------------------
class CDD(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=32,
                 num_blocks=[2,3,6],
                 num_refinement_blocks=4,
                 ffn_expansion_factor=3,
                 bias=False,
                 ):
        super(CDD, self).__init__()
        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        self.encoder_level1 = nn.Sequential(*[
            AttBlock(dim=dim, ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for i in range(num_blocks[0])])

        self.down1_2 = Downsample(dim)

        self.encoder_level2 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias,) 
            for i in range(num_blocks[1])])

        self.down2_3 = Downsample(int(dim * 2 ** 1))

        self.encoder_level3 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for i in range(num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 2), ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim * 2 ** 2))

        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias)

        self.decoder_level2 = nn.Sequential(*[
            AttBlock(dim=int(dim * 2 ** 1), ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim * 2 ** 1))

        self.decoder_level1 = nn.Sequential(*[
            AttBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[
            AttBlock(dim=int(dim), ffn_expansion_factor=ffn_expansion_factor, bias=bias) 
            for i in range(num_refinement_blocks)])

        self.fuse2 = Fuse(dim * 2)
        self.fuse1 = Fuse(dim)
        self.output = nn.Conv2d(int(dim), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        # self.dual_pixel_task = dual_pixel_task
        # if self.dual_pixel_task:
        #     self.skip_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
    def forward(self, inp_img):
        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        inp_dec_level2 = self.up3_2(out_dec_level3)

        inp_dec_level2 = self.fuse2(inp_dec_level2, out_enc_level2)

        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        inp_dec_level1 = self.up2_1(out_dec_level2)

        inp_dec_level1 = self.fuse1(inp_dec_level1, out_enc_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        out_dec_level1 = self.refinement(out_dec_level1)
        
        # if self.dual_pixel_task:
        #     out_dec_level1 = out_dec_level1 + self.skip_conv(inp_enc_level1)
        #     out_dec_level1 = self.output(out_dec_level1)
        # else:
        out_dec_level1 = self.output(out_dec_level1) + inp_img

        return out_dec_level1
    
if __name__ == '__main__':
    model = CDD()
    # print(model)
    import numpy as np
    from ptflops import get_model_complexity_info

    macs, _ = get_model_complexity_info(model, (3,256,256), as_strings=True, print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    para_num_ = sum([np.prod(p.size()) for p in model.parameters()]) / 1000000.
    para_num = sum(p.numel() for p in model.parameters()) / 1000000.
    print('total parameters is %.2fM' % (para_num))