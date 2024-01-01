'''by lyuwenyu
'''

import copy
import numpy as np
import torch 
import torch.nn as nn 
from torch.nn.parameter import Parameter
import torch.nn.functional as F 

from .utils import get_activation
from .common import SimFusion_3in, SimFusion_4in, AdvPoolFusion
from .transformer import PyramidPoolAgg, TopBasicLayer, InjectionMultiSum_Auto_pool


from src.core import register

# for test
import sys



__all__ = ['HybridEncoder']

class Conv(nn.Module):
    '''Normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class SimConv(nn.Module):
    '''Normal Conv with ReLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1, bias=False, padding=None):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))


def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result

class ConvNormLayer(nn.Module):
    """
    S1、S2 layer in paper
    """
    def __init__(self, ch_in, ch_out, kernel_size, stride, padding=None, bias=False, act=None):
        super().__init__()
        self.conv = nn.Conv2d(
            ch_in, 
            ch_out, 
            kernel_size, 
            stride, 
            padding=(kernel_size-1)//2 if padding is None else padding, 
            bias=bias)
        self.norm = nn.BatchNorm2d(ch_out)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))


class RepVGGBlock(nn.Module):
    '''RepVGGBlock is a basic rep-style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    '''
    
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, dilation=1, groups=1, padding_mode='zeros', deploy=False, use_se=False):
        super(RepVGGBlock, self).__init__()
        """ Initialization of the class.
        Args:
            in_channels (int): Number of channels in the input image
            out_channels (int): Number of channels produced by the convolution
            kernel_size (int or tuple): Size of the convolving kernel
            stride (int or tuple, optional): Stride of the convolution. Default: 1
            padding (int or tuple, optional): Zero-padding added to both sides of
                the input. Default: 1
            dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
            groups (int, optional): Number of blocked connections from input
                channels to output channels. Default: 1
            padding_mode (string, optional): Default: 'zeros'
            deploy: Whether to be deploy status or training status. Default: False
            use_se: Whether to use se. Default: False
        """
        self.deploy = deploy
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert kernel_size == 3
        assert padding == 1
        
        padding_11 = padding - kernel_size // 2
        
        self.nonlinearity = nn.ReLU()
        
        if use_se:
            raise NotImplementedError("se block not supported yet")
        else:
            self.se = nn.Identity()
        
        if deploy:
            self.rbr_reparam = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=True,
                                         padding_mode=padding_mode)
        
        else:
            self.rbr_identity = nn.BatchNorm2d(
                    num_features=in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                                     stride=stride, padding=padding, groups=groups)
            self.rbr_1x1 = conv_bn(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride,
                                   padding=padding_11, groups=groups)
    
    def forward(self, inputs):
        '''Forward process'''
        if hasattr(self, 'rbr_reparam'):
            return self.nonlinearity(self.se(self.rbr_reparam(inputs)))
        
        if self.rbr_identity is None:
            id_out = 0
        else:
            id_out = self.rbr_identity(inputs)
        
        return self.nonlinearity(self.se(self.rbr_dense(inputs) + self.rbr_1x1(inputs) + id_out))
    
    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid
    
    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return torch.nn.functional.pad(kernel1x1, [1, 1, 1, 1])
    
    def _fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std
    
    def switch_to_deploy(self):
        if hasattr(self, 'rbr_reparam'):
            return
        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(in_channels=self.rbr_dense.conv.in_channels,
                                     out_channels=self.rbr_dense.conv.out_channels,
                                     kernel_size=self.rbr_dense.conv.kernel_size, stride=self.rbr_dense.conv.stride,
                                     padding=self.rbr_dense.conv.padding, dilation=self.rbr_dense.conv.dilation,
                                     groups=self.rbr_dense.conv.groups, bias=True)
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias
        for para in self.parameters():
            para.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')
        self.deploy = True

class RepVggBlock(nn.Module):
    def __init__(self, ch_in, ch_out, act='relu'):
        super().__init__()
        self.ch_in = ch_in
        self.ch_out = ch_out
        self.conv1 = ConvNormLayer(ch_in, ch_out, 3, 1, padding=1, act=None)
        self.conv2 = ConvNormLayer(ch_in, ch_out, 1, 1, padding=0, act=None)
        self.act = nn.Identity() if act is None else get_activation(act) 

    def forward(self, x):
        if hasattr(self, 'conv'):
            y = self.conv(x)
        else:
            y = self.conv1(x) + self.conv2(x)

        return self.act(y)

    def convert_to_deploy(self):
        if not hasattr(self, 'conv'):
            self.conv = nn.Conv2d(self.ch_in, self.ch_out, 3, 1, padding=1)

        kernel, bias = self.get_equivalent_kernel_bias()
        self.conv.weight.data = kernel
        self.conv.bias.data = bias 
        # self.__delattr__('conv1')
        # self.__delattr__('conv2')

    def get_equivalent_kernel_bias(self):
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.conv1)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.conv2)
        
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1), bias3x3 + bias1x1

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: ConvNormLayer):
        if branch is None:
            return 0, 0
        kernel = branch.conv.weight
        running_mean = branch.norm.running_mean
        running_var = branch.norm.running_var
        gamma = branch.norm.weight
        beta = branch.norm.bias
        eps = branch.norm.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

class BottleRep(nn.Module):
    
    def __init__(self, in_channels, out_channels, basic_block=RepVGGBlock, weight=False):
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels != out_channels:
            self.shortcut = False
        else:
            self.shortcut = True
        if weight:
            self.alpha = Parameter(torch.ones(1))
        else:
            self.alpha = 1.0
    
    def forward(self, x):
        outputs = self.conv1(x)
        outputs = self.conv2(outputs)
        return outputs + self.alpha * x if self.shortcut else outputs

class RepBlock(nn.Module):
    '''
        RepBlock is a stage block with rep-style basic block
    '''
    
    def __init__(self, in_channels, out_channels, n=1, block=RepVGGBlock, basic_block=RepVGGBlock):
        super().__init__()
        
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(*(block(out_channels, out_channels) for _ in range(n - 1))) if n > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            n = n // 2
            self.block = nn.Sequential(
                    *(BottleRep(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in
                      range(n - 1))) if n > 1 else None
    
    def forward(self, x):
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x

class CSPRepLayer(nn.Module):
    """
    Fusion module in paper
    """
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_blocks=3,
                 expansion=1.0,
                 bias=None,
                 act="silu"):
        super(CSPRepLayer, self).__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.conv2 = ConvNormLayer(in_channels, hidden_channels, 1, 1, bias=bias, act=act)
        self.bottlenecks = nn.Sequential(*[
            RepVggBlock(hidden_channels, hidden_channels, act=act) for _ in range(num_blocks)
        ])
        if hidden_channels != out_channels:
            self.conv3 = ConvNormLayer(hidden_channels, out_channels, 1, 1, bias=bias, act=act)
        else:
            self.conv3 = nn.Identity()

    def forward(self, x):
        x_1 = self.conv1(x)
        x_1 = self.bottlenecks(x_1)
        x_2 = self.conv2(x)
        return self.conv3(x_1 + x_2)


# transformer
class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu",
                 normalize_before=False):
        super().__init__()
        self.normalize_before = normalize_before

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout, batch_first=True)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation(activation) 

    @staticmethod
    def with_pos_embed(tensor, pos_embed):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        residual = src
        if self.normalize_before:
            src = self.norm1(src)
        q = k = self.with_pos_embed(src, pos_embed)
        src, _ = self.self_attn(q, k, value=src, attn_mask=src_mask)

        src = residual + self.dropout1(src)
        if not self.normalize_before:
            src = self.norm1(src)

        residual = src
        if self.normalize_before:
            src = self.norm2(src)
        src = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = residual + self.dropout2(src)
        if not self.normalize_before:
            src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(num_layers)])
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, src_mask=None, pos_embed=None) -> torch.Tensor:
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=src_mask, pos_embed=pos_embed)

        if self.norm is not None:
            output = self.norm(output)

        return output


@register
class HybridEncoder(nn.Module):
    def __init__(self,
                 in_channels=[512, 1024, 2048],
                 feat_strides=[8, 16, 32],
                 hidden_dim=256,
                 nhead=8,
                 dim_feedforward = 1024,
                 dropout=0.0,
                 enc_act='gelu',
                 use_encoder_idx=[2],
                 num_encoder_layers=1,
                 pe_temperature=10000,
                 expansion=1.0,
                 depth_mult=1.0,
                 act='silu',
                 eval_size=None):
        super().__init__()
        self.in_channels = in_channels
        self.feat_strides = feat_strides
        self.hidden_dim = hidden_dim
        self.use_encoder_idx = use_encoder_idx
        self.num_encoder_layers = num_encoder_layers
        self.pe_temperature = pe_temperature
        self.eval_size = eval_size

        self.out_channels = [hidden_dim for _ in range(len(in_channels))]
        self.out_strides = feat_strides
        
        # channel projection
        # 输入投影层，匹配通道数
        self.input_proj = nn.ModuleList()
        for in_channel in in_channels:
            self.input_proj.append(
                nn.Sequential(
                    nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
                    nn.BatchNorm2d(hidden_dim)
                )
            )

        # encoder transformer
        encoder_layer = TransformerEncoderLayer(
            hidden_dim, 
            nhead=nhead,
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
            activation=enc_act)

        self.encoder = nn.ModuleList([
            TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
        ])

        # top-down fpn
        self.lateral_convs = nn.ModuleList()
        self.fpn_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1, 0, -1):
            self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
            self.fpn_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        # bottom-up pan
        self.downsample_convs = nn.ModuleList()
        self.pan_blocks = nn.ModuleList()
        for _ in range(len(in_channels) - 1):
            self.downsample_convs.append(
                ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
            )
            self.pan_blocks.append(
                CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
            )

        self._reset_parameters()

    def _reset_parameters(self):
        if self.eval_size:
            for idx in self.use_encoder_idx:
                stride = self.feat_strides[idx]
                pos_embed = self.build_2d_sincos_position_embedding(
                    self.eval_size[1] // stride, self.eval_size[0] // stride,
                    self.hidden_dim, self.pe_temperature)
                setattr(self, f'pos_embed{idx}', pos_embed)
                # self.register_buffer(f'pos_embed{idx}', pos_embed)

    @staticmethod
    def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
        '''
        '''
        grid_w = torch.arange(int(w), dtype=torch.float32)
        grid_h = torch.arange(int(h), dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
        assert embed_dim % 4 == 0, \
            'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature ** omega)

        out_w = grid_w.flatten()[..., None] @ omega[None]
        out_h = grid_h.flatten()[..., None] @ omega[None]

        return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

    def forward(self, feats):
        # print('the len of feats = ',len(feats))
        assert len(feats) == len(self.in_channels)
        # sys.exit(0)
        proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
        # print('proj_feats=',proj_fea ts)
        # encoder
        # AIFI
        if self.num_encoder_layers > 0:
            for i, enc_ind in enumerate(self.use_encoder_idx):
                h, w = proj_feats[enc_ind].shape[2:]
                # flatten [B, C, H, W] to [B, HxW, C]
                src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
                if self.training or self.eval_size is None:
                    pos_embed = self.build_2d_sincos_position_embedding(
                        w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
                else:
                    pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

                memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
                proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                # print([x.is_contiguous() for x in proj_feats ])
        # print('after encoder, proj_feats=',proj_feats)
        # broadcasting and fusion
        inner_outs = [proj_feats[-1]]
        for idx in range(len(self.in_channels) - 1, 0, -1):
            feat_heigh = inner_outs[0]
            feat_low = proj_feats[idx - 1]
            feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
            inner_outs[0] = feat_heigh
            upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
            inner_out = self.fpn_blocks[0](torch.concat([upsample_feat, feat_low], dim=1))
            inner_outs.insert(0, inner_out)

        # print('len of inner_outs:',len(inner_outs))
        # for i in range(len(inner_outs)):
        #     print('shape of inner_outs[{a}]={b}'.format(a=i, b=inner_outs[i].shape))

        outs = [inner_outs[0]]
        for idx in range(len(self.in_channels) - 1):
            feat_low = outs[-1]
            feat_height = inner_outs[idx + 1]
            downsample_feat = self.downsample_convs[idx](feat_low)
            out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
            outs.append(out)

        return outs
    





# @register
# class Gold_HybridEncoder(nn.Module):
#     def __init__(self,
#                  in_channels=[512, 1024, 2048],
#                  feat_strides=[8, 16, 32],
#                  trans_channels=[256, 256, 256, 256],
#                  hidden_dim=256,
#                  nhead=8,
#                  dim_feedforward = 1024,
#                  dropout=0.0,
#                  enc_act='gelu',
#                  use_encoder_idx=[3],
#                  num_encoder_layers=1,
#                  pe_temperature=10000,
#                  expansion=1.0,
#                  depth_mult=1.0,
#                  act='silu',
#                  block=RepVggBlock,
#                  eval_size=None):
#         super().__init__()
#         self.in_channels = in_channels
#         self.feat_strides = feat_strides
#         self.hidden_dim = hidden_dim
#         self.use_encoder_idx = use_encoder_idx
#         self.num_encoder_layers = num_encoder_layers
#         self.pe_temperature = pe_temperature
#         self.eval_size = eval_size
#         self.trans_channels = trans_channels

#         self.out_channels = [hidden_dim for _ in range(len(in_channels))]
#         self.out_strides = feat_strides
        
#         # channel projection
#         # 输入投影层，匹配通道数
#         self.input_proj = nn.ModuleList()
#         for in_channel in in_channels:
#             self.input_proj.append(
#                 nn.Sequential(
#                     nn.Conv2d(in_channel, hidden_dim, kernel_size=1, bias=False),
#                     nn.BatchNorm2d(hidden_dim)
#                 )
#             )

#         # encoder transformer
#         encoder_layer = TransformerEncoderLayer(
#             hidden_dim, 
#             nhead=nhead,
#             dim_feedforward=dim_feedforward, 
#             dropout=dropout,
#             activation=enc_act)

#         self.encoder = nn.ModuleList([
#             TransformerEncoder(copy.deepcopy(encoder_layer), num_encoder_layers) for _ in range(len(use_encoder_idx))
#         ])

#         # top-down fpn
#         self.lateral_convs = nn.ModuleList()
#         self.fpn_blocks = nn.ModuleList()
#         for _ in range(len(in_channels) - 1, 0, -1):
#             self.lateral_convs.append(ConvNormLayer(hidden_dim, hidden_dim, 1, 1, act=act))
#             self.fpn_blocks.append(
#                 CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
#             )

#         # bottom-up pan
#         self.downsample_convs = nn.ModuleList()
#         self.pan_blocks = nn.ModuleList()
#         for _ in range(len(in_channels) - 1):
#             self.downsample_convs.append(
#                 ConvNormLayer(hidden_dim, hidden_dim, 3, 2, act=act)
#             )
#             self.pan_blocks.append(
#                 CSPRepLayer(hidden_dim * 2, hidden_dim, round(3 * depth_mult), act=act, expansion=expansion)
#             )

#         self._reset_parameters()

#         # gold style feature fusion(low level)
#         self.low_FAM = SimFusion_4in()
#         self.low_IFM = nn.Sequential(
#             Conv(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
#             *[RepVggBlock(ch_in=512, ch_out=512) for _ in range(3)],
#             Conv(in_channels=512, out_channels=sum(self.trans_channels[0:2]), kernel_size=1, stride=1, padding=0)
#         )

#         self.reduce_layer_c5 = SimConv(
#             in_channels=1024,
#             out_channels=512,
#             kernel_size=1,
#             stride=1            
#         )
#         self.LAF_p4 = SimFusion_3in(
#             in_channel_list=[hidden_dim, hidden_dim],
#             out_channels=hidden_dim,
#         )
        
#         self.Inject_p4 = InjectionMultiSum_Auto_pool(hidden_dim, hidden_dim, norm_cfg=dict(type='SyncBN', requires_grad=True), activations=nn.ReLU6)

#         self.Rep_p4 = RepBlock(
#             in_channels=hidden_dim,
#             out_channels=hidden_dim,
#             n=12,
#             block=block
#         )

#         self.reduce_layer_p4 = SimConv(
#                 in_channels=hidden_dim,  # 256
#                 out_channels=hidden_dim,  # 128
#                 kernel_size=1,
#                 stride=1
#         )
#         self.LAF_p3 = SimFusion_3in(
#                 in_channel_list=[hidden_dim, hidden_dim],  # 512, 256
#                 out_channels=hidden_dim,  # 256
#         )
#         self.Inject_p3 = InjectionMultiSum_Auto_pool(256, 256, norm_cfg=dict(type='SyncBN', requires_grad=True),
#                                                      activations=nn.ReLU6)
#         self.Rep_p3 = RepBlock(
#                 in_channels=hidden_dim,  # 128
#                 out_channels=hidden_dim,  # 128
#                 n=12,
#                 block=block
#         )        




#     def _reset_parameters(self):
#         if self.eval_size:
#             for idx in self.use_encoder_idx:
#                 stride = self.feat_strides[idx]
#                 pos_embed = self.build_2d_sincos_position_embedding(
#                     self.eval_size[1] // stride, self.eval_size[0] // stride,
#                     self.hidden_dim, self.pe_temperature)
#                 setattr(self, f'pos_embed{idx}', pos_embed)
#                 # self.register_buffer(f'pos_embed{idx}', pos_embed)

#     @staticmethod
#     def build_2d_sincos_position_embedding(w, h, embed_dim=256, temperature=10000.):
#         '''
#         '''
#         grid_w = torch.arange(int(w), dtype=torch.float32)
#         grid_h = torch.arange(int(h), dtype=torch.float32)
#         grid_w, grid_h = torch.meshgrid(grid_w, grid_h, indexing='ij')
#         assert embed_dim % 4 == 0, \
#             'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
#         pos_dim = embed_dim // 4
#         omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
#         omega = 1. / (temperature ** omega)

#         out_w = grid_w.flatten()[..., None] @ omega[None]
#         out_h = grid_h.flatten()[..., None] @ omega[None]

#         return torch.concat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None, :, :]

#     def forward(self, feats):
#         # print('the shape of feats = ',feats.shape)
#         assert len(feats) == len(self.in_channels)
#         # sys.exit(0)
#         proj_feats = [self.input_proj[i](feat) for i, feat in enumerate(feats)]
#         # print('proj_feats=',proj_fea ts)
#         # encoder
#         # AIFI
#         if self.num_encoder_layers > 0:
#             for i, enc_ind in enumerate(self.use_encoder_idx):
#                 h, w = proj_feats[enc_ind].shape[2:]
#                 # flatten [B, C, H, W] to [B, HxW, C]
#                 src_flatten = proj_feats[enc_ind].flatten(2).permute(0, 2, 1)
#                 if self.training or self.eval_size is None:
#                     pos_embed = self.build_2d_sincos_position_embedding(
#                         w, h, self.hidden_dim, self.pe_temperature).to(src_flatten.device)
#                 else:
#                     pos_embed = getattr(self, f'pos_embed{enc_ind}', None).to(src_flatten.device)

#                 memory = self.encoder[i](src_flatten, pos_embed=pos_embed)
#                 proj_feats[enc_ind] = memory.permute(0, 2, 1).reshape(-1, self.hidden_dim, h, w).contiguous()
                
#                 print([x.is_contiguous() for x in proj_feats ])
#                 print('proj_feats')
#         # print('after encoder, proj_feats=',proj_feats)
#         # broadcasting and fusion


#         # Original low level feature fusion method
#         # inner_outs = [proj_feats[-1]]
#         # for idx in range(len(self.in_channels) - 1, 0, -1):
#         #     feat_heigh = inner_outs[0]
#         #     feat_low = proj_feats[idx - 1]
#         #     feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - idx](feat_heigh)
#         #     inner_outs[0] = feat_heigh
#         #     upsample_feat = F.interpolate(feat_heigh, scale_factor=2., mode='nearest')
#         #     inner_out = self.fpn_blocks[0](torch.concat([upsample_feat, feat_low], dim=1))
#         #     inner_outs.insert(0, inner_out)

#         # Gold low level feature fusion method
#         # proj_feats = (s2, s3, s4, f5)

#         # Low-GD
        
#         for i in range(len(proj_feats)):
#             print('proj_feats第{a}层的shape为{b}'.format(a=(i+1), b=proj_feats[i].shape))

#         low_align_feat = self.low_FAM(proj_feats)
#         print('shape of low_align_feat:',low_align_feat.shape)
#         low_fuse_feat = self.low_IFM(low_align_feat)
#         print('shape of low_fuse_feat:', low_fuse_feat.shape)
#         low_global_info = low_fuse_feat.split(self.trans_channels[0:2],dim=1)

#         print('len of low_global_info:', len(low_global_info))
#         print('shape of low_global_info[0]:{a} \nshape of low_global_info[1]:{b}'.format(a=low_global_info[0].shape, b=low_global_info[1].shape))

#         # c5_half = self.reduce_layer_c5(proj_feats[3])
#         # print()
#         p4_adjacent_info = self.LAF_p4(proj_feats[1:])
#         print('shape of p4_adjacent_info:',p4_adjacent_info.shape)
#         for i in range(len(p4_adjacent_info)):
#             print('shape of p4_adjacent_info[{a}]={b}'.format(a=i, b=p4_adjacent_info[i].shape))
#         p4 = self.Inject_p4(p4_adjacent_info, low_global_info[0])
#         p4 = self.Rep_p4(p4)

#         # p4_half = self.reduce_layer_p4(p4)
#         print('len of proj_feats:',len(proj_feats[:3]))
#         p3_adjacent_info = self.LAF_p3(proj_feats[0:3])
#         print('shape of p3_adjacent_info',p3_adjacent_info[i].shape)
#         p3 = self.Inject_p3(p3_adjacent_info, low_global_info[1])
#         p3 = self.Rep_p3(p3)        

#         # Low-GD END

#         feat_heigh = self.lateral_convs[len(self.in_channels) - 1 - 3](proj_feats[3])

#         inner_outs = [p3, p4, feat_heigh]

#         for i in range(len(inner_outs)):
#             print('shape of inner_outs[{a}]={b}'.format(a=i, b=inner_outs[i].shape))


#         outs = [inner_outs[0]]
#         for idx in range(len(self.in_channels) - 2):
#             feat_low = outs[-1]
#             feat_height = inner_outs[idx + 1]
#             downsample_feat = self.downsample_convs[idx](feat_low)
#             out = self.pan_blocks[idx](torch.concat([downsample_feat, feat_height], dim=1))
#             outs.append(out)

#         return outs