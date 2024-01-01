from pickle import FALSE
import warnings
import numpy as np
import torch
from torch import nn
from torch.nn.parameter import Parameter
# from yolov6.layers.common import BottleRep, RepVGGBlock, RepBlock, BepC3, SimSPPF, SPPF, SimCSPSPPF, CSPSPPF, \
#     ConvWrapper
from src.core import register






__all__ = ['EfficientRep']



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
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

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

class ConvWrapper(nn.Module):
    '''Wrapper for normal Conv with SiLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = Conv(in_channels, out_channels, kernel_size, stride, groups, bias)
    
    def forward(self, x):
        return self.block(x)


class SimConvWrapper(nn.Module):
    '''Wrapper for normal Conv with ReLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, bias=True):
        super().__init__()
        self.block = SimConv(in_channels, out_channels, kernel_size, stride, groups, bias)
    
    def forward(self, x):
        return self.block(x)


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv_C3(nn.Module):
    '''Standard convolution in BepC3-Block'''
    
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
    def forward_fuse(self, x):
        return self.act(self.conv(x))

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

def conv_bn(in_channels, out_channels, kernel_size, stride, padding, groups=1, bias=False):
    '''Basic cell for rep-style block, including conv and bn'''
    result = nn.Sequential()
    result.add_module('conv', nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding, groups=groups,
                                        bias=bias))
    result.add_module('bn', nn.BatchNorm2d(num_features=out_channels))
    return result


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

class SimSPPF(nn.Module):
    '''Simplified SPPF with ReLU VAN_activation'''
    
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class SPPF(nn.Module):
    '''Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher'''
    
    def __init__(self, in_channels, out_channels, kernel_size=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = in_channels // 2  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, out_channels, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class SimCSPSPPF(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super(SimCSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = SimConv(in_channels, c_, 1, 1)
        self.cv2 = SimConv(in_channels, c_, 1, 1)
        self.cv3 = SimConv(c_, c_, 3, 1)
        self.cv4 = SimConv(c_, c_, 1, 1)
        
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = SimConv(4 * c_, c_, 1, 1)
        self.cv6 = SimConv(c_, c_, 3, 1)
        self.cv7 = SimConv(2 * c_, out_channels, 1, 1)
    
    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


class CSPSPPF(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, in_channels, out_channels, kernel_size=5, e=0.5):
        super(CSPSPPF, self).__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv(in_channels, c_, 1, 1)
        self.cv2 = Conv(in_channels, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, out_channels, 1, 1)
    
    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], 1)))
        return self.cv7(torch.cat((y0, y3), dim=1))


@register
class EfficientRep(nn.Module):
    '''EfficientRep Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''
    
    def __init__(
            self,
            in_channels=3,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            fuse_P2=False,
            cspsppf=False
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2
        
        self.stem = block(
                in_channels=in_channels,
                out_channels=channels_list[0],
                kernel_size=3,
                stride=2
        )
        
        self.ERBlock_2 = nn.Sequential(
                block(
                        in_channels=channels_list[0],
                        out_channels=channels_list[1],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[1],
                        out_channels=channels_list[1],
                        n=num_repeats[1],
                        block=block,
                )
        )
        
        self.ERBlock_3 = nn.Sequential(
                block(
                        in_channels=channels_list[1],
                        out_channels=channels_list[2],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[2],
                        out_channels=channels_list[2],
                        n=num_repeats[2],
                        block=block,
                )
        )
        
        self.ERBlock_4 = nn.Sequential(
                block(
                        in_channels=channels_list[2],
                        out_channels=channels_list[3],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[3],
                        out_channels=channels_list[3],
                        n=num_repeats[3],
                        block=block,
                )
        )
        
        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF
        
        self.ERBlock_5 = nn.Sequential(
                block(
                        in_channels=channels_list[3],
                        out_channels=channels_list[4],
                        kernel_size=3,
                        stride=2,
                ),
                RepBlock(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        n=num_repeats[4],
                        block=block,
                ),
                channel_merge_layer(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        kernel_size=5
                )
        )
    
    def forward(self, x):
        
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        
        return tuple(outputs)


class EfficientRep6(nn.Module):
    '''EfficientRep+P6 Backbone
    EfficientRep is handcrafted by hardware-aware neural network design.
    With rep-style struct, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    '''
    
    def __init__(
            self,
            in_channels=3,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            fuse_P2=False,
            cspsppf=False
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2
        
        self.stem = block(
                in_channels=in_channels,
                out_channels=channels_list[0],
                kernel_size=3,
                stride=2
        )
        
        self.ERBlock_2 = nn.Sequential(
                block(
                        in_channels=channels_list[0],
                        out_channels=channels_list[1],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[1],
                        out_channels=channels_list[1],
                        n=num_repeats[1],
                        block=block,
                )
        )
        
        self.ERBlock_3 = nn.Sequential(
                block(
                        in_channels=channels_list[1],
                        out_channels=channels_list[2],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[2],
                        out_channels=channels_list[2],
                        n=num_repeats[2],
                        block=block,
                )
        )
        
        self.ERBlock_4 = nn.Sequential(
                block(
                        in_channels=channels_list[2],
                        out_channels=channels_list[3],
                        kernel_size=3,
                        stride=2
                ),
                RepBlock(
                        in_channels=channels_list[3],
                        out_channels=channels_list[3],
                        n=num_repeats[3],
                        block=block,
                )
        )
        
        self.ERBlock_5 = nn.Sequential(
                block(
                        in_channels=channels_list[3],
                        out_channels=channels_list[4],
                        kernel_size=3,
                        stride=2,
                ),
                RepBlock(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        n=num_repeats[4],
                        block=block,
                )
        )
        
        channel_merge_layer = SimSPPF if not cspsppf else SimCSPSPPF
        
        self.ERBlock_6 = nn.Sequential(
                block(
                        in_channels=channels_list[4],
                        out_channels=channels_list[5],
                        kernel_size=3,
                        stride=2,
                ),
                RepBlock(
                        in_channels=channels_list[5],
                        out_channels=channels_list[5],
                        n=num_repeats[5],
                        block=block,
                ),
                channel_merge_layer(
                        in_channels=channels_list[5],
                        out_channels=channels_list[5],
                        kernel_size=5
                )
        )
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)
        
        return tuple(outputs)


class BepC3(nn.Module):
    '''Beer-mug RepC3 Block'''
    
    def __init__(self, in_channels, out_channels, n=1, e=0.5, concat=True,
                 block=RepVGGBlock):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(out_channels * e)  # hidden channels
        self.cv1 = Conv_C3(in_channels, c_, 1, 1)
        self.cv2 = Conv_C3(in_channels, c_, 1, 1)
        self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1)
        if block == ConvWrapper:
            self.cv1 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv2 = Conv_C3(in_channels, c_, 1, 1, act=nn.SiLU())
            self.cv3 = Conv_C3(2 * c_, out_channels, 1, 1, act=nn.SiLU())
        
        self.m = RepBlock(in_channels=c_, out_channels=c_, n=n, block=BottleRep, basic_block=block)
        self.concat = concat
        if not concat:
            self.cv3 = Conv_C3(c_, out_channels, 1, 1)
    
    def forward(self, x):
        if self.concat is True:
            return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        else:
            return self.cv3(self.m(self.cv1(x)))

class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone module.
    """
    
    def __init__(
            self,
            in_channels=3,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            csp_e=float(1) / 2,
            fuse_P2=False,
            cspsppf=False
    ):
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2
        
        self.stem = block(
                in_channels=in_channels,
                out_channels=channels_list[0],
                kernel_size=3,
                stride=2
        )
        
        self.ERBlock_2 = nn.Sequential(
                block(
                        in_channels=channels_list[0],
                        out_channels=channels_list[1],
                        kernel_size=3,
                        stride=2
                ),
                BepC3(
                        in_channels=channels_list[1],
                        out_channels=channels_list[1],
                        n=num_repeats[1],
                        e=csp_e,
                        block=block,
                )
        )
        
        self.ERBlock_3 = nn.Sequential(
                block(
                        in_channels=channels_list[1],
                        out_channels=channels_list[2],
                        kernel_size=3,
                        stride=2
                ),
                BepC3(
                        in_channels=channels_list[2],
                        out_channels=channels_list[2],
                        n=num_repeats[2],
                        e=csp_e,
                        block=block,
                )
        )
        
        self.ERBlock_4 = nn.Sequential(
                block(
                        in_channels=channels_list[2],
                        out_channels=channels_list[3],
                        kernel_size=3,
                        stride=2
                ),
                BepC3(
                        in_channels=channels_list[3],
                        out_channels=channels_list[3],
                        n=num_repeats[3],
                        e=csp_e,
                        block=block,
                )
        )
        
        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF
        
        self.ERBlock_5 = nn.Sequential(
                block(
                        in_channels=channels_list[3],
                        out_channels=channels_list[4],
                        kernel_size=3,
                        stride=2,
                ),
                BepC3(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        n=num_repeats[4],
                        e=csp_e,
                        block=block,
                ),
                channel_merge_layer(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        kernel_size=5
                )
        )
    
    def forward(self, x):
        
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        if self.fuse_P2:
            outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        
        return tuple(outputs)


class CSPBepBackbone_P6(nn.Module):
    """
    CSPBepBackbone+P6 module. 
    """
    
    def __init__(
            self,
            in_channels=3,
            channels_list=None,
            num_repeats=None,
            block=RepVGGBlock,
            csp_e=float(1) / 2,
            fuse_P2=False,
            cspsppf=False
    ):
        super().__init__()
        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_P2 = fuse_P2
        
        self.stem = block(
                in_channels=in_channels,
                out_channels=channels_list[0],
                kernel_size=3,
                stride=2
        )
        
        self.ERBlock_2 = nn.Sequential(
                block(
                        in_channels=channels_list[0],
                        out_channels=channels_list[1],
                        kernel_size=3,
                        stride=2
                ),
                BepC3(
                        in_channels=channels_list[1],
                        out_channels=channels_list[1],
                        n=num_repeats[1],
                        e=csp_e,
                        block=block,
                )
        )
        
        self.ERBlock_3 = nn.Sequential(
                block(
                        in_channels=channels_list[1],
                        out_channels=channels_list[2],
                        kernel_size=3,
                        stride=2
                ),
                BepC3(
                        in_channels=channels_list[2],
                        out_channels=channels_list[2],
                        n=num_repeats[2],
                        e=csp_e,
                        block=block,
                )
        )
        
        self.ERBlock_4 = nn.Sequential(
                block(
                        in_channels=channels_list[2],
                        out_channels=channels_list[3],
                        kernel_size=3,
                        stride=2
                ),
                BepC3(
                        in_channels=channels_list[3],
                        out_channels=channels_list[3],
                        n=num_repeats[3],
                        e=csp_e,
                        block=block,
                )
        )
        
        channel_merge_layer = SPPF if block == ConvWrapper else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvWrapper else SimCSPSPPF
        
        self.ERBlock_5 = nn.Sequential(
                block(
                        in_channels=channels_list[3],
                        out_channels=channels_list[4],
                        kernel_size=3,
                        stride=2,
                ),
                BepC3(
                        in_channels=channels_list[4],
                        out_channels=channels_list[4],
                        n=num_repeats[4],
                        e=csp_e,
                        block=block,
                ),
        )
        self.ERBlock_6 = nn.Sequential(
                block(
                        in_channels=channels_list[4],
                        out_channels=channels_list[5],
                        kernel_size=3,
                        stride=2,
                ),
                BepC3(
                        in_channels=channels_list[5],
                        out_channels=channels_list[5],
                        n=num_repeats[5],
                        e=csp_e,
                        block=block,
                ),
                channel_merge_layer(
                        in_channels=channels_list[5],
                        out_channels=channels_list[5],
                        kernel_size=5
                )
        )
    
    def forward(self, x):
        outputs = []
        x = self.stem(x)
        x = self.ERBlock_2(x)
        outputs.append(x)
        x = self.ERBlock_3(x)
        outputs.append(x)
        x = self.ERBlock_4(x)
        outputs.append(x)
        x = self.ERBlock_5(x)
        outputs.append(x)
        x = self.ERBlock_6(x)
        outputs.append(x)
        
        return tuple(outputs)
