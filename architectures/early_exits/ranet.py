import copy
import math
from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from architectures.early_exits.heads import ConvBasic, ranet_head_class


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bnAfter, bnWidth):
        """
        a basic conv in RANet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bnAfter: the location of batch Norm
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bnAfter is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
            if type == 'normal':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                       stride=1, padding=1, bias=False))
            elif type == 'down':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                       stride=2, padding=1, bias=False))
            else:
                raise ValueError
            layer.append(nn.BatchNorm2d(nOut))
            layer.append(nn.ReLU(True))

        else:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.BatchNorm2d(nIn))
            layer.append(nn.ReLU(True))
            layer.append(nn.Conv2d(
                nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
            if type == 'normal':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                       stride=1, padding=1, bias=False))
            elif type == 'down':
                layer.append(nn.Conv2d(nInner, nOut, kernel_size=3,
                                       stride=2, padding=1, bias=False))
            else:
                raise ValueError
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)


class ConvUpNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2, compress_factor, down_sample):
        '''
        The convolution with normal and up-sampling connection.
        '''
        super(ConvUpNormal, self).__init__()
        self.conv_up = ConvBN(nIn2, math.floor(nOut * compress_factor), 'normal',
                              bottleneck, bnWidth2)
        if down_sample:
            self.conv_normal = ConvBN(nIn1, nOut - math.floor(nOut * compress_factor), 'down',
                                      bottleneck, bnWidth1)
        else:
            self.conv_normal = ConvBN(nIn1, nOut - math.floor(nOut * compress_factor), 'normal',
                                      bottleneck, bnWidth1)

    def forward(self, x):
        res = self.conv_normal(x[1])
        _, _, h, w = res.size()
        res = [F.interpolate(x[1], size=(h, w), mode='bilinear', align_corners=True),
               F.interpolate(self.conv_up(x[0]), size=(h, w), mode='bilinear', align_corners=True),
               res]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        '''
        The convolution with normal connection.
        '''
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal', bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0], self.conv_normal(x[0])]
        return torch.cat(res, dim=1)


class _BlockNormal(nn.Module):
    def __init__(self, num_layers, nIn, growth_rate, reduction_rate, trans, bnFactor):
        '''
        The basic computational block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        '''
        super(_BlockNormal, self).__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            self.layers.append(ConvNormal(nIn + i * growth_rate, growth_rate, True, bnFactor))
        self.trans_flag = trans
        if trans:
            nOut = nIn + num_layers * growth_rate
            self.trans = ConvBasic(nOut, math.floor(1.0 * reduction_rate * nOut), kernel=1, stride=1, padding=0)

    def forward(self, x):
        output = [x]
        for i in range(self.num_layers):
            x = self.layers[i](x)
            output.append(x)
        x = output[-1]
        if self.trans_flag:
            x = self.trans(x)
        return x, output

    def _blockType(self):
        return 'norm'


class _BlockUpNormal(nn.Module):
    def __init__(self, num_layers, nIn, nIn_lowFtrs, growth_rate, reduction_rate, trans, down, compress_factor,
                 bnFactor1, bnFactor2):
        '''
        The basic fusion block in RANet with num_layers layers.
        trans: If True, the block will add a transiation layer at the end of the block
                with reduction_rate.
        compress_factor: There will be compress_factor*100% information from the previous
                sub-network.
        '''
        super().__init__()
        self.layers = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers - 1):
            self.layers.append(
                ConvUpNormal(nIn + i * growth_rate, nIn_lowFtrs[i], growth_rate, True, bnFactor1, bnFactor2,
                             compress_factor, False))
        self.layers.append(
            ConvUpNormal(nIn + (i + 1) * growth_rate, nIn_lowFtrs[i + 1], growth_rate, True, bnFactor1, bnFactor2,
                         compress_factor, down))
        nOut = nIn + num_layers * growth_rate
        self.conv_last = ConvBasic(nIn_lowFtrs[num_layers], math.floor(nOut * compress_factor), kernel=1, stride=1,
                                   padding=0)
        nOut = nOut + math.floor(nOut * compress_factor)
        self.trans_flag = trans
        if trans:
            self.trans = ConvBasic(nOut, math.floor(1.0 * reduction_rate * nOut), kernel=1, stride=1, padding=0)

    def forward(self, x, low_feat):
        output = [x]
        for i in range(self.num_layers):
            inp = [low_feat[i]]
            inp.append(x)
            x = self.layers[i](inp)
            output.append(x)
        x = output[-1]
        _, _, h, w = x.size()
        x = [x]
        x.append(
            F.interpolate(self.conv_last(low_feat[self.num_layers]), size=(h, w), mode='bilinear', align_corners=True))
        x = torch.cat(x, dim=1)
        if self.trans_flag:
            x = self.trans(x)
        return x, output

    def _blockType(self):
        return 'up'


class RAFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, gr_factor, scale_list, variant, num_scales):
        '''
        RAFirstLayer gennerates the base features for RANet.
        The scale 1 means the lowest resoultion in the network.
        '''
        super(RAFirstLayer, self).__init__()
        _grFactor = gr_factor[::-1]  # 1-2-4
        _scale_list = scale_list[::-1]  # 3-2-1
        self.layers = nn.ModuleList()
        if variant.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * _grFactor[0],
                                         kernel=3, stride=1, padding=1))
        elif variant == 'tinyimagenet':
            self.layers.append(ConvBasic(nIn, nOut * _grFactor[0], kernel=3, stride=2, padding=1))
        elif variant == 'imagenet':
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * _grFactor[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * _grFactor[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)
        nIn = nOut * _grFactor[0]
        s = _scale_list[0]
        for i in range(1, num_scales):
            if s == _scale_list[i]:
                self.layers.append(ConvBasic(nIn, nOut * _grFactor[i],
                                             kernel=3, stride=1, padding=1))
            else:
                self.layers.append(ConvBasic(nIn, nOut * _grFactor[i],
                                             kernel=3, stride=2, padding=1))
                s = _scale_list[i]
            nIn = nOut * _grFactor[i]

    def forward(self, x):
        # res[0] with the smallest resolutions
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res[::-1]


class RANet(nn.Module):
    def __init__(self, num_classes: int, num_blocks: int, step: int, stepmode: str, compress_factor: float,
                 num_channels: int, variant: str, growth_rate: int, gr_factors: List[int], bn_factors: List[int],
                 scale_list: List[int], reduction: float, block_step: int):
        super().__init__()
        self._num_classes = num_classes
        self.scale_flows = nn.ModuleList()
        # self.classifier = nn.ModuleList()
        self.compress_factor = compress_factor
        self.bnFactor = copy.copy(bn_factors)
        self.nScales = len(scale_list)  # 3
        self.num_scales = self.nScales  # 3
        # The number of blocks in each scale flow
        self.nBlocks = [0]
        for i in range(self.nScales):
            self.nBlocks.append(block_step * i + num_blocks)  # [0, 2, 4, 6]
        # The number of layers in each block
        self.steps = step
        self.FirstLayer = RAFirstLayer(3, num_channels, gr_factors, scale_list, variant, self.nScales)
        steps = [step]
        for ii in range(self.nScales):
            scale_flow = nn.ModuleList()
            nIn = num_channels * gr_factors[ii]  # grFactor = [4,2,1]
            _nIn_lowFtrs = []
            for i in range(self.nBlocks[ii + 1]):
                current_growth_rate = growth_rate * gr_factors[ii]
                # If transiation
                trans = self._trans_flag(i + 1, n_block_all=self.nBlocks[ii + 1], inScale=scale_list[ii])
                if i + 1 > self.nBlocks[ii]:
                    # print(f'{ii=} {i=} _build_norm_block {nIn=} {steps[i + 1 - 1]=} {current_growth_rate=}'
                    #       f'{reduction=} {trans=} {self.bnFactor[ii]=}')
                    m, nOuts = self._build_norm_block(nIn, steps[i + 1 - 1], current_growth_rate, reduction,
                                                      trans,
                                                      bnFactor=self.bnFactor[ii])
                    if stepmode == 'even':
                        steps.append(step)
                    elif stepmode == 'lg':
                        steps.append(steps[-1] + step)
                    else:
                        raise NotImplementedError
                else:
                    if i + 1 in self.nBlocks[:ii + 1][-(scale_list[ii] - 1):]:
                        m, nOuts = self._build_upNorm_block(nIn, nIn_lowFtrs[i], steps[i + 1 - 1],
                                                            current_growth_rate, reduction, trans, down=True,
                                                            bnFactor1=self.bnFactor[ii],
                                                            bnFactor2=self.bnFactor[ii - 1])
                    else:
                        m, nOuts = self._build_upNorm_block(nIn, nIn_lowFtrs[i], steps[i + 1 - 1],
                                                            current_growth_rate, reduction, trans, down=False,
                                                            bnFactor1=self.bnFactor[ii],
                                                            bnFactor2=self.bnFactor[ii - 1])
                nIn = nOuts[-1]
                scale_flow.append(m)
                # if i + 1 > self.nBlocks[ii]:
                #     if variant.startswith('cifar100'):
                #         self.classifier.append(self._build_classifier_cifar(nIn, 100))
                #     elif variant.startswith('cifar10'):
                #         self.classifier.append(self._build_classifier_cifar(nIn, 10))
                #     elif variant == 'ImageNet':
                #         self.classifier.append(
                #             self._build_classifier_imagenet(nIn, 1000))
                #     else:
                #         raise NotImplementedError
                if i == self.nBlocks[ii + 1] - 1 and ii == self.nScales - 1:
                    if variant.startswith('cifar100'):
                        self.classifier = self._build_classifier_cifar(nIn, num_classes)
                    elif variant.startswith('cifar10'):
                        self.classifier = self._build_classifier_cifar(nIn, num_classes)
                    elif variant.startswith('tinyimagenet'):
                        self.classifier = self._build_classifier_cifar(nIn, num_classes)
                    elif variant == 'imagenet':
                        self.classifier = self._build_classifier_imagenet(nIn, num_classes)
                    else:
                        raise NotImplementedError
                _nIn_lowFtrs.append(nOuts[:-1])
            nIn_lowFtrs = _nIn_lowFtrs
            self.scale_flows.append(scale_flow)
        for m in self.scale_flows:
            for _m in m.modules():
                self._init_weights(_m)
        # for m in self.classifier:
        #     for _m in m.modules():
        #         self._init_weights(_m)
        for _m in self.classifier.modules():
            self._init_weights(_m)
        # print(f'Created RANet model: {self}')

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_norm_block(self, nIn, step, growth_rate, reduction_rate, trans, bnFactor=2):
        block = _BlockNormal(step, nIn, growth_rate, reduction_rate, trans, bnFactor=bnFactor)
        nOuts = []
        for i in range(step + 1):
            nOut = (nIn + i * growth_rate)
            nOuts.append(nOut)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)
        return block, nOuts

    def _build_upNorm_block(self, nIn, nIn_lowFtr, step, growth_rate, reduction_rate, trans, down, bnFactor1=1,
                            bnFactor2=2):
        compress_factor = self.compress_factor
        block = _BlockUpNormal(step, nIn, nIn_lowFtr, growth_rate, reduction_rate, trans, down, compress_factor,
                               bnFactor1=bnFactor1, bnFactor2=bnFactor2)
        nOuts = []
        for i in range(step + 1):
            nOut = (nIn + i * growth_rate)
            nOuts.append(nOut)
        nOut = nOut + math.floor(nOut * compress_factor)
        if trans:
            nOut = math.floor(1.0 * reduction_rate * nOut)
        nOuts.append(nOut)
        return block, nOuts

    def _trans_flag(self, n_block_curr, n_block_all, inScale):
        flag = False
        for i in range(inScale - 1):
            if n_block_curr == math.floor((i + 1) * n_block_all / inScale):
                flag = True
        return flag

    def _build_classifier_cifar(self, nIn, num_classes):
        return ranet_head_class(128, 128, 4)(nIn, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        return ranet_head_class(nIn, nIn, 4)(nIn, num_classes)

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward(self, x):
        inp = self.FirstLayer(x)
        res, low_ftrs = None, []
        for ii in range(self.num_scales):
            _x = inp[ii]
            _low_ftrs = []
            for i in range(self.nBlocks[ii + 1]):
                if self.scale_flows[ii][i]._blockType() == 'norm':
                    _x, _low_ftr = self.scale_flows[ii][i](_x)
                    _low_ftrs.append(_low_ftr)
                else:
                    _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
                    _low_ftrs.append(_low_ftr)
                if i == self.nBlocks[ii + 1] - 1 and ii == self.num_scales - 1:
                    res = self.classifier(_x)
            low_ftrs = _low_ftrs
        return res

    def forward_generator(self, x):
        inp = self.FirstLayer(x)
        low_ftrs = []
        for ii in range(self.num_scales):
            _x = inp[ii]
            _low_ftrs = []
            for i in range(self.nBlocks[ii + 1]):
                if self.scale_flows[ii][i]._blockType() == 'norm':
                    _x, _low_ftr = self.scale_flows[ii][i](_x)
                    _low_ftrs.append(_low_ftr)
                else:
                    _x, _low_ftr = self.scale_flows[ii][i](_x, low_ftrs[i])
                    _low_ftrs.append(_low_ftr)
                if ((i + 1 > self.nBlocks[ii] and ii < self.num_scales - 1) or
                        (i + 1 > self.nBlocks[ii] and i + 1 < self.nBlocks[ii + 1])):
                    # for intermediate
                    _x = yield _x, None
                elif i == self.nBlocks[ii + 1] - 1 and ii == self.num_scales - 1:
                    # for the last one (longest):
                    _ = yield None, self.classifier(_x)
            low_ftrs = _low_ftrs
