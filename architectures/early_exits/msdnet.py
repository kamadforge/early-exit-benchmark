import math
from typing import List

import torch
from torch import nn

from architectures.early_exits.heads import ConvBasic, ranet_head_class


class ConvBN(nn.Module):
    def __init__(self, nIn, nOut, type: str, bottleneck, bnWidth):
        """
        a basic conv in MSDNet, two type
        :param nIn:
        :param nOut:
        :param type: normal or down
        :param bottleneck: use bottlenet or not
        :param bnWidth: bottleneck factor
        """
        super(ConvBN, self).__init__()
        layer = []
        nInner = nIn
        if bottleneck is True:
            nInner = min(nInner, bnWidth * nOut)
            layer.append(nn.Conv2d(nIn, nInner, kernel_size=1, stride=1, padding=0, bias=False))
            layer.append(nn.BatchNorm2d(nInner))
            layer.append(nn.ReLU(True))
        if type == 'normal':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=1, padding=1, bias=False))
        elif type == 'down':
            layer.append(nn.Conv2d(nInner, nOut, kernel_size=3, stride=2, padding=1, bias=False))
        else:
            raise ValueError
        layer.append(nn.BatchNorm2d(nOut))
        layer.append(nn.ReLU(True))
        self.net = nn.Sequential(*layer)

    def forward(self, x):
        return self.net(x)


class ConvDownNormal(nn.Module):
    def __init__(self, nIn1, nIn2, nOut, bottleneck, bnWidth1, bnWidth2):
        super(ConvDownNormal, self).__init__()
        self.conv_down = ConvBN(nIn1, nOut // 2, 'down', bottleneck, bnWidth1)
        self.conv_normal = ConvBN(nIn2, nOut // 2, 'normal', bottleneck, bnWidth2)

    def forward(self, x):
        res = [x[1],
               self.conv_down(x[0]),
               self.conv_normal(x[1])]
        return torch.cat(res, dim=1)


class ConvNormal(nn.Module):
    def __init__(self, nIn, nOut, bottleneck, bnWidth):
        super(ConvNormal, self).__init__()
        self.conv_normal = ConvBN(nIn, nOut, 'normal', bottleneck, bnWidth)

    def forward(self, x):
        if not isinstance(x, list):
            x = [x]
        res = [x[0],
               self.conv_normal(x[0])]
        return torch.cat(res, dim=1)


class MSDNFirstLayer(nn.Module):
    def __init__(self, nIn, nOut, gr_factors, variant, num_scales):
        super(MSDNFirstLayer, self).__init__()
        self.layers = nn.ModuleList()
        if variant.startswith('cifar'):
            self.layers.append(ConvBasic(nIn, nOut * gr_factors[0], kernel=3, stride=1, padding=1))
        elif variant == 'imagenet':
            conv = nn.Sequential(
                nn.Conv2d(nIn, nOut * gr_factors[0], 7, 2, 3),
                nn.BatchNorm2d(nOut * gr_factors[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, 1))
            self.layers.append(conv)
        nIn = nOut * gr_factors[0]
        for i in range(1, num_scales):
            self.layers.append(ConvBasic(nIn, nOut * gr_factors[i], kernel=3, stride=2, padding=1))
            nIn = nOut * gr_factors[i]

    def forward(self, x):
        res = []
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            res.append(x)
        return res


class MSDNLayer(nn.Module):
    def __init__(self, nIn, nOut, gr_factors, bn_factors, num_scales, bottleneck, inScales=None, outScales=None):
        super(MSDNLayer, self).__init__()
        self.nIn = nIn
        self.nOut = nOut
        self.inScales = inScales if inScales is not None else num_scales
        self.outScales = outScales if outScales is not None else num_scales
        self.nScales = num_scales
        self.discard = self.inScales - self.outScales
        self.offset = self.nScales - self.outScales
        self.layers = nn.ModuleList()
        if self.discard > 0:
            nIn1 = nIn * gr_factors[self.offset - 1]
            nIn2 = nIn * gr_factors[self.offset]
            _nOut = nOut * gr_factors[self.offset]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, bottleneck,
                                              bn_factors[self.offset - 1], bn_factors[self.offset]))
        else:
            self.layers.append(ConvNormal(nIn * gr_factors[self.offset], nOut * gr_factors[self.offset],
                                          bottleneck, bn_factors[self.offset]))
        for i in range(self.offset + 1, self.nScales):
            nIn1 = nIn * gr_factors[i - 1]
            nIn2 = nIn * gr_factors[i]
            _nOut = nOut * gr_factors[i]
            self.layers.append(ConvDownNormal(nIn1, nIn2, _nOut, bottleneck, bn_factors[i - 1], bn_factors[i]))

    def forward(self, x):
        if self.discard > 0:
            inp = []
            for i in range(1, self.outScales + 1):
                inp.append([x[i - 1], x[i]])
        else:
            inp = [[x[0]]]
            for i in range(1, self.outScales):
                inp.append([x[i - 1], x[i]])
        res = []
        for i in range(self.outScales):
            res.append(self.layers[i](inp[i]))
        return res


class ParallelModule(nn.Module):
    """
    This module is similar to luatorch's Parallel Table
    input: N tensor
    network: N module
    output: N tensor
    """

    def __init__(self, parallel_modules):
        super(ParallelModule, self).__init__()
        self.m = nn.ModuleList(parallel_modules)

    def forward(self, x):
        res = []
        for i in range(len(x)):
            res.append(self.m[i](x[i]))
        return res


class MSDNet(nn.Module):
    def __init__(self, num_classes: int, num_blocks: int, step: int, stepmode: str, num_channels: int, variant: str,
                 growth_rate: int, gr_factors: List[int], bn_factors: List[int], base: int, prune: str,
                 bottleneck: bool, reduction: float):
        super().__init__()
        self._num_classes = num_classes
        self.blocks = nn.ModuleList()
        # self.classifier = nn.ModuleList()
        self.nBlocks = num_blocks
        self.steps = [base]
        n_layers_all, n_layer_curr = base, 0
        for i in range(1, self.nBlocks):
            self.steps.append(step if stepmode == 'even' else step * i + 1)
            n_layers_all += self.steps[-1]
        # print(f'building network of steps: {self.steps=},{n_layers_all=}')
        nIn = num_channels
        for i in range(self.nBlocks):
            # print(f' ********************** Block {i + 1} **********************')
            m, nIn = self._build_block(nIn, self.steps[i], n_layers_all, n_layer_curr,
                                       prune, variant, gr_factors, bn_factors, len(gr_factors), bottleneck,
                                       growth_rate, reduction)
            self.blocks.append(m)
            n_layer_curr += self.steps[i]
        if variant.startswith('cifar'):
            self.classifier = self._build_classifier_cifar(nIn * gr_factors[-1], self._num_classes)
        elif variant == 'imagenet':
            self.classifier = self._build_classifier_imagenet(nIn * gr_factors[-1], self._num_classes)
        else:
            raise NotImplementedError
        for m in self.blocks:
            if hasattr(m, '__iter__'):
                for _m in m:
                    self._init_weights(_m)
            else:
                self._init_weights(m)
        if hasattr(self.classifier, '__iter__'):
            for _m in m:
                self._init_weights(_m)
        else:
            self._init_weights(self.classifier)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()

    def _build_block(self, nIn, step, n_layer_all, n_layer_curr,
                     prune, variant, gr_factors, bn_factors, num_scales, bottleneck,
                     growth_rate, reduction):
        layers = [MSDNFirstLayer(3, nIn, gr_factors, variant, num_scales)] if n_layer_curr == 0 else []
        for i in range(step):
            n_layer_curr += 1
            if prune == 'min':
                inScales = min(num_scales, n_layer_all - n_layer_curr + 2)
                outScales = min(num_scales, n_layer_all - n_layer_curr + 1)
            elif prune == 'max':
                interval = math.ceil(1.0 * n_layer_all / num_scales)
                inScales = num_scales - math.floor(1.0 * (max(0, n_layer_curr - 2)) / interval)
                outScales = num_scales - math.floor(1.0 * (n_layer_curr - 1) / interval)
            else:
                raise ValueError
            layers.append(MSDNLayer(nIn, growth_rate, gr_factors, bn_factors,
                                    num_scales, bottleneck, inScales, outScales))
            # print(f'|\t\t{inScales=} {outScales=} {nIn=} {growth_rate=}\t\t|')
            nIn += growth_rate
            if prune == 'max' and inScales > outScales and reduction > 0:
                offset = num_scales - outScales
                layers.append(self._build_transition(nIn, math.floor(reduction * nIn), outScales, offset, gr_factors))
                _t = nIn
                nIn = math.floor(reduction * nIn)
                # print(f'|\t\tTransition layer inserted! (max), inChannels {_t},'
                #       f' outChannels {math.floor(reduction * _t)}\t|')
            elif prune == 'min' and reduction > 0 and \
                    ((n_layer_curr == math.floor(1.0 * n_layer_all / 3)) or
                     n_layer_curr == math.floor(2.0 * n_layer_all / 3)):
                offset = num_scales - outScales
                layers.append(self._build_transition(nIn, math.floor(reduction * nIn), outScales, offset, gr_factors))
                nIn = math.floor(reduction * nIn)
                # print('|\t\tTransition layer inserted! (min)\t|')
            # print('')
        return nn.Sequential(*layers), nIn

    def _build_transition(self, nIn, nOut, outScales, offset, gr_factors):
        net = []
        for i in range(outScales):
            net.append(ConvBasic(nIn * gr_factors[offset + i], nOut * gr_factors[offset + i],
                                 kernel=1, stride=1, padding=0))
        return ParallelModule(net)

    def _build_classifier_cifar(self, nIn, num_classes):
        return ranet_head_class(128, 128, 4)(nIn, num_classes)

    def _build_classifier_imagenet(self, nIn, num_classes):
        return ranet_head_class(nIn, nIn, 4)(nIn, num_classes)

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return self.classifier(x[-1])

    def forward_generator(self, x):
        for block in self.blocks:
            for layer in block:
                x = layer(x)
                # selects the last scale as the features that the IC will use
                _x = x[-1]
                _x = yield _x, None
                x[-1] = _x
        _ = yield None, self.classifier(x[-1])

    def per_head_block_modules(self):
        return [layer for block in self.blocks for layer in block]
