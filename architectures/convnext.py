from functools import partial
from typing import List, Optional, Callable, Any, Sequence

import torch
from torch import nn, Tensor
from torchvision.models.convnext import CNBlockConfig, CNBlock, LayerNorm2d
from torchvision.ops import Conv2dNormActivation


class ConvNeXt(nn.Module):
    def __init__(
            self,
            block_setting: List[CNBlockConfig],
            stochastic_depth_prob: float = 0.0,
            layer_scale: float = 1e-6,
            num_classes: int = 1000,
            block: Optional[Callable[..., nn.Module]] = None,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
            first_conv_kernel_size: int = 4,
            first_conv_stride: int = 4,
            first_conv_padding: int = 0,
            **kwargs: Any,
    ) -> None:
        super().__init__()

        if not block_setting:
            raise ValueError("The block_setting should not be empty")
        elif not (isinstance(block_setting, Sequence) and all([isinstance(s, CNBlockConfig) for s in block_setting])):
            raise TypeError("The block_setting should be List[CNBlockConfig]")

        if block is None:
            block = CNBlock

        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3,
                firstconv_output_channels,
                kernel_size=first_conv_kernel_size,
                stride=first_conv_stride,
                padding=first_conv_padding,
                norm_layer=norm_layer,
                activation_layer=None,
                bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            # Bottlenecks
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                # adjust stochastic depth probability based on the depth of the stage block
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1
            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling
                layers.append(
                    nn.Sequential(
                        norm_layer(cnf.input_channels),
                        nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                    )
                )

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        lastblock = block_setting[-1]
        lastconv_output_channels = (
            lastblock.out_channels if lastblock.out_channels is not None else lastblock.input_channels
        )
        self.classifier = nn.Sequential(
            norm_layer(lastconv_output_channels), nn.Flatten(1), nn.Linear(lastconv_output_channels, num_classes)
        )

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def get_convnext_t(input_size=224, stochastic_depth_prob=0.1, num_classes=1000,
                   first_conv_kernel_size=4, first_conv_stride=4, first_conv_padding=0):
    block_setting = [
        CNBlockConfig(96, 192, 3),
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 9),
        CNBlockConfig(768, None, 3),
    ]
    model = ConvNeXt(block_setting, stochastic_depth_prob=stochastic_depth_prob, num_classes=num_classes,
                     first_conv_kernel_size=first_conv_kernel_size, first_conv_stride=first_conv_stride,
                     first_conv_padding=first_conv_padding)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.avgpool(x)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = input_size
    model.input_channels = 3
    model.number_of_classes = num_classes
    return model
