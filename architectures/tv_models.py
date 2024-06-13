from functools import partial

import torch
import torchvision


def get_efficientnet_b0(num_classes, input_size=64):
    model = torchvision.models.efficientnet_b0(progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    if isinstance(block, (torchvision.models.efficientnet.FusedMBConv,
                                          torchvision.models.efficientnet.MBConv)):
                        x = yield x, None
            else:
                x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_channels = 3
    model.input_size = input_size
    model.number_of_classes = num_classes
    return model


def get_efficientnet_v2_s(num_classes, input_size=64, dropout_prob=0.2, stochastic_depth_prob=0.2):
    model = torchvision.models.efficientnet_v2_s(num_classes=num_classes, stochastic_depth_prob=stochastic_depth_prob)  # change
    model.classifier[0] = torch.nn.Dropout(p=dropout_prob, inplace=True)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    if isinstance(block, (torchvision.models.efficientnet.FusedMBConv,
                                          torchvision.models.efficientnet.MBConv)):
                        x = yield x, None
            else:
                x = stage(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_channels = 3
    model.input_size = input_size
    model.number_of_classes = num_classes
    return model


def get_convnext_t(num_classes, input_size=64):
    model = torchvision.models.convnext_tiny(progress=False)

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
    model.input_channels = 3
    model.input_size = input_size
    model.number_of_classes = num_classes
    return model


def get_vit_b_16(num_classes, input_size=64):
    model = torchvision.models.vit_b_16(progress=False)

    def forward_generator(self, x):
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]
        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        # BEGIN ENCODER
        # equivalent to: x = self.encoder(x)
        x = x + self.encoder.pos_embedding
        x = self.encoder.dropout(x)
        # go through encoder blocks
        for block in self.encoder.layers:
            x = block(x)
            x = yield x, None
        x = self.encoder.ln(x)
        # END OF ENCODER
        # classifier token
        x = x[:, 0]
        x = self.heads(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_channels = 3
    model.input_size = input_size
    model.number_of_classes = num_classes
    return model


def get_swin_v2_s(num_classes, input_size=64):
    model = torchvision.models.swin_v2_s(progress=False)

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_channels = 3
    model.input_size = input_size
    model.number_of_classes = num_classes
    return model



def get_resnet50(num_classes, input_size=64):
    model = torchvision.models.resnet50(num_classes=num_classes)
    
    # model.maxpool = torch.nn.Identity()
    # model.conv1 = torch.nn.Conv2d(
    #         3, int(64 * 1.0), kernel_size=3, stride=1, padding=2, bias=False
    #     )

    def forward_generator(self, x):
        for stage in self.features:
            if isinstance(stage, torch.nn.Sequential):
                for block in stage:
                    x = block(x)
                    x = yield x, None
            else:
                x = stage(x)
                # x = yield x, None
        x = self.norm(x)
        x = self.permute(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.head(x)
        _ = yield None, x

    model.forward_generator = partial(forward_generator, model)
    model.input_size = input_size
    model.input_channels = 3
    model.number_of_classes = num_classes
    return model



