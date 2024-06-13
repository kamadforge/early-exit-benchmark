import math

import torch
from torch import nn

from utils import find_module_names, get_module_by_name, set_module_by_name, get_module_name, get_parent_module_name


class FCNet(nn.Module):
    def __init__(self, input_size, channels, num_layers, layer_size, classes):
        super().__init__()
        assert input_size > 1
        assert channels >= 1
        assert num_layers > 1
        assert layer_size > 1
        assert classes > 1
        self.input_size = input_size
        self.input_channels = channels
        self.num_layers = num_layers
        self.layer_size = layer_size
        self._num_classes = classes
        self.layers = nn.ModuleList()
        # first layer
        self.layers.append(nn.Linear(self.input_size ** 2 * self.input_channels, self.layer_size))
        num_layers -= 1
        # remaining layers
        for i in range(num_layers):
            self.layers.append(nn.Linear(self.layer_size, self.layer_size))
        self.layers.append(nn.Linear(self.layer_size, self._num_classes))

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for fc_layer in self.layers:
            x = torch.relu(fc_layer(x))
        return x

    def forward_generator(self, x):
        x = x.view(x.size(0), -1)
        for fc_layer in self.layers[:-1]:
            x = torch.relu(fc_layer(x))
            x = yield x, None
        x = torch.relu(self.layers[-1](x))
        yield None, x


class DCNet(nn.Module):
    def __init__(self, input_size, channels, num_layers, num_filters, kernel_size, classes, batchnorm=True):
        super().__init__()
        assert input_size > 1
        assert channels >= 1
        assert classes > 1
        assert num_layers >= 1
        self.input_size = input_size
        self.input_channels = channels
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.num_filters = num_filters
        self._num_classes = classes
        self.batchnorm = batchnorm
        self.layers = nn.ModuleList()
        if self.batchnorm:
            self.bn_layers = nn.ModuleList()
        # assume, for simplicity, that we only use 'same' padding and stride 1
        padding = (self.kernel_size - 1) // 2
        c_in = self.input_channels
        c_out = self.num_filters
        for layer in range(self.num_layers):
            self.layers.append(nn.Conv2d(c_in, c_out, kernel_size=self.kernel_size, stride=1, padding=padding))
            c_in, c_out = c_out, c_out
            # c_in, c_out = c_out, c_out + self.filters_inc
            if self.batchnorm:
                self.bn_layers.append(nn.BatchNorm2d(c_out))
        self.layers.append(nn.Linear(c_out, self._num_classes))

    @property
    def number_of_classes(self):
        return self._num_classes

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            if self.batchnorm:
                x = self.bn_layers[i](x)
        x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
        last_activations = self.layers[-1](x_transformed)
        return last_activations

    def forward_generator(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = torch.relu(layer(x))
            if self.batchnorm:
                x = self.bn_layers[i](x)
            x = yield x, None
        x_transformed = nn.functional.max_pool2d(x, (x.size(2), x.size(3))).view(x.size(0), -1)
        last_activations = self.layers[-1](x_transformed)
        _ = yield None, last_activations


class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, 'Embedding dimension must be divisible by the number of heads.'
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout_p = dropout_p
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=True)
        self.o_proj = nn.Linear(embed_dim, embed_dim, bias=True)

    @staticmethod
    def scaled_dot_product(q, k, v, mask=None, dropout_p=0.0):
        d_k = q.size(-1)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        attention = nn.functional.softmax(attn_logits, dim=-1)
        if dropout_p > 0.0:
            attention = nn.functional.dropout(attention, p=dropout_p)
        values = torch.matmul(attention, v)
        return values, attention

    def forward(self, query, key, value, _key_padding_mask=None, need_weights=False, attn_mask=None):
        assert query.size() == key.size() == value.size()
        batch_size, seq_length, embed_dim = query.size()
        query, key, value = self.q_proj(query), self.k_proj(key), self.v_proj(value),
        # separate the head dimension, and permute dimensions into [Batch, Head, SeqLen, Dims]
        query = query.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.reshape(batch_size, seq_length, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        # Determine value outputs
        values, attention = self.scaled_dot_product(query, key, value, mask=attn_mask,
                                                    dropout_p=self.dropout_p if self.training else 0.0)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, embed_dim)
        o = self.o_proj(values)
        if need_weights:
            return o, attention
        else:
            return o, None
