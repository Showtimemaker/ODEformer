import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torchdiffeq import odeint
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(
            self, 'minute_embed') else 0.

        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        return self.dropout(x)

class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars

#自创魂技
# class DataEmbedding1(nn.Module):
#     def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1,input=4):
#         super(DataEmbedding1, self).__init__()
#
#         self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
#         self.position_embedding = PositionalEmbedding(d_model=d_model)
#         self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
#                                                     freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
#             d_model=d_model, embed_type=embed_type, freq=freq)
#         self.dropout = nn.Dropout(p=dropout)
#         self.odeblock1 = ODEblock(ODEfunc1(input))
#         self.odeblock2 = ODEblock(ODEfunc2(input))
#         self.norm = nn.LayerNorm(64)
#
#     def forward(self, x, x_mark):
#         print(x.shape)
#         x = self.odeblock2(x)
#         if x_mark is None:
#             x = self.value_embedding(x) + self.position_embedding(x)
#         else:
#             x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
#         x= self.dropout(x) #torch.Size([16, 4, 32])
#         x=self.odeblock1(x)
#         x= self.norm(x)
#         return x
#
# class ODEfunc1(torch.nn.Module):
#     def __init__(self,input):
#         super(ODEfunc1, self).__init__()
#         #可以比较卷积提取器，按道理就是提取特征没错
#         #torch.Size([16, 4, 64])，下一个就是trans应该层归一化一下
#         #为什么对64效果不好，对4效果好呢？
#         d_ff =  4 * input
#         self.net=nn.Sequential(nn.Linear(input,d_ff),
#                                nn.Tanh(),
#                                nn.Linear(d_ff,input))
#
#     def forward(self, t,x):
#         x = x.transpose(1,2)
#         x = self.net(x)
#         x = x.transpose(1,2)
#         return x
#
# class ODEfunc2(torch.nn.Module):
#     def __init__(self,input,dropout=0.1, activation="relu"):
#         super(ODEfunc2, self).__init__()
#         d_ff = 4 * input
#         self.conv1 = nn.Conv1d(in_channels=input, out_channels=d_ff, kernel_size=1)
#         self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=input, kernel_size=1)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = F.relu if activation == "relu" else F.gelu
#
#     def forward(self, t,x):
#        # x = x.transpose(1,2)
#       #  x = self.net(x)
#         x = self.dropout(self.activation(self.conv1(x)))
#         x = self.dropout(self.conv2(x))
#       #  x = x.transpose(1,2)
#         return x
#
# class ODEblock(nn.Module):
#     def __init__(self, ODEfunc):
#         super(ODEblock, self).__init__()
#         self.t = torch.tensor([0, 1]).float()
#         self.ODEfunc = ODEfunc
#     def forward(self, x):
#         z = odeint(self.ODEfunc, x, self.t, method='euler')[1]  # ODESolver
#         return z  # 梯度

