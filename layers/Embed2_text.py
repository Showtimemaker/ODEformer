import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
import math
from torch.nn.utils import weight_norm

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

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
                                   kernel_size=3, padding=padding, padding_mode='circular')
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()
        minute_size = 4;
        hour_size = 24
        weekday_size = 7;
        day_size = 32;
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

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding1(nn.Module):
    def __init__(self, c_in, d_model,embed_type='fixed', freq='h', dropout=0.1,input=8):#+seq_len
        super(DataEmbedding1, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.odeblock1 = ODEblock(ODEfunc1(input))
        self.odeblock2 = ODEblock(ODEfunc2(input))
        self.odeblock3 = ODEblock(TemporalBlock(input, input))
        self.tcn = TemporalBlock(input, input)
        self.norm = nn.LayerNorm(64)

    def forward(self, x, x_mark):
     #   print(x.shape)
        x = self.tcn(x)
     #   x = self.odeblock3(x)
      #  print(x.shape)
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        x= self.dropout(x) #torch.Size([16, 4, 32])
       # x=self.odeblock1(x)
        return x

class DataEmbedding2(nn.Module):
    def __init__(self, c_in, d_model,embed_type='fixed', freq='h', dropout=0.1,input=10):
        super(DataEmbedding2, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                    freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
            d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)
        self.odeblock1 = ODEblock(ODEfunc1(input))
        self.odeblock2 = ODEblock(ODEfunc2(input))#卷积
        self.odeblock3=ODEblock(TemporalBlock(input,input))
        self.tcn=TemporalBlock(input,input)
        self.norm = nn.LayerNorm(64)

    def forward(self, x, x_mark):
       # print(x.shape)
        x = self.tcn(x)
      #  x=self.odeblock3(x)#TCN
       # print(x.shape)
        #torch.Size([16, 7, 1]), torch.Size([16, 7, 1])
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        x= self.dropout(x)
      #  x= self.odeblock1(x)#
       # x = self.norm(x)#进入之前
        return x


# 这个函数是用来修剪卷积之后的数据的尺寸，让其与输入数据尺寸相同。
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

 #这个就是TCN的基本模块，包含8个部分，两个（卷积+修剪+relu+dropout）
# 里面提到的downsample就是下采样，其实就是实现残差链接的部分。不理解的可以无视这个
#这一步没错。
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, padding=2, dropout=0.05):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size=1,
                                          padding=1, dilation=2))#1,2-k=1

        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size=1,
                                            padding=1, dilation=1))#1,1
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,#)
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self,x):
       # print(x.shape)torch.Size([16, 6, 1])

        out = self.net(x)

        res = x if self.downsample is None else self.downsample(x)

        return self.relu(out + res) #+ res)


class ODEfunc1(torch.nn.Module):
    def __init__(self,input):
        super(ODEfunc1, self).__init__()
        #可以比较卷积提取器，按道理就是提取特征没错
        #torch.Size([16, 4, 64])，下一个就是trans应该层归一化一下
        #为什么对64效果不好，对4效果好呢？
        d_ff =  4 * input
        self.net=nn.Sequential(nn.Linear(input,d_ff),
                               nn.Tanh(),
                               nn.Linear(d_ff,input))

    def forward(self, t,x):
        x = x.transpose(1,2)
        x = self.net(x)
        x = x.transpose(1,2)
        return x

class ODEfunc2(torch.nn.Module):
    def __init__(self,input,dropout=0.1, activation="relu"):
        super(ODEfunc2, self).__init__()
        d_ff = 4 * input
        self.conv1 = nn.Conv1d(in_channels=input, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=input, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, t,x):
        x = self.dropout(self.activation(self.conv1(x)))
        x = self.dropout(self.conv2(x))
        return x

class ODEblock(nn.Module):
    def __init__(self, ODEfunc):
        super(ODEblock, self).__init__()
        self.t = torch.tensor([0, 1]).float()
        self.ODEfunc = ODEfunc
    def forward(self, x):
        z = odeint(self.ODEfunc, x, self.t, method='euler')[1]  # ODESolver
        return z  # 梯度