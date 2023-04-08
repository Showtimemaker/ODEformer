import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

#使用了NODE-spilt 组件：具体在

class ConvLayer(nn.Module):
    def __init__(self, c_in,seq_len=6):
        super(ConvLayer, self).__init__()

        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,#卷积核的尺寸
                                  padding=2,#卷积步长
                                  padding_mode='circular')#循环的进行填充
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)#3


    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)

        self.ODEfunc = ODEfunc1
        self.odeblock = ODEblock(ODEfunc1(d_model, d_ff=None))
        self.odeblock1 = ODEblock(ODEfunc2(d_model//2))
        self.norm1 = nn.LayerNorm(d_model//2)#砍半了，还没合并
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):

        #切分处理:实现分成两半。
        split_x = torch.split(x, x.shape[2] // 2, dim=2)

        x1 = split_x[1].clone()
        x2 = split_x[0].clone()
  #      print("x2",x2.shape)
        #这两个融进去就可以做
        # norm_x = self.conv3(x2.permute(0, 2, 1))
        # x2 = norm_x.transpose(1, 2)
      #  print("1",x2.shape)
        x2=self.odeblock(x2)  #CNN+NODE
       # print("1",x2.shape)

        new_x, attn = self.attention(
            x1, x1, x1,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )

      #  print("success")
        x1 = x1 + self.dropout(new_x)
        x1 = self.norm1(x1)
       # print(x1.shape,x2.shape)
        x = torch.cat((x2, x1), 2)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class ODEfunc1(torch.nn.Module):
    def __init__(self,d_model, d_ff=None):
        super(ODEfunc1, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        d_ff = d_ff or 4 * d_model
        self.conv3 = nn.Conv1d(in_channels=d_model//2, out_channels=d_model//2, kernel_size=1)
        #就是说我可不可以把他映射到高维再蒸馏下来
    def forward(self, t, x):
        x = x.transpose(1, 2)
        norm_x = self.conv3(x)#.permute(0, 2, 1)
        x = norm_x.transpose(1, 2)
        return x

class ODEfunc2(torch.nn.Module):
    def __init__(self,c_in):
        super(ODEfunc2, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        #就是说我可不可以把他映射到高维再蒸馏下来,卷积+池化
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,#卷积核的尺寸
                                  padding_mode='circular')#循环的进行填充
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, t, x):
        x = self.downConv(x.permute(0, 2, 1))  # 下采样
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        print(x.shape)
        return x

class ODEblock(nn.Module):
    def __init__(self, ODEfunc):
        super(ODEblock, self).__init__()
        self.t = torch.tensor([0, 1]).float()
        self.ODEfunc = ODEfunc

    def forward(self, x):
        z = odeint(self.ODEfunc, x, self.t, method='euler')[1]  # ODESolver
        return z  # 梯度



class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)

            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)

            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns

class DecoderLayer1(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer1, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.ODEfunc = ODEfunc1
        self.odeblock1 = ODEblock(ODEfunc1(d_model, d_ff=None))
        self.norm1 = nn.LayerNorm(d_model//2)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        #切分处理:实现分成两半。
        split_x1 = torch.split(x, x.shape[2] // 2, dim=2)

        x1_1 = split_x1[1].clone()#Attention-1
        x1_2 = split_x1[0].clone()#CNN-1

        #
        x3_C=self.odeblock1(x1_2)#L/2

        x1_1 = x1_1 + self.dropout(self.self_attention(
            x1_1, x1_1, x1_1,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])

        x1_1 = self.norm1(x1_1)

        x3_A=x1_1
        x3 = torch.cat((x3_C, x3_A), 2)

        x3 = x3 + self.dropout(self.cross_attention(
             x3, cross, cross,
             attn_mask=cross_mask,
             tau=tau, delta=delta
         )[0])

        y = x3

        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)

class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x
#常规
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None, tau=None, delta=None):

        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])

        x = self.norm1(x)

        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm3(x + y)