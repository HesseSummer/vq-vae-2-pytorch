import torch
from torch import nn
from torch.nn import functional as F


# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


# Borrowed from https://github.com/deepmind/sonnet and ported it to PyTorch


class Quantize(nn.Module):
    """
    输入：input的通道数，也是词向量的长度【64】、词向量的个数【512】
    功能：对输入input的每个像素位置，找到词向量表中，最接近该像素本身的词向量
    输出：每个像素位置对应的词向量quantize(h, w, 64)、每个像素位置对应的差异diff(h, w, 64)、每个像素位置对应的词向量的索引（非one-hot）embed_ind(h, w)

    算法：
        网络初始化：随机初始化词向量表embed(64, 512)、0初始化cluster_size(512,)、embed_avg初始化同词向量表(64, 512)
        每次前向传播：
            1) 求每个像素（<拓长>到词向量长度64）和词向量表中<每个>词向量（长度64）的距离dist(x, 512)（每个元素都是像素和词向量的距离），
               并找出每个像素对应的所有词向量的距离中最小的词向量的索引embed_ind(x,)=>(h, w)，
            2) 根据索引，在词向量表中找到对应的词向量quantize(h, w, 64)，并求出quantize和input的差异diff(h, w, 64)
            3) 返回1)2)的结果embed_ind、quantize、diff
            4) 在训练时还更新embed：
                4.1) 将embed_ind按512one-hot编码得到embed_onehot(x, 512)，进而得到每个词向量的总使用频次，更新cluster_size
                4.2) 将embed_onehot拓展到64个通道得到embed_sum，进而得到各个通道上每个词向量的使用频次，更新embed_avg
                4.3) embed_avg和cluster_size做除法，得到各个通道上每个词向量的平均使用频次，将该值更新（加）进embed
    """
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        embed = torch.randn(dim, n_embed)  # e.g. (64, 512)
        self.register_buffer('embed', embed)  # (64, 512)
        self.register_buffer('cluster_size', torch.zeros(n_embed))  # (512,)
        self.register_buffer('embed_avg', embed.clone())  # (64, 512)

    def forward(self, input):
        flatten = input.reshape(-1, self.dim)  # (x, 64)
        # dist是(x, 512),含义是input(flatten)和embed的距离
        dist = (
            flatten.pow(2).sum(1, keepdim=True)  # (x, 1)每个像素位置的各通道上的所有像素的平方相加
            - 2 * flatten @ self.embed  # (x, 64)@(64, 512)=(x, 512) 横着看：每个像素位置对应的词向量
            + self.embed.pow(2).sum(0, keepdim=True)  # (1, 512) 所有通道的词向量平方再相加
        )
        _, embed_ind = (-dist).max(1)  # 为什是-dist.max，而不是dist.min？ 得到(x,)，每个元素是索引
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)  # n_embed必须大于embed_ind的x （x, 512）
        embed_ind = embed_ind.view(*input.shape[:-1])  # embed_ind和input的前两个维度一样
        # 上一行，view的东西，应该是*input.shape[1:]吧？
        quantize = self.embed_code(embed_ind)  # (x, 64) 找到每个单词对应的词嵌入

        if self.training:
            # 改变cluster_size，不知道为啥这样操作 (512,)
            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, embed_onehot.sum(0)  # 这两个参数相乘，再加到前面
            )
            embed_sum = flatten.transpose(0, 1) @ embed_onehot  # （64, x）@(x, 512)=(64,512)
            self.embed_avg.data.mul_(self.decay).add_(1 - self.decay, embed_sum)  # (64, 512)和cluster_size的算法略不同
            n = self.cluster_size.sum()
            # 再精细调整下cluster_size
            cluster_size = (
                (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            )
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)  # normalize embed
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()  # 取均值了，是标量
        quantize = input + (quantize - input).detach()  # 为啥不直接detach，还绕个弯？

        return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))  # 从embed中查找embed_id对应的词嵌入


class ResBlock(nn.Module):
    def __init__(self, in_channel, channel):
        super().__init__()

        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, channel, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel, in_channel, 1),
        )

    def forward(self, input):
        out = self.conv(input)
        out += input

        return out


class Encoder(nn.Module):
    """
    参数：n_res_block决定有几个参差块、stride决定残差网络之前有几个全卷积
    功能：简单地前向传播实现Encoder，in_channel到channel
    """
    def __init__(self, in_channel, channel, n_res_block, n_res_channel, stride):
        super().__init__()

        if stride == 4:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel, 3, padding=1),
            ]

        elif stride == 2:
            blocks = [
                nn.Conv2d(in_channel, channel // 2, 4, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 2, channel, 3, padding=1),
            ]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class Decoder(nn.Module):
    """
    参数：n_res_block决定有几个参差块、stride决定残差网络之后有几个反卷积
    功能：简单地前向传播实现Decoder，in_channel到out_channel
    """
    def __init__(
        self, in_channel, out_channel, channel, n_res_block, n_res_channel, stride
    ):
        super().__init__()

        blocks = [nn.Conv2d(in_channel, channel, 3, padding=1)]

        for i in range(n_res_block):
            blocks.append(ResBlock(channel, n_res_channel))

        blocks.append(nn.ReLU(inplace=True))

        if stride == 4:
            blocks.extend(
                [
                    nn.ConvTranspose2d(channel, channel // 2, 4, stride=2, padding=1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(
                        channel // 2, out_channel, 4, stride=2, padding=1
                    ),
                ]
            )

        elif stride == 2:
            blocks.append(
                nn.ConvTranspose2d(channel, out_channel, 4, stride=2, padding=1)
            )

        self.blocks = nn.Sequential(*blocks)

    def forward(self, input):
        return self.blocks(input)


class VQVAE(nn.Module):
    """

    """
    def __init__(
        self,
        in_channel=3,
        channel=128,
        n_res_block=2,
        n_res_channel=32,
        embed_dim=64,
        n_embed=512,
        decay=0.99,
    ):
        super().__init__()

        self.enc_b = Encoder(in_channel, channel, n_res_block, n_res_channel, stride=4)  # 3通道变128通道-Eb
        self.enc_t = Encoder(channel, channel, n_res_block, n_res_channel, stride=2)  # 128通道变128通道-Et
        self.quantize_b = Quantize(embed_dim, n_embed)  # Quantize实例-Qb
        self.quantize_t = Quantize(embed_dim, n_embed)  # Quantize实例-Qt
        self.dec = Decoder(  # (64+64)通道到3通道-D
            embed_dim + embed_dim,
            in_channel,
            channel,
            n_res_block,
            n_res_channel,
            stride=4,
        )
        self.dec_t = Decoder(
            embed_dim, embed_dim, channel, n_res_block, n_res_channel, stride=2  # 64通道到64通道-Dt
        )
        self.quantize_conv_b = nn.Conv2d(embed_dim + channel, embed_dim, 1)  # (64+128)通道到64通道-Cb
        self.quantize_conv_t = nn.Conv2d(channel, embed_dim, 1)  # 128通道到64通道-简单卷积-Ct
        self.upsample_t = nn.ConvTranspose2d(
            embed_dim, embed_dim, 4, stride=2, padding=1  # 64通道到64通道-简单反卷积-UCt
        )

    def forward(self, input):
        """
        :param input: 图片(bs, 3, imgdim, imgdim)
        :return: 拟合的图片(bs, 3, imgdim, imgdim)
        """
        quant_t, quant_b, diff, _, _ = self.encode(input)
        dec = self.decode(quant_t, quant_b)

        return dec, diff

    def encode(self, input):
        enc_b = self.enc_b(input)  # 3->128
        enc_t = self.enc_t(enc_b)  # 128->128

        quant_t = self.quantize_conv_t(enc_t).permute(0, 2, 3, 1)  # 128->64，将通道放在最后一维
        quant_t, diff_t, id_t = self.quantize_t(quant_t)  # (bs, h, w, 64)、scalar、(bs, h, w)每个元素都是索引对应着词向量表
        quant_t = quant_t.permute(0, 3, 1, 2)  # quant_t再交换回轴
        diff_t = diff_t.unsqueeze(0)  # 给标量diff增加一个维度

        dec_t = self.dec_t(quant_t)  # 64->64
        enc_b = torch.cat([dec_t, enc_b], 1)  # ❤ ❤ 将quant_t的解码64和input的编码128 cat到一起->64+128

        quant_b = self.quantize_conv_b(enc_b).permute(0, 2, 3, 1)  # (64+128)->64
        quant_b, diff_b, id_b = self.quantize_b(quant_b)  # 64->(bs, h, w, 64)、scalar、(bs, h, w)
        quant_b = quant_b.permute(0, 3, 1, 2)  # quant_b再交换回轴
        diff_b = diff_b.unsqueeze(0)  # 给标量diff增加一个维度

        return quant_t, quant_b, diff_t + diff_b, id_t, id_b  # 第三个返回值是[scalar1, scalar2]

    def decode(self, quant_t, quant_b):
        upsample_t = self.upsample_t(quant_t)  # 64->64
        quant = torch.cat([upsample_t, quant_b], 1)  # quant_t的反卷积和quant_b cat起来->64+64
        dec = self.dec(quant)  # 64+64->3

        return dec

    def decode_code(self, code_t, code_b):
        """
        类内没有使用该函数的地方
        """
        quant_t = self.quantize_t.embed_code(code_t)
        quant_t = quant_t.permute(0, 3, 1, 2)
        quant_b = self.quantize_b.embed_code(code_b)
        quant_b = quant_b.permute(0, 3, 1, 2)

        dec = self.decode(quant_t, quant_b)

        return dec
