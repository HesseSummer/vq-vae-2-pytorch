#! -*- coding: utf-8 -*-

import numpy as np
import scipy as sp
from scipy import misc
import glob
import imageio
from keras.models import Model
from keras.layers import *
from keras import backend as K
from keras.optimizers import RMSprop
from keras.callbacks import Callback
from keras.initializers import RandomNormal
import os
import warnings
warnings.filterwarnings("ignore")  # 忽略keras带来的满屏警告


def imread(f, mode='gan'):
    """
    :param f: 单张图片路径
    :param mode: gan或者fid
    :return: resize后的numpy array
    """
    x = misc.imread(f, mode='RGB')
    if mode == 'gan':
        x = misc.imresize(x, (img_dim, img_dim))
        x = x.astype(np.float32)
        return x / 255 * 2 - 1
    elif mode == 'fid':
        x = misc.imresize(x, (299, 299))
        return x.astype(np.float32)


class img_generator:
    """图片迭代器，方便重复调用
    每次迭代返回一个batch的[真实图片，128维随机噪声]（当mode取'gan'时）
    或者返回一个batch的[真实图片]（当mode取'fid'时）
    """
    def __init__(self, imgs, mode='gan', batch_size=64):
        self.imgs = imgs
        self.batch_size = batch_size
        self.mode = mode
        if len(imgs) % batch_size == 0:
            self.steps = len(imgs) // batch_size
        else:
            self.steps = len(imgs) // batch_size + 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        X = []
        while True:
            np.random.shuffle(self.imgs)
            for i, f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    if self.mode == 'gan':
                        Z = np.random.randn(len(X), z_dim)
                        yield [X, Z], None
                    elif self.mode == 'fid':
                        yield X
                    X = []


img_dim = 128  # 图片宽、高
z_dim = 128  # 一维噪声的维度
num_layers = int(np.log2(img_dim)) - 3  # 4。层数和输入图片的尺寸有关，不过为虾米是这个关系？理论是？
max_num_channels = img_dim * 8  # 1024。最大通道数也和输入图片的尺寸有关，不过同样，理论是？
f_size = img_dim // 2**(num_layers + 1)  # 4。z先经过全连接映射到(f_size, f_size, max_num_channels)，再用来生成
batch_size = 64

# # 以下为判别器
x_in = Input(shape=(img_dim, img_dim, 3))  # from keras.layers import Input
x = x_in

for i in range(num_layers + 1):  # 五次卷积（BN+激活），同卷积核同步长，通道变化：3->64->128->256->512->1024
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               use_bias=False,
               padding='same',
               kernel_initializer=RandomNormal(0, 0.02))(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(1, use_bias=False,  # 全连接到(bs, 1)
          kernel_initializer=RandomNormal(0, 0.02))(x)

d_model = Model(x_in, x)
d_model.summary()


# # 以下为生成器
z_in = Input(shape=(z_dim, ))
z = z_in

z = Dense(f_size**2 * max_num_channels,  # 全连接到f_size^2 * max_num_channels
          kernel_initializer=RandomNormal(0, 0.02))(z)
z = Reshape((f_size, f_size, max_num_channels))(z)  # 再reshape到(f_size, f_size, max_num_channels)
z = BatchNormalization()(z)
z = Activation('relu')(z)

for i in range(num_layers):  # 四次反卷积（BN、激活），卷积核和步长均相同，通道变化：1024->512->256->128->64
    num_channels = max_num_channels // 2**(i + 1)
    z = Conv2DTranspose(num_channels,
                        (5, 5),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=RandomNormal(0, 0.02))(z)
    z = BatchNormalization()(z)
    z = Activation('relu')(z)

z = Conv2DTranspose(3,
                    (5, 5),
                    strides=(2, 2),
                    padding='same',
                    kernel_initializer=RandomNormal(0, 0.02))(z)
z = Activation('tanh')(z)

g_model = Model(z_in, z)
g_model.summary()


# 整合模型
x_in = Input(shape=(img_dim, img_dim, 3))
x_real = x_in
x_real_score = d_model(x_real)

z_in = Input(shape=(z_dim, ))
x_fake = g_model(z_in)
x_fake_score = d_model(x_fake)

x_fake_ng = Lambda(K.stop_gradient)(x_fake)  # 停止G的梯度 -> 固定G
x_fake_ng_score = d_model(x_fake_ng)

train_model = Model([x_in, z_in],  # 输入
                    [x_real_score, x_fake_score, x_fake_ng_score])  # 输出

d_loss = K.relu(1 + x_real_score) + K.relu(1 - x_fake_ng_score)  # 真实图0，生成图1
g_loss = x_fake_score - x_fake_ng_score  # 企图欺骗使 生成图0

train_model.add_loss(K.mean(d_loss + g_loss))  # 1:1优化G、D
train_model.compile(optimizer=RMSprop(1e-4, 0.99))

# 检查模型结构
train_model.summary()


# 采样函数
def sample(path, n=9, z_samples=None):
    """
    使用随机噪声、训练好的G，随机产生n*n张图片（放到一个图中），存入path
    """
    figure = np.zeros((img_dim * n, img_dim * n, 3))  # 横着n张竖着n张，一共n^2（81）张图片
    if z_samples is None:  # 如果没有传入噪声，重新初始化噪声
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]  # 取出一行噪声(z_dim,)，然后增加一个维度变成(1, z_dim)，1表示batch size为1
            x_sample = g_model.predict(z_sample)  # 得到图片(1, 3, img_dim, img_dim)，1表示batch size为1
            digit = x_sample[0]  # digit(3, img_dim, img_dim)
            figure[i*img_dim: (i+1)*img_dim,
                   j*img_dim: (j+1)*img_dim] = digit
    figure = (figure + 1) / 2 * 255  # 网络输出的digit的每个值都是[-1, 1]之间的，所以这样回到[0, 255]
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9  # sample时sample多少张图片
        self.iters_per_sample = 100  # 每多少次迭代sample一次
        self.Z = np.random.randn(self.n_size**2, z_dim)  # 用于sample的随机噪声

    def on_batch_end(self, batch, logs=None):
        """sample图片、保存模型、batch加一"""
        if self.batch % self.iters_per_sample == 0:
            sample('samples/test_%s.png' % self.batch,
                self.n_size, self.Z)
            train_model.save_weights('./train_model.weights')
        self.batch += 1


if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('../../CelebA-HQ/train/*.png')  # train文件下的 所有图片的路径列表
np.random.shuffle(imgs)

if __name__ == '__main__':

    trainer = Trainer()
    img_data = img_generator(imgs, 'gan', batch_size)

    train_model.fit_generator(img_data.__iter__(),  # img_data.__iter__()返回[真实图片，128维随机噪声]，刚好是train_model的输入
                              steps_per_epoch=len(img_data),
                              epochs=1000,
                              callbacks=[trainer])  # list of callbacks to be called during training
