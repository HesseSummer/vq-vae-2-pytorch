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
warnings.filterwarnings("ignore")  # 忽略警告

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
            for i,f in enumerate(self.imgs):
                X.append(imread(f, self.mode))
                if len(X) == self.batch_size or i == len(self.imgs)-1:
                    X = np.array(X)
                    if self.mode == 'gan':
                        Z = np.random.randn(len(X), z_dim)
                        yield [X, Z], None
                    elif self.mode == 'fid':
                        yield X
                    X = []

if not os.path.exists('samples'):
    os.mkdir('samples')

imgs = glob.glob('../../CelebA-HQ/train/*.png')  # train文件下的 所有图片的路径列表
np.random.shuffle(imgs)

img_dim = 128  # 图片宽、高
z_dim = 128  # 一维噪声的维度
num_layers = int(np.log2(img_dim)) - 3  # 4。层数和输入图片的尺寸有关，不过为虾米是这个关系？理论是？
max_num_channels = img_dim * 8  # 1024。最大通道数也和输入图片的尺寸有关，不过同样，理论是？
f_size = img_dim // 2**(num_layers + 1)  # 4。z先经过全连接映射到(f_size, f_size, max_num_channels)，再用来生成
batch_size = 64

# 编码器
x_in = Input(shape=(img_dim, img_dim, 3))
x = x_in

for i in range(num_layers + 1):   # 五次卷积（BN+激活），同卷积核同步长，通道变化：3->64->128->256->512->1024
    num_channels = max_num_channels // 2**(num_layers - i)
    x = Conv2D(num_channels,
               (5, 5),
               strides=(2, 2),
               padding='same',
               kernel_initializer=RandomNormal(0, 0.02))(x)
    if i > 0:
        x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

x = Flatten()(x)
x = Dense(z_dim,  # 注意：全连接到(bs, 128)
          kernel_initializer=RandomNormal(0, 0.02))(x)

e_model = Model(x_in, x)
e_model.summary()


# 生成器
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


def correlation(x, y):  # z_in, z_fake，均为(bs, z_dim)
    x = x - K.mean(x, 1, keepdims=True)  # (bs, z_dim)-(bs, 1)
    y = y - K.mean(y, 1, keepdims=True)  # (bs, z_dim)-(bs, 1)
    x = K.l2_normalize(x, 1)  # (bs, z_dim)
    y = K.l2_normalize(y, 1)  # (bs, z_dim)
    return K.sum(x * y, 1, keepdims=True)  # (bs, 1)


# 整合模型
x_in = Input(shape=(img_dim, img_dim, 3))
x_real = x_in
z_real = e_model(x_real)  # (bs, z_dim)
z_real_mean = K.mean(z_real, 1, keepdims=True)  # (bs, 1)

z_in = Input(shape=(z_dim, ))
x_fake = g_model(z_in)
z_fake = e_model(x_fake)  # (bs, z_dim)
z_fake_mean = K.mean(z_fake, 1, keepdims=True)  # (bs, 1)

x_fake_ng = Lambda(K.stop_gradient)(x_fake)  # 固定G
z_fake_ng = e_model(x_fake_ng)
z_fake_ng_mean = K.mean(z_fake_ng, 1, keepdims=True)

train_model = Model([x_in, z_in],
                    [z_real, z_fake, z_fake_ng])

t1_loss = z_real_mean - z_fake_ng_mean  # Dloss，给avg(z_real)0，给avg(z_fake)1（固定G）
t2_loss = z_fake_mean - z_fake_ng_mean  # Gloss，企图欺骗让avg(z_fake)0
z_corr = correlation(z_in, z_fake)  # Pearson相关系数
qp_loss = 0.25 * t1_loss[:, 0]**2 / K.mean((x_real - x_fake_ng)**2, axis=[1, 2, 3])

train_model.add_loss(K.mean(t1_loss + t2_loss - 0.5 * z_corr) + K.mean(qp_loss))
train_model.compile(optimizer=RMSprop(1e-4, 0.99))
train_model.metrics_names.append('t_loss')
train_model.metrics_tensors.append(K.mean(t1_loss))
train_model.metrics_names.append('z_corr')
train_model.metrics_tensors.append(K.mean(z_corr))

# 检查模型结构
train_model.summary()


# 采样函数
def sample(path, n=9, z_samples=None):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    if z_samples is None:
        z_samples = np.random.randn(n**2, z_dim)
    for i in range(n):
        for j in range(n):
            z_sample = z_samples[[i * n + j]]
            x_sample = g_model.predict(z_sample)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


# 重构采样函数
def sample_ae(path, n=8):
    figure = np.zeros((img_dim * n, img_dim * n, 3))
    for i in range(n):
        for j in range(n):
            if j % 2 == 0:
                x_sample = [imread(np.random.choice(imgs))]
            else:
                z_sample = e_model.predict(np.array(x_sample))
                z_sample -= (z_sample).mean(axis=1, keepdims=True)
                z_sample /= (z_sample).std(axis=1, keepdims=True)
                x_sample = g_model.predict(z_sample * 0.9)
            digit = x_sample[0]
            figure[i * img_dim:(i + 1) * img_dim,
                   j * img_dim:(j + 1) * img_dim] = digit
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    imageio.imwrite(path, figure)


class Trainer(Callback):
    def __init__(self):
        self.batch = 0
        self.n_size = 9
        self.iters_per_sample = 100
        self.Z = np.random.randn(self.n_size**2, z_dim)
    def on_batch_end(self, batch, logs=None):
        if self.batch % self.iters_per_sample == 0:
            sample('samples/test_%s.png' % self.batch,
                self.n_size, self.Z)
            sample_ae('samples/test_ae_%s.png' % self.batch)
            train_model.save_weights('./train_model.weights')
        self.batch += 1


if __name__ == '__main__':

    trainer = Trainer()
    img_data = img_generator(imgs, 'gan', batch_size)

    train_model.fit_generator(img_data.__iter__(),
                              steps_per_epoch=len(img_data),
                              epochs=1000,
                              callbacks=[trainer])
