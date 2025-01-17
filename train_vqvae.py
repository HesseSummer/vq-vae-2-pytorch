import argparse

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import logging
from myutils import init_logging

def train(epoch, loader, model, optimizer, scheduler, device, log):
    loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (img, label) in enumerate(loader):
        model.zero_grad()

        img = img.to(device)

        out, latent_loss = model(img)  # latent_loss即图片和对应编码的距离diff
        recon_loss = criterion(out, img)  # 即生成图和真实图的距离
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        mse_sum += recon_loss.item() * img.shape[0]
        mse_n += img.shape[0]

        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; '
                f'latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; '
                f'lr: {lr:.5f}'
            )
        )

        log(
            'epoch: %d; mse: %.5f; latent: %.3f; avg mse: %.5f; lr: %.5f',
            (epoch + 1), recon_loss.item(), latent_loss.item(), (mse_sum / mse_n), lr
        )

        if i % 100 == 0:
            model.eval()

            sample = img[:sample_size]  # 取sample_size张图片->(sample_size, 3, h, w)

            with torch.no_grad():
                out, _ = model(sample)

            utils.save_image(
                torch.cat([sample, out], 0),  # (2*sample_size, 3, h, w)
                f'sample/{str(epoch + 1).zfill(5)}_{str(i).zfill(5)}.png',  # zfill保正5个字符，不足补0，右对齐
                nrow=sample_size,  # 每行sample_size张图片
                normalize=True,
                range=(-1, 1),
            )

            model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=560)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sched', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = datasets.ImageFolder(args.path, transform=transform)
    loader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=4)

    model = nn.DataParallel(VQVAE()).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None  # len(loader)=batch size
        )

    log = init_logging(logging)  # log函数
    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device, log)
        torch.save(
            model.module.state_dict(), f'checkpoint/vqvae_{str(i + 1).zfill(3)}.pt'
        )
