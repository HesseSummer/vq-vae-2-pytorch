import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging
from myutils import init_logging
from tensorboardX import SummaryWriter


try:
    from apex import amp

except ImportError:
    amp = None

from dataset import LMDBDataset
from pixelsnail import PixelSNAIL
from scheduler import CycleScheduler


def train(args, epoch, loader, model, optimizer, scheduler, device, log, writer):
    loader = tqdm(loader)

    criterion = nn.CrossEntropyLoss()

    for i, (top, bottom, label) in enumerate(loader):
        model.zero_grad()

        top = top.to(device)

        if args.hier == 'top':
            target = top
            out, _ = model(top)

        elif args.hier == 'bottom':
            bottom = bottom.to(device)
            target = bottom
            out, _ = model(bottom, condition=top)

        loss = criterion(out, target)
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        _, pred = out.max(1)
        correct = (pred == target).float()
        accuracy = correct.sum() / target.numel()

        lr = optimizer.param_groups[0]['lr']

        log('epoch: %d; loss: %.5f; acc: %.5f; lr: %.5f',
            (epoch+1), loss.item(), accuracy, lr)

        loader.set_description(
            (
                f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                f'acc: {accuracy:.5f}; lr: {lr:.5f}'
            )
        )

        niter = args.batch * epoch + i
        writer.add_scalar('Train/Loss', loss.item(), niter)
        writer.add_scalar('Train/Acc', accuracy, niter)


def test(args, epoch, loader, model, device, log, writer):
    loader = tqdm(loader)
    criterion = nn.CrossEntropyLoss()

    test_loss = 0
    test_correct = 0
    total = 0

    for i, (top, bottom, label) in enumerate(loader):
        with torch.no_grad():

            top = top.to(device)

            if args.hier == 'top':
                target = top
                out, _ = model(top)

            elif args.hier == 'bottom':
                bottom = bottom.to(device)
                target = bottom
                out, _ = model(bottom, condition=top)

            loss = criterion(out, target)
            test_loss += loss.item() * target.numel()

            _, pred = out.max(1)
            correct = (pred == target).float()
            test_correct += correct
            accuracy = correct.sum() / target.numel()

            total += target.numel()

            loader.set_description(
                (
                    f'epoch: {epoch + 1}; loss: {loss.item():.5f}; '
                    f'acc: {accuracy:.5f}; '
                )
            )

            niter = args.batch * epoch + i
            writer.add_scalar('Test/Loss', loss.item(), niter)
            writer.add_scalar('Test/Acc', accuracy, niter)

    test_loss = test_loss / total
    test_acc = test_correct / total
    log('epoch: %d; test_loss: %.5f; test_acc: %.5f;',
        (epoch + 1), test_loss, test_acc)


class PixelTransform:
    def __init__(self):
        pass

    def __call__(self, input):
        ar = np.array(input)

        return torch.from_numpy(ar).long()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--epoch', type=int, default=420)
    parser.add_argument('--hier', type=str, default='top')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--channel', type=int, default=256)
    parser.add_argument('--n_res_block', type=int, default=4)
    parser.add_argument('--n_res_channel', type=int, default=256)
    parser.add_argument('--n_out_res_block', type=int, default=0)
    parser.add_argument('--n_cond_res_block', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--amp', type=str, default='O0')
    parser.add_argument('--sched', type=str)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    print(args)

    device = 'cuda'

    dataset = LMDBDataset(args.path)
    loader = DataLoader(
        dataset, batch_size=args.batch, shuffle=True, num_workers=4, drop_last=True
    )

    ckpt = {}

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        args = ckpt['args']

    if args.hier == 'top':
        model = PixelSNAIL(
            [32, 32],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
        )

    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [64, 64],
            512,
            args.channel,
            5,
            4,
            args.n_res_block,
            args.n_res_channel,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
        )

    if 'model' in ckpt:
        model.load_state_dict(ckpt['model'])

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if amp is not None:
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.amp)

    model = nn.DataParallel(model)
    model = model.to(device)

    scheduler = None
    if args.sched == 'cycle':
        scheduler = CycleScheduler(
            optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
        )

    logging = init_logging('log')
    log = logging.log
    writer = SummaryWriter('log/board')
    for i in range(args.epoch):
        log('--' * 50)
        train(args, i, loader, model, optimizer, scheduler, device, log, writer)
        log('--' * 50)
        test(args, i, loader, model, device, log, writer)
        torch.save(
            {'model': model.module.state_dict(), 'args': args},
            f'checkpoint/pixelsnail_{args.hier}_{str(i + 1).zfill(3)}.pt',
        )
