import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset, CodeRow
from vqvae import VQVAE


def extract(lmdb_env, loader, model, device):
    index = 0

    with lmdb_env.begin(write=True) as txn:  # 准备操作数据库
        pbar = tqdm(loader)

        for img, _, filename in pbar:
            img = img.to(device)

            _, _, _, id_t, id_b = model.encode(img)
            id_t = id_t.detach().cpu().numpy()  # index是和图片同规模（增加一个维度），每个元素是词向量表中的一个词向量
            id_b = id_b.detach().cpu().numpy()

            for file, top, bottom in zip(filename, id_t, id_b):
                row = CodeRow(top=top, bottom=bottom, filename=file)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))  # 最后把图片的总个数也写进数据库


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = 'cuda'

    transform = transforms.Compose(
        [
            transforms.Resize(args.size),
            transforms.CenterCrop(args.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    dataset = ImageFileDataset(args.path, transform=transform)  # 继承自ImageFolder，每次返回 sample, target, filename('classname/imgname.jpg')
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    model = VQVAE()
    model.load_state_dict(torch.load(args.ckpt))
    model = model.to(device)
    model.eval()  # 将model设置为测试模式

    map_size = 100 * 1024 * 1024 * 1024

    # 创建数据库
    env = lmdb.open(args.name, map_size=map_size)  # 数据库的最大容量是100 * 1024 * 1024 * 1024 B = 100 G

    extract(env, loader, model, device)
