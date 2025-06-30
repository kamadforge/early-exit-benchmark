import argparse
import logging
import os
import sys

from torchvision.datasets import CIFAR10, CIFAR100, ImageNet
from torchvision.datasets.imagenet import ARCHIVE_META


class Imagenet:
    def __init__(self, data_dir, disk_dir) -> None:
        self.data_dir = data_dir
        self.disk_dir = disk_dir

    def load_to_memfs(self):
        logging.info(f'data_dir: {self.data_dir}, disk_dir: {self.disk_dir}')
        for tar_file, _ in ARCHIVE_META.values():
            symlink_path = os.path.join(self.data_dir, tar_file)
            if not os.path.islink(symlink_path):
                os.symlink(os.path.join(self.disk_dir, tar_file), symlink_path)
        logging.info(f'data_dir: {self.data_dir}, disk_dir: {self.disk_dir}')
        print('unpacking train set', file=sys.stderr)
        logging.info('unpacking train set')
        ImageNet(self.data_dir, split='train')
        print('unpacking val set', file=sys.stderr)
        logging.info('unpacking val set')
        ImageNet(self.data_dir, split='val')
        print('done unpacking imagenet at:', self.data_dir, file=sys.stderr)
        logging.info(f'done unpacking imagenet at: {self.data_dir}')


def main(args):
    parser = argparse.ArgumentParser(description='Load dataset to memory file system.')
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--disk_dir', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    args = parser.parse_args()

    if args.dataset == "imagenet1k":
        Imagenet(args.data_dir, args.disk_dir).load_to_memfs()
    elif args.dataset == "cifar10":
        print('downloading cifar10', file=sys.stderr)
        CIFAR10(
            root=args.data_dir,
            train=True,
            download=True,
        )
        CIFAR10(
            root=args.data_dir,
            train=False,
            download=True,
        )
    elif args.dataset == "cifar100":
        print('downloading cifar100', file=sys.stderr)
        CIFAR100(
            root=args.data_dir,
            train=True,
            download=True,
        )
        CIFAR100(
            root=args.data_dir,
            train=False,
            download=True,
        )


if __name__ == "__main__":
    main(sys.argv[1:])