import os
import json
import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
import torch

from util.data_util import sa_create


class ShapeNetPart(Dataset):
    def __init__(self, split='train', data_root='dataset/shapenetpart', transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.data_root, self.split, self.transform, self.shuffle_index, self.loop = data_root, split, transform, shuffle_index, loop

        self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
                            'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
                            'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
                            'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
                            'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}
        catefile = os.path.join(data_root, 'synsetoffset2category.txt')
        self.cate2id = {}
        self.id2cate = {}
        with open(catefile, 'r') as f:
            for line in f:
                ls = line.strip().split()
                self.cate2id[ls[0]] = ls[1]
                self.id2cate[ls[1]] = ls[0]
        self.cate2cls = dict(zip(self.cate2id, range(len(self.cate2id))))

        if split == 'train':
            with open(os.path.join(self.data_root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
                self.data_list = ['%s_%s' % (d.split('/')[1], d.split('/')[2]) for d in json.load(f)]
        elif split == 'val':
            with open(os.path.join(self.data_root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
                self.data_list = ['%s_%s' % (d.split('/')[1], d.split('/')[2]) for d in json.load(f)]
        elif split == 'test':
            with open(os.path.join(self.data_root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
                self.data_list = ['%s_%s' % (d.split('/')[1], d.split('/')[2]) for d in json.load(f)]
        else:
            raise NotImplementedError()

        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(self.data_root, '%s.txt' % item.replace('_', '/'))
                data = np.loadtxt(data_path)  # coord+normal+label, N*7
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def _data_prepare(self, coord, feat, label, transform=None, shuffle_index=False):
        if transform:
            coord, feat, label = transform(coord, feat, label)

        if shuffle_index:
            shuf_idx = np.arange(coord.shape[0])
            np.random.shuffle(shuf_idx)
            coord, feat, label = coord[shuf_idx], feat[shuf_idx], label[shuf_idx]

        coord_min = np.min(coord, 0)
        coord -= coord_min
        coord = torch.FloatTensor(coord)
        feat = torch.FloatTensor(feat)
        label = torch.LongTensor(label)
        return coord, feat, label

    def _to_categorical(self, cate_cls, num_classes):
        new_cate_cls = torch.eye(num_classes)[cate_cls]
        return new_cate_cls

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]

        item_name = self.data_list[data_idx]
        cate = self.id2cate[item_name.split('_')[0]]
        cate_cls = self.cate2cls[cate]
        cate_cls = self._to_categorical(cate_cls, len(self.id2cate))

        data = SA.attach("shm://{}".format(item_name)).copy()
        coord, feat, label = data[:, 0:3], data[:, 3:6], data[:, 6]
        coord, feat, label = self._data_prepare(coord, feat, label, self.transform, self.shuffle_index)
        return coord, feat, label, cate_cls

    def __len__(self):
        return len(self.data_idx) * self.loop


# for test
if __name__ == '__main__':
    snp = ShapeNetPart(split='val')
    print(snp.cate2cls)
