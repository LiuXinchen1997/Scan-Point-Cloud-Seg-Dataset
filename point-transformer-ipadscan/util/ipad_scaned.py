import os

import numpy as np
import SharedArray as SA
from torch.utils.data import Dataset
from glob import glob

from util.data_util import sa_create
from util.data_util import data_prepare


class IPAD_SCANED(Dataset):
    def __init__(self, split='train', data_root='dataset/ipad_scaned/label_data', voxel_size=0.04, voxel_max=None, transform=None, shuffle_index=False, loop=1):
        super().__init__()
        self.split, self.voxel_size, self.transform, self.voxel_max, self.shuffle_index, self.loop = split, voxel_size, transform, voxel_max, shuffle_index, loop
        if split == 'train':
            self.data_list = glob(os.path.join(data_root, 'train', '*.npy'))
            self.data_list = [os.path.basename(file).split('.')[0] for file in self.data_list]
        else:
            self.data_list = glob(os.path.join(data_root, 'test', '*.npy'))
            self.data_list = [os.path.basename(file).split('.')[0] for file in self.data_list]
        
        for item in self.data_list:
            if not os.path.exists("/dev/shm/{}".format(item)):
                data_path = os.path.join(data_root, item + '.npy')
                data = np.load(data_path)  # xyl, N*4
                sa_create("shm://{}".format(item), data)
        self.data_idx = np.arange(len(self.data_list))
        print("Totally {} samples in {} set.".format(len(self.data_idx), split))

    def __getitem__(self, idx):
        data_idx = self.data_idx[idx % len(self.data_idx)]
        data = SA.attach("shm://{}".format(self.data_list[data_idx])).copy()
        coord, label = data[:, 0:3], data[:, 3]

        self.voxel_size, self.voxel_max = None, None
        coord, label = data_prepare(coord, label, self.split, self.voxel_size, self.voxel_max, self.transform, self.shuffle_index)
        
        return coord, label

    def __len__(self):
        return len(self.data_idx) * self.loop


if __name__ == '__main__':
    points =  np.load('./dataset/s3dis/trainval_fullarea/Area_4_office_14.npy')
    coord = points[:, :3]
    label = points[:, -1]
    
    for i in range(13):
        if i in [0, 1, 2, 3, 4, 5, 6]:
            label[label == i] = 0
        else:
            label[label == i] = 1
    
    print(label[label == 0].shape)
    print(label[label == 1].shape)
    print(label.shape)
