import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from glob import glob

from util import config
from util.common_util import check_makedirs
from util.voxelize import voxelize

random.seed(123)
np.random.seed(123)


def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/ipad_scaned/ipad_scaned_baseline.yaml', help='config file')
    parser.add_argument('opts', help='see config/ipad_scaned/ipad_scaned_baseline.yaml for all options', default=None, nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


def main():
    global args
    args = get_parser()
    print(args)
    assert args.classes > 1
    print("=> creating model ...")
    print("Classes: {}".format(args.classes))

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.test_gpu)

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim).cuda()
    print(model)
    if os.path.isfile(args.model_path):
        print("=> loading checkpoint '{}'".format(args.model_path))
        checkpoint = torch.load(args.model_path)
        state_dict = checkpoint['state_dict']
        new_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=True)
        print("=> loaded checkpoint '{}' (epoch {})".format(args.model_path, checkpoint['epoch']))
        args.epoch = checkpoint['epoch']
    else:
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.model_path))
    test(model)


def data_prepare():
    data_list = [os.path.basename(file).split('.')[0] for file in glob(os.path.join(args.data_root, '*.npy'))]
    print('*******', data_list)
    return data_list


def data_load(data_name):
    data_path = os.path.join(args.data_root, '%s.npy' % data_name)
    data = np.load(data_path)  # xyzl, N*4
    coord = data[:, :3]

    idx_data = []
    idx_data.append(np.arange(coord.shape[0]))
    return coord, idx_data


def input_normalize(coord):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    return coord


def test(model):
    print('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    args.batch_size_test = 1
    model.eval()

    check_makedirs(args.save_folder)
    pred_save, label_save = [], []
    data_list = data_prepare()
    for idx, item in enumerate(data_list):
        pred_save_path = os.path.join(args.save_folder, '{}_{}_pred.npy'.format(item, args.epoch))
        coord, idx_data = data_load(item)
        pred = torch.zeros((coord.shape[0], 2)).cuda()
        pred_cnt = torch.zeros((coord.shape[0], 1)).cuda()
        idx_size = len(idx_data)
        idx_list, coord_list, offset_list = [], [], []
        for i in range(idx_size):
            print('{}/{}: {}/{}/{}, {}'.format(idx + 1, len(data_list), i + 1, idx_size, idx_data[0].shape[0], item))
            idx_part = idx_data[i]
            coord_part = coord[idx_part]
            
            coord_part = input_normalize(coord_part)
            idx_list.append(idx_part), coord_list.append(coord_part), offset_list.append(idx_part.size)
        batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
        for i in range(batch_num):
            s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
            idx_part, coord_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], offset_list[s_i:e_i]
            idx_part = np.concatenate(idx_part)
            coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
            offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
            with torch.no_grad():
                pred_part = model([coord_part, offset_part])  # (n, k)
            torch.cuda.empty_cache()
            pred[idx_part, :] += pred_part
            pred_cnt[idx_part, :] += 1
            print('Test: {}/{}, {}/{}, {}/{}'.format(idx + 1, len(data_list), e_i, len(idx_list), args.voxel_max, idx_part.shape[0]))
        # pred = pred / pred_cnt
        
        pred = pred.max(1)[1].data.cpu().numpy()
        pred = np.concatenate((coord, pred.reshape((-1, 1))), axis=1)
        pred_save.append(pred)
        np.save(pred_save_path, pred)

    with open(os.path.join(args.save_folder, "pred.pickle"), 'wb') as handle:
        pickle.dump({'pred': pred_save}, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    main()
