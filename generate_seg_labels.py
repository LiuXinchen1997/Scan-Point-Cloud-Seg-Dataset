import numpy as np
import os

from utils import points2set, set2points, generate_seg_obj_file
from config import seg_labels_cfg as cfg


"""
Input:  raw scan points `%d.xyz` (in `raw_data_dir`), 
        foreground points `%d_1.xyz` (in `label_part_data_dir`)
Output: background points `%d_0.xyz` (in `label_part_data_dir`), 
        seg labels points `%d.xyz` or `%d.npy` (in `label_data_dir`)
        seg label color files `%d.obj` (in `./`)
"""
if __name__ == '__main__':
    # config info
    model_ids               = cfg.MODEL_IDS
    xyz_delimiter           = cfg.XYZ_DELIMITER
    raw_data_dir            = cfg.RAW_DATA_DIR
    label_part_data_dir     = cfg.LABEL_PART_DATA_DIR

    shuffle                 = cfg.SHUFFLE
    label_data_dir          = cfg.LABEL_DATA_DIR

    obj_mode                = cfg.OBJ_MODE
    output_obj_dir          = cfg.OUTPUT_OBJ_DIR

    # Assert u have prepared raw scan points `%d.xyz` and extracted foreground points `%d_1.xyz`
    # Step 1. Extract background points from scan points
    # -> Output: background points `%d_0.xyz` (xyz, shape: N*3)
    for model_id in model_ids:
        print('[extract background points]: %03d' % model_id)
        points = np.loadtxt(os.path.join(raw_data_dir, '%d.xyz' % model_id), delimiter=xyz_delimiter)
        points1 = np.loadtxt(os.path.join(label_part_data_dir, '%d_1.xyz' % model_id), delimiter=xyz_delimiter)
        assert points1.shape[0] < points.shape[0]
        
        ps = points2set(points)
        ps1 = points2set(points1)
        ps0 = ps - ps1
        points0 = set2points(ps0)
        np.savetxt(os.path.join(label_part_data_dir, '%d_0.xyz' % model_id), points0, fmt='%.6f')
    
    # Step 2. Generate points with seg labels
    # -> Output: seg labels points `%d.xyz` or `%d.npy` (xyzl, shape: N*4)
    for model_id in model_ids:
        print('[generate label points]: %03d' % model_id)
        label0 = np.loadtxt(os.path.join(label_part_data_dir, '%d_0.xyz' % model_id))
        label1 = np.loadtxt(os.path.join(label_part_data_dir, '%d_1.xyz' % model_id))

        label0 = np.concatenate((label0, np.zeros((label0.shape[0], 1))), axis=1)
        label1 = np.concatenate((label1, np.ones((label1.shape[0], 1))), axis=1)

        data = np.concatenate((label0, label1), axis=0)
        if shuffle:
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data[:] = data[indices, ...]
        
        np.savetxt(os.path.join(label_data_dir, '%d.xyz' % model_id), data, fmt='%.6f')
        np.save(os.path.join(label_data_dir, '%d.npy' % model_id), data)
    
    # Step 3. Generate obj files to check correctness of seg labels (option)
    # Assert u have generated seg labels points `%d.xyz`
    # -> Output: seg label color files `%d.obj`
    if obj_mode is not None:
        for model_id in model_ids:
            print('[generate obj files]: %03d' % model_id)
            data = np.loadtxt(os.path.join(label_data_dir, '%d.xyz' % model_id))
            points = data[:, :3]
            labels = data[:, -1].astype(np.int32)

            basename = '%d' % model_id
            generate_seg_obj_file(points, labels, obj_mode=obj_mode, basename=basename, output_dir=output_obj_dir)
