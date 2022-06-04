import numpy as np
import os

from utils import generate_color_points_obj, ColorPointsGenerator
from config import color_points_cfg as cfg


"""
Input:  raw scan mesh `textured_output.obj` (in `raw_scan_data_dir`), 
        label part points `%d_0.xyz` and `%d_1.xyz` (xyz, shape: N*3, in `input_label_part_data_dir`)
Output: label part points with colors `%d_0.xyz` and `%d_1.xyz` (xyzrgb, shape: N*6, in `output_label_part_data_dir`)
        seg labels points with colors `%d.xyz` or `%d.npy` (xyzrgbl, shape: N*7, in `output_label_data_dir`)
        obj files `%d.obj` (in `output_label_data_dir`)
"""
if __name__ == '__main__':
    # config info
    model_ids                      = cfg.MODEL_IDS
    raw_scan_data_dir              = cfg.RAW_SCAN_DATA_DIR
    input_label_part_data_dir      = cfg.INPUT_LABEL_PART_DATA_DIR
    output_label_part_data_dir     = cfg.OUTPUT_LABEL_PART_DATA_DIR
    
    output_label_data_dir          = cfg.OUTPUT_LABEL_DATA_DIR
    shuffle                        = cfg.SHUFFLE 
    obj_mode                       = cfg.OBJ_MODE
    
    # Assert u have prepared raw scan mesh `textured_output.obj` and label part points `%d_0.xyz` and `%d_1.xyz` (xyz, shape: N*3)
    # Step 1. Generate label part points files with colors (xyzrgb), attch rgb info to label part points
    # -> Output: label part points with colors `%d_0.xyz` and `%d_1.xyz` (xyzrgb, shape: N*6)
    for model_id in model_ids:
        print('[generate part points file with color]: %03d' % model_id)
        cpg0 = ColorPointsGenerator(basename=model_id, mesh_path='%s/%d/textured_output.obj' % (raw_scan_data_dir, model_id), 
                                    texture_path='%s/%d/textured_output.jpg' % (raw_scan_data_dir, model_id), 
                                    points_path='%s/%d_0.xyz' % (input_label_part_data_dir, model_id), 
                                    color_points_output_dir=output_label_part_data_dir, save_name='%d_0' % model_id, obj_mode=obj_mode)
        cpg0.generate_color_points_file()
        cpg1 = ColorPointsGenerator(basename=model_id, mesh_path='%s/%d/textured_output.obj' % (raw_scan_data_dir, model_id), 
                                    texture_path='%s/%d/textured_output.jpg' % (raw_scan_data_dir, model_id), 
                                    points_path='%s/%d_1.xyz' % (input_label_part_data_dir, model_id), 
                                    color_points_output_dir=output_label_part_data_dir, save_name='%d_1' % model_id, obj_mode=obj_mode)
        cpg1.generate_color_points_file()

    # Step 2. Generate label points files with rgb (xyzrgbl)
    # -> Output: seg labels points with colors `%d.xyz` or `%d.npy` (xyzrgbl, shape: N*7), obj files to check correctness `%d.obj`
    for model_id in model_ids:
        print('[generate label points file with color]: %03d' % model_id)
        label0 = np.loadtxt(os.path.join(output_label_part_data_dir, '%d_0.xyz' % model_id))
        label1 = np.loadtxt(os.path.join(output_label_part_data_dir, '%d_1.xyz' % model_id))

        label0 = np.concatenate((label0, np.zeros((label0.shape[0], 1))), axis=1)
        label1 = np.concatenate((label1, np.ones((label1.shape[0], 1))), axis=1)

        data = np.concatenate((label0, label1), axis=0)
        if shuffle:
            indices = np.arange(data.shape[0])
            np.random.shuffle(indices)
            data[:] = data[indices, ...]
        
        np.savetxt(os.path.join(output_label_data_dir, '%d.xyz' % model_id), data, fmt='%.6f')
        np.save(os.path.join(output_label_data_dir, '%d.npy' % model_id), data)

        generate_color_points_obj(data[:, :3], data[:, 3:6], os.path.join(output_label_data_dir, '%d.obj' % model_id), obj_mode=obj_mode)
