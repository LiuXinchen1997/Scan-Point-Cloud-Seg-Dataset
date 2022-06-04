import numpy as np
import os

from utils import load_file, save_file, guass_noise_point_cloud, rotate_point_cloud
from config import train_test_split_cfg as cfg


if __name__ == '__main__':
    # config info
    train_list_file           = cfg.TRAIN_LIST_FILE
    test_list_file            = cfg.TEST_LIST_FILE
    label_data_dir            = cfg.LABEL_DATA_DIR
    ext                       = cfg.EXT

    output_train_data_dir     = cfg.OUTPUT_TRAIN_DATA_DIR
    output_test_data_dir      = cfg.OUTPUT_TEST_DATA_DIR

    gauss_noise_sigmas        = cfg.GAUSS_NOISE_SIGMAS
    num_rotate_group          = cfg.NUM_ROTATE_GROUP

    rotate_angles_groups = []
    for _ in range(num_rotate_group):
        angles = np.random.uniform(size=(3)) * 2 * np.pi
        rotate_angles_groups.append(angles)

    train_list = open(train_list_file, 'r').readlines()
    train_list = [os.path.join(label_data_dir, '%s.%s' % (basename.strip(), ext)) for basename in train_list]
    test_list = open(test_list_file, 'r').readlines()
    test_list = [os.path.join(label_data_dir, '%s.%s' % (basename.strip(), ext)) for basename in test_list]

    print('[train data list]:', train_list)
    print('[test data list]:', test_list)

    if not os.path.exists(output_train_data_dir):
        os.mkdir(output_train_data_dir)
    if not os.path.exists(output_test_data_dir):
        os.mkdir(output_test_data_dir)
    
    # extend and generate train data in `output_train_data_dir`
    for file in train_list:
        basename = os.path.basename(file)
        print('[extend and generate train data in %s]: %s' % (output_train_data_dir, basename))
        basename = basename[:basename.rfind('.')]
        points = load_file(file)
        save_file(output_train_data_dir, '%s.%s' % (basename, ext), points)
        
        xyz = points[:, :3]
        for i, sigma in enumerate(gauss_noise_sigmas):
            gauss_xyz = guass_noise_point_cloud(xyz.copy(), sigma=sigma)
            gauss_points = np.concatenate((gauss_xyz, points[:, 3:]), axis=1)
            save_file(output_train_data_dir, '%s_g%d.%s' % (basename, i, ext), gauss_points)
        
        for i, angles in enumerate(rotate_angles_groups):
            rotate_xyz = rotate_point_cloud(xyz.copy(), angles[0], angles[1], angles[2])
            rotate_points = np.concatenate((rotate_xyz, points[:, 3:]), axis=1)
            save_file(output_train_data_dir, '%s_r%d.%s' % (basename, i, ext), rotate_points)

    # generate test data in `output_test_data_dir`
    for file in test_list:
        basename = os.path.basename(file)
        print('[generate test data in %s]: %s' % (output_test_data_dir, basename))
        basename = basename[:basename.rfind('.')]
        points = load_file(file)
        save_file(output_test_data_dir, '%s.%s' % (basename, ext), points)
