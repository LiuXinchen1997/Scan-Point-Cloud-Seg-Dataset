import os
from easydict import EasyDict


seg_labels_cfg                                   = EasyDict()
seg_labels_cfg.ROOT_DIR                          = './'
seg_labels_cfg.MODEL_IDS                         = [id for id in range(1, 21)]                                                    # process models' ids
seg_labels_cfg.XYZ_DELIMITER                     = None                                                                           # xyz files' delimiter (comma, blank, ...)
seg_labels_cfg.RAW_DATA_DIR                      = os.path.join(seg_labels_cfg.ROOT_DIR, 'ipad_scaned')                           # raw scan points files dir
seg_labels_cfg.LABEL_PART_DATA_DIR               = os.path.join(seg_labels_cfg.ROOT_DIR, 'ipad_scaned/label_part_data')           # label part points files dir (<model_id>_0.xyz, <model_id>_1.xyz)

seg_labels_cfg.SHUFFLE                           = True                                                                           # generate seg labels files if shuffle points permutation
seg_labels_cfg.LABEL_DATA_DIR                    = os.path.join(seg_labels_cfg.ROOT_DIR, 'ipad_scaned/label_data')                # points files with seg labels

seg_labels_cfg.OBJ_MODE                          = 2                                                                              # generate obj files' type
seg_labels_cfg.OUTPUT_OBJ_DIR                    = './'                                                                           # generate obj files' dir


color_points_cfg                                 = EasyDict()
color_points_cfg.ROOT_DIR                        = './'
color_points_cfg.MODEL_IDS                       = [id for id in range(17, 21)]                                                   # process models' ids
color_points_cfg.RAW_SCAN_DATA_DIR               = os.path.join(color_points_cfg.ROOT_DIR, 'ipad_scaned_color/raw_scan_data')     # raw scan data dir (mesh and texture)
color_points_cfg.INPUT_LABEL_PART_DATA_DIR       = os.path.join(color_points_cfg.ROOT_DIR, 'ipad_scaned/label_part_data')         # label part points files dir (without rgb)
color_points_cfg.OUTPUT_LABEL_PART_DATA_DIR      = os.path.join(color_points_cfg.ROOT_DIR, 'ipad_scaned_color/label_part_data')   # label part points files dir (with rgb)

color_points_cfg.OUTPUT_LABEL_DATA_DIR           = os.path.join(color_points_cfg.ROOT_DIR, 'ipad_scaned_color/label_data')        # generate color label points files dir (xyzrgbl)
color_points_cfg.SHUFFLE                         = True                                                                           # generate seg labels files if shuffle points permutation
color_points_cfg.OBJ_MODE                        = 2                                                                              # generate obj files' type


train_test_split_cfg                             = EasyDict()
train_test_split_cfg.ROOT_DIR                    = './'
train_test_split_cfg.TRAIN_LIST_FILE             = os.path.join(train_test_split_cfg.ROOT_DIR, 'ipad_scaned/label_data/train_list.txt')
train_test_split_cfg.TEST_LIST_FILE              = os.path.join(train_test_split_cfg.ROOT_DIR, 'ipad_scaned/label_data/test_list.txt')
train_test_split_cfg.LABEL_DATA_DIR              = os.path.join(train_test_split_cfg.ROOT_DIR, 'ipad_scaned/label_data')
train_test_split_cfg.EXT                         = 'xyz'

train_test_split_cfg.OUTPUT_TRAIN_DATA_DIR       = os.path.join(train_test_split_cfg.ROOT_DIR, 'ipad_scaned/label_data/train')
train_test_split_cfg.OUTPUT_TEST_DATA_DIR        = os.path.join(train_test_split_cfg.ROOT_DIR, 'ipad_scaned/label_data/test')

train_test_split_cfg.GAUSS_NOISE_SIGMAS          = [0.001, 0.005, 0.01]
train_test_split_cfg.NUM_ROTATE_GROUP            = 5
