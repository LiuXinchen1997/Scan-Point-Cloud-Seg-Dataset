import os
import numpy as np

from util.common_util import AverageMeter, intersectionAndUnion, check_makedirs


def test(pred_dir, label_dir, files):
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    
    pred_save, label_save = [], []
    for idx, file in enumerate(files):
        pred_save_path = os.path.join(pred_dir, file)
        label_save_path = os.path.join(label_dir, file)
        pred, label = np.load(pred_save_path), np.load(label_save_path)
        if len(pred.shape) > 1:
            pred = pred[:, -1]
        if len(label.shape) > 1:
            label = label[:, -1]

        # calculation 1
        intersection, union, target = intersectionAndUnion(pred, label, K=2)
        intersection_meter.update(intersection)
        union_meter.update(union)
        target_meter.update(target)

        accuracy = sum(intersection) / (sum(target) + 1e-10)
        print(file, 'Accuracy {accuracy:.4f}.'.format(accuracy=accuracy))
        pred_save.append(pred); label_save.append(label)
        
    # calculation 1
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU1 = np.mean(iou_class)
    mAcc1 = np.mean(accuracy_class)
    allAcc1 = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # calculation 2
    intersection, union, target = intersectionAndUnion(np.concatenate(pred_save), np.concatenate(label_save), 2)
    iou_class = intersection / (union + 1e-10)
    accuracy_class = intersection / (target + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection) / (sum(target) + 1e-10)
    print('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
    print('Val1 result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU1, mAcc1, allAcc1))

    for i in range(2):
        print('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

if __name__ == '__main__':
    files = ['5.npy', '7.npy', '8.npy']
    pred_dir = '/data1/liuxinchen/point-transformer-ipadscan/temp'
    label_dir = '/data1/liuxinchen/ipad_scaned/label_data/test'
    test(pred_dir, label_dir, files)
