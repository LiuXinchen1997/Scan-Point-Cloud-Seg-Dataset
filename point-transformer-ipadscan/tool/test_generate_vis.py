from posixpath import basename
import numpy as np
import os
import open3d as o3d
import colorsys


COLOR_MAP = [
    [0xfa/0xff, 0x2a/0xff, 0x2a/0xff],
    [0x0c/0xff, 0xf9/0xff, 0xc8/0xff],
    [0xfc/0xff, 0x79/0xff, 0x08/0xff],
    [0xf3/0xff, 0xe4/0xff, 0x13/0xff],
    [0x9b/0xff, 0xf6/0xff, 0x10/0xff],
    [0x3d/0xff, 0xf6/0xff, 0x1e/0xff],
    [0x21/0xff, 0xf5/0xff, 0x66/0xff],
    [0x1e/0xff, 0xc1/0xff, 0xf9/0xff],
    [0x33/0xff, 0x6a/0xff, 0xf8/0xff],
    [0x4b/0xff, 0x24/0xff, 0xf4/0xff],
    [0xa5/0xff, 0x13/0xff, 0xf3/0xff],
    [0xf9/0xff, 0x34/0xff, 0xe2/0xff],
    [0xfc/0xff, 0x25/0xff, 0x7f/0xff],
]


def create_sphere_at_xyz(xyz, colors=None, radius=0.005):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=3)
    # sphere.compute_vertex_normals()
    if colors is None:
        sphere.paint_uniform_color([0xd2/0xff, 0x69/0xff, 0x1e/0xff])  # To be changed to the point color.
    else:
        sphere.paint_uniform_color(colors)
    sphere = sphere.translate(xyz)
    return sphere


def get_cross_prod_mat(pVec_Arr):
    # pVec_Arr shape (3)
    qCross_prod_mat = np.array([
        [0, -pVec_Arr[2], pVec_Arr[1]],
        [pVec_Arr[2], 0, -pVec_Arr[0]],
        [-pVec_Arr[1], pVec_Arr[0], 0],
    ])
    return qCross_prod_mat


def caculate_align_mat(pVec_Arr):
    scale = np.linalg.norm(pVec_Arr)
    pVec_Arr = pVec_Arr / scale
    # must ensure pVec_Arr is also a unit vec.
    z_unit_Arr = np.array([0, 0, 1])
    z_mat = get_cross_prod_mat(z_unit_Arr)
 
    z_c_vec = np.matmul(z_mat, pVec_Arr)
    z_c_vec_mat = get_cross_prod_mat(z_c_vec)
 
    if np.dot(z_unit_Arr, pVec_Arr) == -1:
        qTrans_Mat = -np.eye(3, 3)
    elif np.dot(z_unit_Arr, pVec_Arr) == 1:
        qTrans_Mat = np.eye(3, 3)
    else:
        qTrans_Mat = np.eye(3, 3) + z_c_vec_mat + np.matmul(z_c_vec_mat,
                                                            z_c_vec_mat) / (1 + np.dot(z_unit_Arr, pVec_Arr))
 
    # qTrans_Mat *= scale
    return qTrans_Mat


def create_line_from_to(xyz1, xyz2, colors=[0xd2/0xff, 0x69/0xff, 0x1e/0xff]):
    length = np.sqrt(np.sum(np.square(xyz1 - xyz2)))

    line = o3d.geometry.TriangleMesh.create_cylinder(radius=0.005, height=length)
    line.compute_vertex_normals()
    line.paint_uniform_color(colors)
    
    rot_vec = xyz2 - xyz1
    rot_mat = caculate_align_mat(rot_vec)
    line.rotate(rot_mat, center=np.array([0, 0, 0]))
    line = line.translate((xyz1 + xyz2) / 2.)
    return line


def generate_mesh_from_file_path(point_file_path, label_file_path, output_dir='./temp'):
    points = np.loadtxt(point_file_path)
    labels = np.load(label_file_path)
    basename = os.path.basename(point_file_path).split('.')[0]
    generate_mesh(points, labels, basename=basename, output_dir=output_dir)


def generate_mesh(points, labels, basename, bounding_box_vertexes=None, output_dir='./temp'):
    floor_labels = [0, 1, 2]
    floor_color = [0xDC/0xff, 0x14/0xff, 0x3C/0xff]  # 红色
    other_color = [0x00/0xff, 0x8B/0xff, 0x8B/0xff]  # 青色

    # mesh = create_sphere_at_xyz(points[0], other_color)
    mesh = create_sphere_at_xyz(points[0], COLOR_MAP[labels[0]])
    for i in range(points.shape[0]):
        # if labels[i] in floor_labels:
            # mesh += create_sphere_at_xyz(points[i], floor_color)
        # else:
            # mesh += create_sphere_at_xyz(points[i], other_color)
        mesh += create_sphere_at_xyz(points[i], COLOR_MAP[labels[i]])
        
    # create bounding box
    if bounding_box_vertexes is not None:
        # for i in range(bounding_box_vertexes.shape[0]):
            # mesh += create_sphere_at_xyz(bounding_box_vertexes[i], COLOR_MAP[8], radius=0.05)
        mesh += create_line_from_to(bounding_box_vertexes[0], bounding_box_vertexes[1], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[1], bounding_box_vertexes[2], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[2], bounding_box_vertexes[3], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[3], bounding_box_vertexes[0], COLOR_MAP[9])

        mesh += create_line_from_to(bounding_box_vertexes[4], bounding_box_vertexes[5], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[5], bounding_box_vertexes[6], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[6], bounding_box_vertexes[7], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[7], bounding_box_vertexes[4], COLOR_MAP[9])

        mesh += create_line_from_to(bounding_box_vertexes[0], bounding_box_vertexes[4], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[1], bounding_box_vertexes[5], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[2], bounding_box_vertexes[6], COLOR_MAP[9])
        mesh += create_line_from_to(bounding_box_vertexes[3], bounding_box_vertexes[7], COLOR_MAP[9])

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    output_path = os.path.join(output_dir, '%s.obj' % (basename))
    o3d.io.write_triangle_mesh(output_path, mesh)
    

# code for vis
if __name__ == '__main__':
    """
    names = [line.rstrip('\n') for line in open('data/s3dis/s3dis_names.txt')]
    print(names)

    input_dir = 'exp/s3dis/pointtransformer_repro_norgb/model/result/best'
    output_dir = os.path.join(input_dir, 'vis')
    epoch = 82
    for i in range(1, 6):
        print(i)
        point_file_path = 'dataset/ipad_scaned/point_cloud{}.xyz'.format(i)
        label_file_path = os.path.join(input_dir, 'point_cloud{}.xyz_{}_pred.npy'.format(i, epoch))

        generate_mesh(point_file_path, label_file_path, output_dir)
    """
    
    names = [line.rstrip('\n') for line in open('data/s3dis/s3dis_names.txt')]
    print(names)
    # input_dir = 'exp/s3dis/pointtransformer_repro_norgb_2cls_v3/model/result/last'
    # point_file_path = 'dataset/ipad_scaned/point_cloud5.xyz'
    # label_file_path = os.path.join(input_dir, 'point_cloud5.xyz_94_pred.npy')
    # points = points[points[:, 6] == 2]
    # points = np.loadtxt(point_file_path)
    # labels = np.load(label_file_path).astype(np.int32)
    # labels = labels.squeeze()
    
    for model_id in [3]:
        seg_label_file_path = '/home/liuxinchen/point-transformer-ipadscan/exp/ipad_scaned/baseline5/model/result/last/%d_199_pred.npy' % model_id
        # seg_label_file_path = '/data1/liuxinchen/point-transformer/exp/s3dis/pointtransformer_repro_norgb_2cls_v4/model/result/last/%d.xyz_100_pred.npy' % model_id
        gt_label_file_path = '/data1/liuxinchen/ipad_scaned/label_data/%d.npy' % model_id
        point_file_path = '/data1/liuxinchen/temp/%dr.xyz' % model_id
        points = np.loadtxt(point_file_path)[:, :3]
        seg_labels = np.load(seg_label_file_path).astype(np.int32)
        if len(seg_labels.shape) > 1:
            seg_labels = seg_labels[:, -1]

        for i in range(2):
            print(i, names[i], seg_labels[seg_labels == i].shape)
            if seg_labels[seg_labels == i].shape[0] == 0:
                continue
            # generate_mesh(points[labels == i], labels[labels == i], basename='label%d' % i, output_dir='./')
        
        basename = os.path.basename(point_file_path).split('.')[0]
        
        gt_labels = np.load(gt_label_file_path)
        if len(gt_labels.shape) > 1:
            gt_labels = gt_labels[:, -1]
        gt_points = points[gt_labels == 1]
        bounding_box_vertexes = []
        factor = 0.98
        x_max = np.max(gt_points[:, 0]) * factor
        y_max = np.max(gt_points[:, 1]) * factor
        z_max = np.max(gt_points[:, 2]) * factor
        x_min = np.min(gt_points[:, 0]) * factor
        y_min = np.min(gt_points[:, 1]) * factor
        z_min = np.min(gt_points[:, 2]) * factor

        bounding_box_vertexes.append([x_max, y_max, z_max])
        bounding_box_vertexes.append([x_max, y_min, z_max])
        bounding_box_vertexes.append([x_max, y_min, z_min])
        bounding_box_vertexes.append([x_max, y_max, z_min])
        bounding_box_vertexes.append([x_min, y_max, z_max])
        bounding_box_vertexes.append([x_min, y_min, z_max])
        bounding_box_vertexes.append([x_min, y_min, z_min])
        bounding_box_vertexes.append([x_min, y_max, z_min])
        bounding_box_vertexes = np.array(bounding_box_vertexes)
        
        generate_mesh(points, seg_labels, basename=basename, bounding_box_vertexes=bounding_box_vertexes, output_dir='./')
        