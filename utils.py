import numpy as np
import os
import open3d as o3d
import trimesh

from PIL import Image
from mesh_to_sdf import mesh_to_sdf


# points operation
def guass_noise_point_cloud(points, sigma=0.005, mu=0.00):
    """
    points: N*3
    """
    points += np.random.normal(mu, sigma, points.shape)
    return points

def rotate_point_cloud(points, angle_x, angle_y, angle_z):
    """
    points: N*3
    angle: [0, 2*pi)
    """
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(angle_x), -np.sin(angle_x)],
                    [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                    [0, 1, 0],
                    [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                    [np.sin(angle_z), np.cos(angle_z), 0],
                    [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))
    
    points = np.dot(points, rotation_matrix)
    return points


class Point(object):
    def __init__(self, x, y, z) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.x == __o.x and self.y == __o.y and self.z == __o.z
        else:
            return False
    
    def __hash__(self):
        return hash(self.x) + hash(self.y) + hash(self.z)

def points2set(points):
    # points numpy array -> points set
    ps = set()
    for i in range(points.shape[0]):
        ps.add(Point(points[i, 0], points[i, 1], points[i, 2]))
    
    return ps

def set2points(ps):
    # points set -> points numpy array
    points = []
    for p in ps:
        points.append([p.x, p.y, p.z])
    
    return np.array(points)


# generate seg label points (.obj file) for vis
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

def create_sphere_at_xyz(xyz, colors=None, radius=0.005, resolution=3):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius, resolution=resolution)
    sphere.compute_vertex_normals()
    if colors is None:
        sphere.paint_uniform_color([0x69/0xff, 0x69/0xff, 0x69/0xff])
    else:
        sphere.paint_uniform_color(colors)
    sphere = sphere.translate(xyz)
    return sphere

def generate_seg_obj_file(points, labels, basename, obj_mode=1, output_dir='./temp'):
    output_path = os.path.join(output_dir, '%s.obj' % (basename))

    if obj_mode == 1:
        mesh = create_sphere_at_xyz(points[0], COLOR_MAP[labels[0]])
        for i in range(points.shape[0]):
            mesh += create_sphere_at_xyz(points[i], COLOR_MAP[labels[i]])

        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        o3d.io.write_triangle_mesh(output_path, mesh)
    elif obj_mode == 2:
        fout = open(output_path, 'w')
        for i in range(points.shape[0]):
            colors = COLOR_MAP[labels[i]]
            colors = [int(c*255) for c in colors]
            fout.write('v %f %f %f %d %d %d\n' % (points[i, 0], points[i, 1], points[i, 2], colors[0], colors[1], colors[2]))
        fout.close()
    else:
      raise Exception('mode must be 1 or 2.')

def generate_color_points_obj(points, pos_colors, output_file_path, obj_mode=2):
        N = points.shape[0]
        if obj_mode == 1:
            mesh = create_sphere_at_xyz(points[0], pos_colors[0] / 0xff)
            for i in range(N):
                mesh += create_sphere_at_xyz(points[i], pos_colors[i] / 0xff)

            o3d.io.write_triangle_mesh(output_file_path, mesh)
        elif obj_mode == 2:
            fout = open(output_file_path, 'w')
            for i in range(N):
                c = pos_colors[i]
                fout.write('v %f %f %f %d %d %d\n' % (points[i,0], points[i,1], points[i,2], c[0], c[1], c[2]))
            fout.close()
        else:
            raise Exception('obj_mode must be 1 or 2.')


# process points with color
class ColorPointsGenerator:
    def __init__(self, basename, mesh_path, texture_path, points_path, color_points_output_dir, save_name=None, obj_mode=1) -> None:
        if get_file_ext(mesh_path) != 'obj':
            raise TypeError('mesh file must be obj file.')
        if obj_mode not in [1, 2, None]:
            raise Exception('obj_mode must be 1, 2 or None.')
        
        self.basename = basename
        self.mesh_path = mesh_path
        self.texture_path = texture_path
        self.points_path = points_path
        self.color_points_output_dir = color_points_output_dir
        self.save_name = save_name
        self.obj_mode = obj_mode
        if self.save_name is None:
            self.save_name = self.basename

        self.points, self.pos_colors = None, None
        
    @staticmethod
    def _read_textured_obj(mesh_path, texture_path):
        f = open(mesh_path)
        lines = f.readlines()
        vs = []
        vts = []
        fs = []
        for line in lines:
            eles = line.split(' ')
            if eles[0] == 'v':
                vs.append([float(eles[1]), float(eles[2]), float(eles[3])])
            elif eles[0] == 'vt':
                vts.append([float(eles[1]), float(eles[2])])
            elif eles[0] == 'f':
                fs.append([int(eles[1].split('/')[0]), int(eles[2].split('/')[0]),
                        int(eles[3].split('/')[0])])

        vs = np.array(vs)
        vts = np.array(vts)

        # load jpg
        colors = []
        if not os.path.exists(texture_path):
            raise Exception('there is no texture file!')
        image = Image.open(texture_path)
        # image = cv2.imread(texture_path)
        image = np.array(image)

        for vt in vts:
            xt = int(vt[0] * image.shape[0])
            yt = int((1 - vt[1]) * image.shape[1])
            colors.append(image[yt, xt, :])
        colors = np.asarray(colors)

        return vs, vts, colors

    @staticmethod
    def _reverse_ids(ids):
        # ids: sorted pos --> pre index
        # return: re_ids: pre pos --> sorted index
        re_ids = np.zeros_like(ids).astype(np.int)
        for i in range(ids.shape[0]):
            id = ids[i]
            re_ids[id] = i
        return re_ids
    
    @staticmethod
    def _mesh2voxel2sdf(input_mesh_path, input_texture_path, input_points_path):
        mesh = trimesh.load(input_mesh_path)
        vertices = np.array(mesh.vertices)
        points = np.loadtxt(input_points_path)
        
        sdf, face_ids = mesh_to_sdf(mesh, points, surface_point_method='sample', sign_method='normal')
        vertices = np.array(mesh.vertices)
        faces = np.array(mesh.faces)
        vs, _, colors = ColorPointsGenerator._read_textured_obj(input_mesh_path, input_texture_path)

        ids = np.argsort(vs, axis=0, kind='mergesort')[:, 2]
        vs[:] = vs[ids]
        colors[:] = colors[ids]
        ids = np.argsort(vs, axis=0, kind='mergesort')[:, 1]
        vs[:] = vs[ids]
        colors[:] = colors[ids]
        ids = np.argsort(vs, axis=0, kind='mergesort')[:, 0]
        vs[:] = vs[ids]
        colors[:] = colors[ids]

        ids = np.argsort(vertices, axis=0, kind='mergesort')[:, 2]
        vertices[:] = vertices[ids]
        faces = ColorPointsGenerator._reverse_ids(ids)[faces]
        ids = np.argsort(vertices, axis=0, kind='mergesort')[:, 1]
        vertices[:] = vertices[ids]
        faces = ColorPointsGenerator._reverse_ids(ids)[faces]
        ids = np.argsort(vertices, axis=0, kind='mergesort')[:, 0]
        vertices[:] = vertices[ids]
        faces = ColorPointsGenerator._reverse_ids(ids)[faces]

        pos_colors = None
        if False:
            colors[faces[face_ids]]  # N*3*3
            real_pos  # N*3
            first_ps = vertices[faces[face_ids][:, 0]]  # N*3
            second_ps = vertices[faces[face_ids][:, 1]]  # N*3
            third_ps = vertices[faces[face_ids][:, 2]]  # N*3
            fisrt_dist = np.sqrt(np.sum(np.power(real_pos - first_ps, 2), axis=-1))  # N*1
            second_dist = np.sqrt(np.sum(np.power(real_pos - second_ps, 2), axis=-1))  # N*1
            third_dist = np.sqrt(np.sum(np.power(real_pos - third_ps, 2), axis=-1))  # N*1
            first_weights = fisrt_dist / (fisrt_dist + second_dist + third_dist)  # N*1
            second_weights = second_dist / (fisrt_dist + second_dist + third_dist)  # N*1
            third_weights = third_dist / (fisrt_dist + second_dist + third_dist)  # N*1
            first_weights = first_weights[:, np.newaxis]
            second_weights = second_weights[:, np.newaxis]
            third_weights = third_weights[:, np.newaxis]

            pos_face_colors = colors[faces[face_ids]]  # N*3*3
            pos_colors = pos_face_colors[:, 0, :] * first_weights + pos_face_colors[:, 1, :] * second_weights \
            + pos_face_colors[:, 2, :] * third_weights
            pos_colors = pos_colors.astype(np.int)
        else:
            pos_colors = np.mean(colors[faces[face_ids]], axis=1).astype(np.int)

        return points, pos_colors

    def generate_color_points_file(self):
        if not os.path.exists(self.color_points_output_dir):
            os.mkdir(self.color_points_output_dir)
        
        if self.points is None or self.pos_colors is None:
            self.points, self.pos_colors = ColorPointsGenerator._mesh2voxel2sdf(self.mesh_path, self.texture_path, self.points_path)
        if self.obj_mode is not None:
            self.generate_color_points_obj()

        data = np.concatenate((self.points, self.pos_colors), axis=1)
        np.savetxt(os.path.join(self.color_points_output_dir, '%s.xyz' % self.save_name), data, fmt='%.6f')
        np.save(os.path.join(self.color_points_output_dir, '%s.npy' % self.save_name), data)

    def generate_color_points_obj(self):
        if self.points is None or self.pos_colors is None:
            self.points, self.pos_colors = ColorPointsGenerator._mesh2voxel2sdf(self.mesh_path, self.texture_path, self.points_path)
        
        output_file_path = os.path.join(self.color_points_output_dir, '%s.obj' % self.save_name)
        N = self.points.shape[0]
        if self.obj_mode == 1:
            mesh = create_sphere_at_xyz(self.points[0], self.pos_colors[0] / 0xff)
            for i in range(N):
                mesh += create_sphere_at_xyz(self.points[i], self.pos_colors[i] / 0xff)

            o3d.io.write_triangle_mesh(output_file_path, mesh)
        elif self.obj_mode == 2:
            fout = open(output_file_path, 'w')
            for i in range(N):
                c = self.pos_colors[i]
                fout.write('v %f %f %f %d %d %d\n' % (self.points[i,0], self.points[i,1], self.points[i,2], c[0], c[1], c[2]))
            fout.close()
        else:
            raise Exception('mode must be 1 or 2.')


# others
def get_file_ext(file_path):
    return file_path[file_path.rfind('.')+1:]

def load_file(file_path):
    ext = get_file_ext(file_path)
    if ext == 'xyz':
        tensor = np.loadtxt(file_path)
    elif ext == 'npy':
        tensor = np.load(file_path)
    else:
        raise TypeError('file extension must be xyz or npy.')
    
    return tensor

def save_file(output_dir, basename, tensor):
    ext = get_file_ext(basename)
    save_file_path = os.path.join(output_dir, basename)
    if ext == 'xyz':
        np.savetxt(save_file_path, tensor, fmt='%.6f')
    elif ext == 'npy':
        np.save(save_file_path, tensor)
    else:
        raise TypeError('file extension must be xyz or npy.')

def delimiter_comma2blank_for_xyz(xyz_files):
    for xyz_file in xyz_files:
        if get_file_ext(xyz_file) != 'xyz':
            raise TypeError('points file must be xyz file.')
        lines = open(xyz_file, 'r').readlines()

        with open(xyz_file, 'w') as output_file:
            for line in lines:
                line = line.replace(',', ' ')
                output_file.write(line)


if __name__ == '__main__':
    from glob import glob
    files = glob('./*.xyz')
    delimiter_comma2blank_for_xyz(files)
