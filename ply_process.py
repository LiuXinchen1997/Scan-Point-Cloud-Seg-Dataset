import numpy as np
import os

from time import time
from plyfile import PlyData

"""
TODO: 
1. get .ply file with texture by ipad scanning.
2. [following code] extract foreground point sets with MeshLab operation (save as .ply, with color).
3. [following code] get background point sets by subtracting with raw points and foreground points.
4. [following code] save foreground points and background points as format xyzrgb.
"""

class Point(object):
    def __init__(self, x, y, z, r, g, b) -> None:
        super().__init__()
        self.x = x
        self.y = y
        self.z = z
        self.r = r
        self.g = g
        self.b = b
    
    def __eq__(self, __o: object) -> bool:
        if isinstance(__o, self.__class__):
            return self.x == __o.x and self.y == __o.y and self.z == __o.z and self.r == __o.r and self.g == __o.g and self.b == __o.b
        else:
            return False
    
    def __hash__(self):
        return hash(self.x) + hash(self.y) + hash(self.z) + hash(self.r) + hash(self.g) + hash(self.b)

def points2set(points):
    # points numpy array -> points set
    ps = set()
    for i in range(points.shape[0]):
        ps.add(Point(points[i, 0], points[i, 1], points[i, 2], points[i, 3], points[i, 4], points[i, 5]))
    
    return ps

def set2points(ps):
    # points set -> points numpy array
    points = []
    for p in ps:
        points.append([p.x, p.y, p.z, p.r, p.g, p.b])
    
    return np.array(points)


if __name__ == '__main__':
    start_time = time()
    precision = 4

    ply_points = PlyData.read(os.path.join('./ipad_scaned_color/raw_scan_data/1', 'point_cloud.ply'))
    print(ply_points['vertex'].data)
    points = np.array([[np.round(point[0], precision), np.round(point[1], precision), np.round(point[2], precision), point[3], point[4], point[5]] for point in ply_points['vertex'].data])

    ply_points0 = PlyData.read('./test_point_cloud0.ply')
    print(ply_points0['vertex'].data)
    points0 = np.array([[np.round(point[0], precision), np.round(point[1], precision), np.round(point[2], precision), point[3], point[4], point[5]] for point in ply_points0['vertex'].data])
    print(points.shape)
    print(points0.shape)

    ps = points2set(points)
    ps0 = points2set(points0)
    ps1 = ps - ps0
    points1 = set2points(ps1)
    print(points1.shape)

    end_time = time()
    print('cost time:', end_time - start_time)
