import open3d as o3d
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
import json
import trimesh

# Removing any `.cuda()` or similar calls, and adjusting tensor operations to run on CPU

def np_to_pcd(arr, color=[1,0,0]):
    arr = np.array(arr)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    if arr.shape[1] >= 6:
        pcd.normals = o3d.utility.Vector3dVector(arr[:,3:6])
    pcd.colors = o3d.utility.Vector3dVector([color]*len(pcd.points))
    return pcd

def np_to_pcd_with_prob(arr, axis=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])

    base_color = np.array([1,0,0])
    base_color_2 = np.array([0,1,0])
    label_colors = np.zeros((arr.shape[0], 3))
    for idx in range(label_colors.shape[0]):
        label_colors[idx] = arr[idx, 3] * (base_color) + (1-arr[idx, 3]) * base_color_2
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def save_pcd(path, arr):
    o3d.io.write_point_cloud(path, arr)

def save_mesh(path, mesh):
    o3d.io.write_triangle_mesh(path, mesh)

def count_unique_by_row(a):
    weight = 1j*np.linspace(0, a.shape[1], a.shape[0], endpoint=False)
    b = a + weight[:, np.newaxis]
    u, ind, cnt = np.unique(b, return_index=True, return_counts=True)
    b = np.zeros_like(a)
    np.put(b, ind, cnt)
    return b

def load_colored_mesh(org_mesh_path, ipr_mesh_path, stl_path_list, up_down_idx, matching_dist_thr):
    global_mesh = load_mesh(org_mesh_path[up_down_idx])
    global_mesh = global_mesh.remove_duplicated_vertices()
    global_mesh_arr = np.asarray(global_mesh.vertices)

    cluster_vertex_colors = np.ones((np.asarray(global_mesh.vertices).shape))

    tree = KDTree(global_mesh_arr, leaf_size=2)

    cluster_vertex_colors = np.zeros((np.asarray(global_mesh.vertices).shape))
    cluster_vertex_colors[:,0] = 1

    ipr_mesh = load_mesh(ipr_mesh_path[up_down_idx])
    ipr_mesh = ipr_mesh.remove_duplicated_vertices()
    ipr_mesh_arr = np.asarray(ipr_mesh.vertices)

    dists, indexs = tree.query(ipr_mesh_arr, k=10)

    for point_num, (corr_idx_ls, dist_ls) in enumerate(zip(indexs,dists)):
        not_matching_flag=True

        for idx_item, dist_item in zip(corr_idx_ls,dist_ls):
            if dist_item<0.0001:
                cluster_vertex_colors[idx_item,:] = np.asarray([0,1,1])
                not_matching_flag = False
                
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(cluster_vertex_colors)
    total_ls = []
    for idx in range(len(stl_path_list)):
        stl_path = stl_path_list[idx]

        tooth_num = get_number_from_name(stl_path)
        if up_down_idx==1:
            if(tooth_num>=30):
                continue
        else:
            if(tooth_num<30):
                continue

        mesh = load_mesh(stl_path)
        mesh_arr = np.asarray(mesh.vertices)
        dists, indexs = tree.query(mesh_arr, k=4)
        tooth_num_color = np.random.rand(3)
        tooth_num_color[0] = tooth_num/50
        for point_num, (corr_idx_ls, dist_ls) in enumerate(zip(indexs,dists)):
            not_matching_flag=True
            for idx_item, dist_item in zip(corr_idx_ls,dist_ls):
                if dist_item < 0.0001:
                    cluster_vertex_colors[idx_item,:] = np.asarray(tooth_num_color)
                    not_matching_flag = False
                    
            if not_matching_flag:
                for idx_item, dist_item in zip(corr_idx_ls,dist_ls):
                    if dist_item< matching_dist_thr and (cluster_vertex_colors[idx_item,:] == np.array([1,0,0])).all():
                        cluster_vertex_colors[idx_item,:] = np.asarray(tooth_num_color)

    for i in range(cluster_vertex_colors.shape[0]):
        if (cluster_vertex_colors[i,:] == np.array([1,0,0])).all():
            cluster_vertex_colors[i,:] = np.array([0,1,1])
    global_mesh.vertex_colors = o3d.utility.Vector3dVector(cluster_vertex_colors)
    return global_mesh


def recomp_normals(arr):
    search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,max_nn=8)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    pcd.normals = o3d.utility.Vector3dVector(arr[:,3:])
    pcd.estimate_normals(search_param)
    return np.array(pcd.normals)

def load_mesh(mesh_path, only_tooth_crop = False):
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.remove_duplicated_vertices()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    if only_tooth_crop:
        cluster_idxes, cluster_nums, _ = mesh.cluster_connected_triangles()
        cluster_idxes = np.asarray(cluster_idxes)
        cluster_nums = np.asarray(cluster_nums)
        tooth_cluster_num = np.argmax(cluster_nums)
        mesh.remove_triangles_by_mask(cluster_idxes!=tooth_cluster_num)
    return mesh

def colorling_mesh_with_label(mesh, label_arr, colorling):
    label_arr = label_arr.reshape(-1)
    if colorling=="sem":
        palte = np.array([
            [255,153,153],

            [153,76,0],
            [153,153,0],
            [76,153,0],
            [0,153,153],
            [0,0,153],
            [153,0,153],
            [153,0,76],
            [64,64,64],

            [255,128,0],
            [153,153,0],
            [76,153,0],
            [0,153,153],
            [0,0,153],
            [153,0,153],
            [153,0,76],
            [64,64,64],
        ])/255
    else:
        palte = np.random.rand(200,3)
        palte[0,:] = np.array([255,153,153]) 
    palte[9:] *= 0.4

    verts_arr = np.array(mesh.vertices)
    label_colors = np.zeros((verts_arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[label_arr==idx] = palte[idx]
    mesh.vertex_colors = o3d.utility.Vector3dVector(label_colors)

def np_to_pcd_with_label(arr, label_arr=None, axis=3):
    if type(label_arr) == np.ndarray:
        arr = np.concatenate([arr[:,:3], label_arr.reshape(-1,1)],axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([
        [255,153,153],

        [153,76,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],

        [255,128,0],
        [153,153,0],
        [76,153,0],
        [0,153,153],
        [0,0,153],
        [153,0,153],
        [153,0,76],
        [64,64,64],
    ])/255
    palte[9:] *= 0.4
    arr = arr.copy()
    arr[:,axis] %= palte.shape[0]
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def np_to_pcd_removed(arr):
    points = arr[:,:3]
    labels = arr[:,3]
    points = points[labels==1, :]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def remove_0_points(arr, target=0):
    points = arr[:,:3]
    labels = arr[:,3]
    points = points[labels==(1-target), :]
    return points

def sigmoid(x):
    return 1 / (1 +np.exp(-x))

def get_number_from_name(path):
    return int(os.path.basename(path).split("_")[-1].split(".")[0])

def get_up_from_name(path):
    return os.path.basename(path).split("_")[-1].split(".")[0]=="up"

def np_to_by_label(arr, axis=3):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr[:,:3])
    
    palte = np.array([[1,0,0],[0,1,0],[0,0,1],[1,1,0],[1,0,1],[0,1,1],[1/2,0,0],[0,1/2,0],[0,0,1/2],
       [0.32736046, 0.71189952, 0.20750141],
       [0.56743345, 0.07504726, 0.34684285],
       [0.35949841, 0.4314623 , 0.98791015],
       [0.31151589, 0.44971993, 0.86484811],
       [0.96808667, 0.42096273, 0.95791817],
       [0.64136201, 0.41471365, 0.11085843],
       [0.4789342 , 0.30820783, 0.34425576],
       [0.50173988, 0.38319907, 0.09296238]])
    
    label_colors = np.zeros((arr.shape[0], 3))
    for idx, palte_color in enumerate(palte):
        label_colors[arr[:,axis]==idx] = palte[idx]
    pcd.colors = o3d.utility.Vector3dVector(label_colors)
    return pcd

def resample_pcd(pcd_ls, n, method):
    """Drop or duplicate points so that pcd has exactly n points"""
    if method=="uniformly":
        idx = np.random.permutation(pcd_ls[0].shape[0])
    elif method == "fps":
        idx = new_fps(pcd_ls[0][:,:3], n)
    pcd_resampled_ls = []
    for i in range(len(pcd_ls)):
        pcd_resampled_ls.append(pcd_ls[i][idx[:n]])
    return pcd_resampled_ls

def new_fps(xyz, npoint):
    if xyz.shape[0]<=npoint:
        raise "new fps error"
    idx = np.random.permutation(xyz.shape[0])
    return idx[:npoint]

def torch_to_numpy(cuda_arr):
    return cuda_arr.detach().numpy()

# Further conversion needed based on the subsequent code. Remove any torch.cuda or other GPU-specific calls, use CPU computations instead.
