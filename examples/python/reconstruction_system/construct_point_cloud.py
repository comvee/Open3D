import os
import cv2
import json
import argparse
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from glob import glob
import shutil
from distinctipy import distinctipy
from sklearn.cluster import KMeans


def visualize_point(point_targets, pcd, volume_len=0.03, colors=[(1,0,0)]):
    # point_targets: (N, 3)
    pcd_additionals = []
    if colors is None:
        N = point_targets.shape[0]
        colors = [(1, 0.8*(i+1)/N, 0.8*(i+1)/N) for i in range(N)]
    for i, point_target in enumerate(point_targets):
        if len(colors) == 1:
            color = colors[0]
        else:
            color = colors[i]
        # print(point_target, color)
        point_target = point_target.reshape((1,3))
        volume = np.random.uniform(-volume_len, volume_len, (100, 3)) + point_target
        pcd_volume = o3d.geometry.PointCloud()
        pcd_volume.points = o3d.utility.Vector3dVector(volume)
        pcd_volume.paint_uniform_color(color)
        pcd_additionals.append(pcd_volume)
    o3d.visualization.draw_geometries([pcd, *pcd_additionals])


def visualize_cluster(point_targets, labels, pcd, centroids=None):
    # point_targets: (N, 3)
    # labels: (N,)
    # cluster_num = len(np.unique(labels))
    cluster_num = labels.max() + 1
    ref_colors = distinctipy.get_colors(cluster_num)
    colors = [ref_colors[label] for label in labels]

    if centroids is not None:
        colors_centroids = [(0, 0, 0)] * centroids.shape[0]
        colors += colors_centroids
        point_targets = np.concatenate([point_targets, centroids])

    visualize_point(point_targets, pcd, volume_len=0, colors=colors)


def camera2world(x_camera, Rt):
    # x_camera: (3, 1)
    # Rt: (4, 4) world to camera matrix
    x_camera = x_camera.reshape((3,1))
    R = Rt[:3,:3]
    t = Rt[:3, 3].reshape((3,1))

    x_world = R.T @ (x_camera - t)
    return x_world


def world2pixel(x_world, K, Rt, k3=None):
    # x_world: (N, 3)
    # K: (3, 4)
    # Rt: (4, 4)
    assert x_world.shape[1] == 3
    x_world = x_world.T
    x_world = np.concatenate([x_world, np.ones((1,x_world.shape[1]))], axis=0)

    x_pixel = K @ Rt @ x_world
    x_pixel /= x_pixel[2,:]
    x_pixel = x_pixel[:2]
    return x_pixel


def draw_box(img, arranged_points, draw_index=False):
    RADIUS = 10
    COLOR = (0, 255, 0) # (255, 255, 255)
    EDGES = [
    #   [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
    #   [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
    #   [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis

      [1, 2], [4, 3], [5, 6], [8, 7],
      [1, 5], [4, 8], [2, 6], [3, 7],
      [1, 4], [2, 3], [6, 7], [5, 8] 
    ] 
    for i in range(arranged_points.shape[0]):
        x, y = arranged_points[i]
        p = (int(x), int(y))
        cv2.circle(img, p, RADIUS, COLOR, -10)
        if draw_index:
            cv2.putText(img, f"{i}", p, cv2.FONT_HERSHEY_SIMPLEX, 
            1, (0, 0, 255), 4, cv2.LINE_AA)
    for edge in EDGES:
        start_points = arranged_points[edge[0]-1]
        start_x = int(start_points[0])
        start_y = int(start_points[1])
        end_points = arranged_points[edge[1]-1]
        end_x = int(end_points[0])
        end_y = int(end_points[1])
        cv2.line(img, (start_x, start_y), (end_x, end_y), COLOR, 2)

    plt.imshow(img)
    cv2.imwrite("/tmp/a.jpg", img)
    return img


def k_means(points, n_clusters=6, n_init=12, print_out=False):
    k_means = KMeans(init="k-means++", n_clusters=n_clusters, n_init=n_init)
    k_means.fit(points)
    labels = k_means.labels_
    centroids = k_means.cluster_centers_

    if print_out:
        print('k_means_labels : ', labels)
        print('k_means_cluster_centers : ', centroids)
    return labels, centroids


def find_right_left_index(ls, target):
    lindex = ls.index(target)
    rindex = len(ls) - ls[::-1].index(target) - 1
    return lindex, rindex


def get_ordered_unqiue(ls, ignore=-1):
    out = []
    for curr in ls:
        if curr != ignore and curr not in out:
            out += [curr]
    return out


def cluster_trajectory(points, n_clusters=6, n_init=12):
    labels, centroids = k_means(points, n_clusters=n_clusters, n_init=n_init)
    first_label, last_label = labels[0], labels[-1]
    label_ls = labels.tolist()
    labels_fine = np.zeros_like(labels)
    n_cluster_sub_prev_sum = 0
    for label in range(n_clusters):
        lindex, rindex = find_right_left_index(label_ls, label)
        points_sub = points[lindex:rindex+1]
        labels_sub = []

        if label in [first_label, last_label]:
            n_clusters_sub = 2
        else:
            n_clusters_sub = 3
        labels_sub_tmp, centroids_sub = k_means(points_sub, n_clusters=n_clusters_sub, n_init=n_init)
        for label_sub_tmp in labels_sub_tmp:
            labels_sub += [label_sub_tmp + n_cluster_sub_prev_sum]
        labels_fine[lindex:rindex+1] = labels_sub 

        n_cluster_sub_prev_sum += n_clusters_sub
    return labels_fine


def cluster_filtering(points, labels, discard_num=2):
    N = points.shape[0]
    labels_filtered = []
    labels_unique = []
    for label in labels:
        if label not in labels_unique:
            labels_unique += [label]
    labels_target = labels_unique[::3]

    for point, label in zip(points, labels):
        if label not in labels_target:
            labels_filtered += [-1]
            continue

        labels_filtered += [label]
    
    labels_unique = [-1] + get_ordered_unqiue(labels_filtered, ignore=-1)
    label_remap = {label: i for i, label in enumerate(labels_unique)}
    labels_filtered = [label_remap[label] for label in labels_filtered]
    labels_filtered = np.array(labels_filtered)
    # Labels of filtered points become 0.

    for i in range(1, len(labels_unique)):
        lindex, rindex = find_right_left_index(labels_filtered.tolist(), i)
        labels_filtered[lindex:min(lindex+discard_num, N)] = 0
        labels_filtered[max(rindex-(discard_num-1), 0):rindex+1] = 0

    labels_filtered = np.array(labels_filtered)
    return points, labels_filtered


def detect_blur_fft(image, size=60, thresh=10):
	(h, w) = image.shape
	(cX, cY) = (int(w / 2.0), int(h / 2.0))
	fft = np.fft.fft2(image)
	fftShift = np.fft.fftshift(fft)

	fftShift[cY - size:cY + size, cX - size:cX + size] = 0
	fftShift = np.fft.ifftshift(fftShift)
	recon = np.fft.ifft2(fftShift)

	magnitude = 20 * np.log(np.abs(recon))
	mean = np.mean(magnitude)
	return (mean, mean <= thresh)


def main():
    parser = argparse.ArgumentParser(description="Check the constructed point cloud")
    parser.add_argument("--config",
                        help="path to the config file, e.g. ./kinetic/data/filename/config.json",
                        required=True)
    args = parser.parse_args()

    base_path = os.path.dirname(args.config)
    pcd_path = os.path.join(base_path, "scene/integrated.ply")
    trajectory_path = os.path.join(base_path, "scene/trajectory.log")
    intrinsic_path = os.path.join(base_path, "intrinsic.json")
    img_paths = sorted(glob(f"{base_path}/color/*.jpg"))
    img_view_dir_path = os.path.join(base_path, "view")
    img_selected_dir_path = os.path.join(base_path, "selected")
    fragments_dir_path = os.path.join(base_path, "fragments")

    if not os.path.exists(pcd_path):
        print(f"ERROR: cannout find the point cloud file at {pcd_path}")
        exit()
    
    if os.path.exists(img_view_dir_path):
        shutil.rmtree(img_view_dir_path)
    if os.path.exists(img_selected_dir_path):
        shutil.rmtree(img_selected_dir_path)
    if os.path.exists(fragments_dir_path):
        shutil.rmtree(fragments_dir_path)
    os.makedirs(img_view_dir_path)
    os.makedirs(img_selected_dir_path)

    # Load point cloud
    print("Load the files...")
    pcd = o3d.io.read_point_cloud(pcd_path)

    # Load intrinsic matrix
    with open(intrinsic_path, "r") as f:
        intrinsic_json = json.load(f)
        K = np.array(intrinsic_json["intrinsic_matrix"]).reshape(3,3).T
        K = np.concatenate([K, np.zeros((3,1))], axis=1)

    # Load extrinsic matrices
    with open(trajectory_path, "r") as f:
        lines = [line.rstrip() for line in f.readlines()]
    Rt_list = []
    for i in range(len(lines)//5):
        lines_sub = lines[i*5+1:i*5+5]
        M_c2w = np.array([list(map(float, line.split())) for line in lines_sub])
        R_c2w = M_c2w[:3,:3]
        t_c2w = M_c2w[:3,3]
        
        R_w2c = R_c2w.T
        t_w2c = -R_c2w.T @ t_c2w
        Rt = np.zeros((4,4))
        Rt[:3,:3] = R_w2c
        Rt[:3,3] = t_w2c
        Rt[3,3] = 1
        Rt_list += [Rt]

    # Compute camera trajectory
    print("Compute the camera trajectory...")
    Cs = []
    for i, Rt in enumerate(Rt_list):
        C = camera2world(np.zeros((3,1)), Rt)
        Cs += [C]
        ratio = (i+1)/len(Rt_list)
    Cs = np.stack(Cs).squeeze()

    # Cluster and filter the trajectory
    print("Cluster and filter the trajectory...")
    labels = cluster_trajectory(Cs, n_clusters=5, n_init=12)
    # visualize_cluster(Cs, labels, pcd, centroids=None)
    Cs_filtered, labels_filtered = cluster_filtering(Cs, labels, discard_num=0)
    Cs_filtered_only = []
    labels_filtered_only = []
    for C, label in zip(Cs_filtered, labels_filtered):
        if label == 0:
            continue
        Cs_filtered_only += [C]
        labels_filtered_only += [label]
    Cs_filtered_only = np.array(Cs_filtered_only)
    labels_filtered_only = np.array(labels_filtered_only)

    # Compute blurry score
    label2data = dict()
    for img_path, label in zip(img_paths, labels_filtered):
        if label == 0:
            continue
        if label not in label2data:
            label2data[label] = []
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score, blurry = detect_blur_fft(gray, size=30, thresh=20)
        label2data[label] += [[img_path, score]]
    
    if len(label2data) != 5:
        print("Error: 5개 미만 또는 5개 초과의 시점이 식별되어 재촬영이 필요합니다. "
              "영상 촬영시 5개의 시점(1초간 정지)만 포함해야 합니다. "
              "손이 흔들리지 않도록 주의하세요. ")
        visualize_cluster(Cs_filtered, labels_filtered, pcd, centroids=None)
        exit()

    for label in label2data:
        target_dir_path = os.path.join(img_view_dir_path, f"view_{label}")
        os.makedirs(target_dir_path)
        # discard_num = int(len(label2data[label])*0.3)
        # if discard_num == 0: 
        #     discard_num = None
        # data_discarded = sorted(label2data[label], reverse=True, key=lambda x: x[1])[:-discard_num]
        for i, data in enumerate(label2data[label]):
            img_path, score = data
            shutil.copy2(img_path, target_dir_path)
    
    print("Finish.")
    visualize_cluster(Cs_filtered, labels_filtered, pcd, centroids=None)

if __name__ == "__main__":
    main()
