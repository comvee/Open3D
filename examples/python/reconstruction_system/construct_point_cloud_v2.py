import os
import cv2
import json
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import matplotlib.pyplot as plt
from glob import glob
import shutil
from distinctipy import distinctipy


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


def main():
    parser = argparse.ArgumentParser(description="Check the constructed point cloud")
    parser.add_argument("--input", help="path to the file directory, e.g. ./kinetic/data/filename", required=True)
    parser.add_argument("--viewer", help="skip (default) / simple / detail", default="skip", type=str)
    parser.add_argument("--skip_assertion", help="skip the assertions", action="store_true")
    args = parser.parse_args()

    base_path = args.input
    intrinsic_path = os.path.join(base_path, "intrinsic.json")
    scene_dir_path = os.path.join(base_path, "scene")
    annotation_dir_path = os.path.join(base_path, "annotation")
    img_selected_dir_path = os.path.join(base_path, "selected")
    rgb_paths = sorted(glob(f"{img_selected_dir_path}/*.jpg"))

    cnt = 0
    for rgb_path in rgb_paths:
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        cnt += "_" == filename[-1]
    if not args.skip_assertion:
        assert len(rgb_paths) == 5, f"Error: {img_selected_dir_path} should include 5 images."
        assert cnt == 1, f"Error: {img_selected_dir_path} should include one top-view image."
    
    if os.path.exists(scene_dir_path):
        shutil.rmtree(scene_dir_path)
    if os.path.exists(annotation_dir_path):
        shutil.rmtree(annotation_dir_path)
    os.makedirs(scene_dir_path)
    os.makedirs(annotation_dir_path)

    # Load intrinsic matrix
    with open(intrinsic_path, "r") as f:
        intrinsic_json = json.load(f)
        K = np.array(intrinsic_json["intrinsic_matrix"]).reshape(3,3).T
        K = np.concatenate([K, np.zeros((3,1))], axis=1)

    for rgb_path in rgb_paths:
        filename = os.path.splitext(os.path.basename(rgb_path))[0]
        depth_path = f"{base_path}/depth/{filename.split('_')[0]}.png"
        pcd_path = os.path.join(scene_dir_path, f"{filename}.ply")
        rgb = o3d.io.read_image(rgb_path)
        depth = o3d.io.read_image(depth_path)

        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
            width=intrinsic_json["width"],
            height=intrinsic_json["height"],
            fx=K[0,0],
            fy=K[1,1],
            cx=K[0,2],
            cy=K[1,2]
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb, depth)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd, intrinsic_matrix
                )
        pcd = pcd.voxel_down_sample(voxel_size=0.004)
        # Flip it, otherwise the pointcloud will be upside down
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # meter metric
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        indices = points[:,2] > -0.8
        points_filterd = points[indices]
        colors_filterd = colors[indices]

        pcd_filtered = o3d.geometry.PointCloud()
        pcd_filtered.points = o3d.utility.Vector3dVector(points_filterd)
        pcd_filtered.colors = o3d.utility.Vector3dVector(colors_filterd)
        o3d.io.write_point_cloud(pcd_path, pcd_filtered)
        print(f"{pcd_path} is saved.")

        if args.viewer == "simple":
            o3d.visualization.draw_geometries([pcd_filtered])
        elif args.viewer == "detail":
            app = gui.Application.instance
            app.initialize()
            vis = o3d.visualization.O3DVisualizer("Detailed viewer", 1680, 1024)
            vis.show_settings = True
            vis.add_geometry("Points", pcd_filtered)
            vis.reset_camera_to_default()
            app.add_window(vis)
            app.run()


if __name__ == "__main__":
    main()
