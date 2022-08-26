import os
import cv2
import sys
import json
import shutil
import argparse
import numpy as np
import open3d as o3d
import open3d.visualization.gui as gui
import open3d as o3d
from glob import glob


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="path to base_path location", type=str)
    args = parser.parse_args(argv)
    return args


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


def draw_box(img, arranged_points, draw_index=False, edge_version=1):
    """
    plot arranged_points on img.
    arranged_points: list of points [[x, y]] in image coordinate. 
    """
    img = np.copy(img)
    RADIUS = 10
    COLOR = (0, 255, 0) # (255, 255, 255)
    EDGES = [
        [
            [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
            [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
            [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
        ],
        [
            [1, 2], [4, 3], [5, 6], [8, 7],
            [1, 5], [4, 8], [2, 6], [3, 7],
            [1, 4], [2, 3], [6, 7], [5, 8] 
        ]
    ] 
    EDGES = EDGES[edge_version]
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
    return img


def denormalize_bbox(bbox_points, pcd_points):
    x_min = pcd_points.min(0)
    x_max = pcd_points.max(0)
    label_translation = ((x_min + x_max) / 2).reshape((1,3))
    label_scale = (x_max - x_min).max()
    bbox_points_denorm = label_scale * bbox_points + label_translation
    return bbox_points_denorm


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    base_path = args.input
    # "/home/myungseo/works/Open3D/kinetic_tutorial/data/food_stop_3"
    pcd_dir_path = os.path.join(base_path, "scene")
    annotation_paths = sorted(glob(os.path.join(base_path, "annotation/*.json")))
    intrinsic_path = f"{base_path}/intrinsic.json"
    img_dir_path = os.path.join(base_path, "selected")
    out_dir_path = os.path.join(base_path, "selected_with_bbox")
    
    if os.path.exists(out_dir_path):
        shutil.rmtree(out_dir_path)
    os.makedirs(out_dir_path)

    # Load intrinsic matrix
    with open(intrinsic_path, "r") as f:
        intrinsic_json = json.load(f)
        K = np.array(intrinsic_json["intrinsic_matrix"]).reshape(3,3).T
        K = np.concatenate([K, np.zeros((3,1))], axis=1)

    for annotation_path in annotation_paths:
        filename = os.path.splitext(os.path.basename(annotation_path))[0]
        pcd_path = os.path.join(pcd_dir_path, f"{filename}.ply")
        img_path = os.path.join(img_dir_path, f"{filename}.jpg")


        pcd = o3d.io.read_point_cloud(pcd_path)
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)

        # Load 3D bounding box
        bbox_keys = ['ddd', 'udd', 'udu', 'ddu', 'dud', 'uud', 'uuu', 'duu']
        with open(annotation_path, "r") as f:
            annotation_json = json.load(f)[0]["boxes"]
        label2ann = {ann["label"]: ann for ann in annotation_json}
        label2bbox = dict()

        for label, ann in label2ann.items():
            label2bbox[label] = []
            for k in bbox_keys:
                coord = ann[k]
                bbox_point = [coord["x"], coord["y"], coord["z"]]
                label2bbox[label] += [bbox_point]
            label2bbox[label] = np.array(label2bbox[label])

        # Denormalize the bounding box
        for label in label2bbox:
            bbox_points = denormalize_bbox(label2bbox[label], points)
            bbox_pcd = o3d.geometry.PointCloud()
            bbox_pcd.points = o3d.utility.Vector3dVector(bbox_points)
            bbox_pcd.paint_uniform_color([1, 0, 0])
            label2bbox[label] = bbox_pcd

        # Visualize the projected bounding box
        imgname = os.path.basename(img_path)
        Rt = np.eye(4,4)
        Rt[1,1] = -1
        Rt[2,2] = -1
        img = cv2.imread(img_path)
        for label, bbox in label2bbox.items():
            x_bbox_proj = world2pixel(np.asarray(bbox.points), K, Rt, k3=None).astype(int).T
            img = draw_box(img, x_bbox_proj, draw_index=True)
        out_path = os.path.join(out_dir_path, imgname)
        cv2.imwrite(out_path, img)
        print(f"{out_path} is saved.")
