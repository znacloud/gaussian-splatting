#
# Copyright (C) 2024, Nianan Zeng
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  znacloud@gmail.com
#

import os
import json
import numpy as np
from PIL import Image
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from typing import NamedTuple

# from scene.dataset_readers import CameraInfo

def move_camera_along_local_z(C, delta_z):
    """
    Move the camera along its local Z-axis.

    Args:
        C (numpy.ndarray): Original camera-to-world pose matrix.
        delta_z (float): Amount by which to move the camera along its local Z-axis.

    Returns:
        numpy.ndarray: Updated camera-to-world pose matrix.
    """
    # Decompose the camera-to-world matrix into translation, rotation, and scale
    translation = C[:3, 3]  # Extract translation component
    rotation = C[:3, :3]  # Extract rotation component

    # Get the local Z-axis in world coordinates
    local_z_world = rotation[:, 2]

    # Transform the translation along the local Z-axis to world coordinates
    translation_delta_world = local_z_world * delta_z

    # Update the translation in world coordinates
    new_translation = translation + translation_delta_world

    # Recombine the updated translation with the original rotation and scale
    updated_C = np.eye(4)
    updated_C[:3, :3] = rotation
    updated_C[:3, 3] = new_translation

    return updated_C


def transform_to_first_camera_coordinate_system(C, first_camera_C_inv):
    """
    Transform camera-to-world pose matrix to the coordinate system of the first camera.

    Args:
        C (numpy.ndarray): Camera-to-world pose matrix.
        first_camera_C_inv (numpy.ndarray): Inverse of camera-to-world pose matrix of the first camera.

    Returns:
        numpy.ndarray: Transformed camera-to-world pose matrix.
    """
    # Transform C relative to the origin
    # relative_to_origin = np.linalg.inv(C)
    relative_to_origin = C

    # Transform relative_to_origin into the coordinate system of the first camera
    transformed_C = first_camera_C_inv @ relative_to_origin

    return transformed_C


def transform_points_to_camera_coordinate_system(points_world, camera_C_inv):
    """
    Transform points from the world coordinate system to a camera's coordinate system.

    Args:
        points_world (numpy.ndarray): Points in the world coordinate system.
        camera_C (numpy.ndarray): Camera-to-world pose matrix of the camera.

    Returns:
        numpy.ndarray: Transformed points in the camera's coordinate system.
    """
    # Add a homogeneous coordinate of 1 to points_world
    points_world_homogeneous = np.hstack((points_world, np.ones((points_world.shape[0], 1))))

    # Transform points to the camera's coordinate system
    points_camera = np.dot(points_world_homogeneous, np.transpose(camera_C_inv))

    return points_camera[:,:-1]

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int


def read_intrinsics_json(path):
    cameras = {}
    with open(path, "r") as fid:
        intr_obj = json.load(fid)

        for key in intr_obj:
            model = intr_obj[key]["model"]
            assert (
                model == "PINHOLE"
            ), "While the loader support other types, the rest of the code assumes PINHOLE"

        cameras = intr_obj
    return cameras


def read_extrinsics_json(path):
    images = {}
    with open(path, "r") as fid:
        extr_obj = json.load(fid)

    images = extr_obj
    return images


def read_conf_points3D_text(path, w2c):
    xyzs = None
    rgbs = None
    confs = None
    masks = None
    num_points = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                num_points += 1

    xyzs = np.empty((num_points, 3))
    rgbs = np.empty((num_points, 3))
    confs = np.empty(num_points)
    masks = np.empty(num_points, dtype=np.uint8)
    count = 0
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split(",")
                xyz = np.array(tuple(map(float, elems[0:3])))
                rgb = np.array(tuple(map(float, elems[3:6])))
                conf = float(elems[6])
                mask = int(float(elems[7]))
                # xyz[1:3] *= -1
                xyzs[count] = xyz
                rgbs[count] = rgb
                confs[count] = conf
                masks[count] = mask
                count += 1

    xyzs = transform_points_to_camera_coordinate_system(xyzs, w2c)

    return xyzs, rgbs, confs, masks


def readDust3rCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    first_c2w_inv = None
    for idx, key in enumerate(cam_extrinsics):
        # the exact output you're looking for:
        print("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr["camera_id"]]
        height = intr["height"]
        width = intr["width"]

        uid = intr["id"]
        R = np.reshape(extr["rflat"], (3, 3))
        T = np.reshape(extr["tvec"], (-1,1))
        # DUSt3R camera pose is a camera-to-world transform
        c2w = np.eye(4,4)
        c2w[:3,:3]=R
        c2w[:3,3:4]=T

        # Test quality by moving the camera distance to object
        # c2w = move_camera_along_local_z(c2w,-0.1)

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        # c2w[:3, 1:3] *= -1

       # Transform to first camera coordinate
        if first_c2w_inv is None:
            first_c2w_inv = np.linalg.inv(c2w)
        
        c2w = transform_to_first_camera_coordinate_system(c2w, first_c2w_inv)

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)

        R = np.transpose(
            w2c[:3, :3]
        )  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        if intr["model"] == "SIMPLE_PINHOLE" or intr["model"] == "PINHOLE":
            focal_length_x = intr["focal_x"]
            focal_length_y = intr["focal_y"]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert (
                False
            ), "DUSt3R camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr["img_name"]))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(
            uid=uid,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=image_name,
            width=width,
            height=height,
        )
        cam_infos.append(cam_info)
    return cam_infos, first_c2w_inv
