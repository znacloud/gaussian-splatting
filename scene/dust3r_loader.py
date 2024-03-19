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

def read_conf_points3D_text(path):
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
                rgb = np.array(tuple(map(int, elems[3:6])))
                conf = float(elems[6])
                mask = int(elems[7])
                xyzs[count] = xyz
                rgbs[count] = rgb
                confs[count] = conf
                masks[count] = mask
                count += 1

    return xyzs, rgbs, confs, masks


def readDust3rCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        # the exact output you're looking for:
        print("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr["camera_id"]]
        height = intr["height"]
        width = intr["width"]

        uid = intr["id"]
        R = np.reshape(extr["rflat"], (3, 3))
        T = np.transpose(extr["tvec"])

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
    return cam_infos
