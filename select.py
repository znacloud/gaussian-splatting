#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from pathlib import Path
import os
from PIL import Image
import torch
import torchvision.transforms.functional as tf
from utils.loss_utils import ssim
from lpipsPyTorch import lpips
import json
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
import shutil


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, group_length, novel_paths):

    per_view_dict = {}
    print("")

    for scene_dir in model_paths:
        try:
            print("Scene:", scene_dir)

            per_view_dict[scene_dir] = {}

            metrics_dir = Path(scene_dir) / "train"
            for method in os.listdir(metrics_dir):
                print("Method:", method)

                per_view_dict[scene_dir][method] = {}

                method_dir = metrics_dir / method
                gt_dir = method_dir / "gt"
                renders_dir = method_dir / "renders"
                renders, gts, image_names = readImages(renders_dir, gt_dir)

                ssims = []

                for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
                    ssims.append(ssim(renders[idx], gts[idx]))

                print("  SSIM : {:>12.7f}".format(torch.tensor(ssims).mean(), ".5"))
                print("")

                ssim_list = [
                    {name: ssim}
                    for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)
                ]  # .sort(key=lambda x: x["name"], reverse=True)

                # Select the one who has the maximum ssim value from every group of specified length
                largest_items = []
                for i in range(0, len(ssim_list), group_length):
                    group = ssim_list[i : i + group_length]
                    max_item = max(group, key=lambda x: list(x.values())[0])
                    largest_items.append(max_item)

                # Save results
                per_view_dict[scene_dir][method].update(
                    {
                        "SSIM": ssim_list,
                        "MAX_SSIM": largest_items,
                        "G_LENGTH": group_length,
                    }
                )

                # select images from novel paths based on the largest_items
                selected_img_names = [list(item.keys())[0] for item in largest_items]
                for novel_dir in novel_paths:
                    novel_method_dir = Path(scene_dir) / novel_dir / method
                    novel_render_dir = novel_method_dir / "renders"
                    novel_train_dir = (
                        Path(scene_dir) / f"{method}-{novel_dir}-train-g{group_length}"
                    )

                    novel_train_dir.mkdir(parents=True, exist_ok=True)

                    # Copy the file to the destination directory
                    for img_name in selected_img_names:
                        name, _, suffix = img_name.rpartition(".")
                        src = novel_render_dir / img_name
                        dst = novel_train_dir / f"{name}_nv.{suffix}"
                        print("Copy file: ", src)
                        shutil.copy(src, dst)

                        src = renders_dir / img_name
                        dst = novel_train_dir / img_name
                        print("Copy file: ", src)
                        shutil.copy(src, dst)

            with open(scene_dir + "/per_view.json", "w") as fp:
                json.dump(per_view_dict[scene_dir], fp, indent=True)
        except Exception as e:
            print(e)
            print("Unable to compute metrics for model", scene_dir)


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument(
        "--model_paths", "-m", required=True, nargs="+", type=str, default=[]
    )
    parser.add_argument("--group_length", "-g", default=10, type=int)

    parser.add_argument("--novel_paths", required=True, nargs="+", type=str, default=[])

    args = parser.parse_args()
    evaluate(args.model_paths, args.group_length, args.novel_paths)
