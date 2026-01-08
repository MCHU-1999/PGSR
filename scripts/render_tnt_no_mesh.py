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
import os, sys
from pathlib import Path
dir_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]
print(f"dir_path {dir_path}")
sys.path.append(dir_path.__str__())

import torch
from scene import Scene
import json
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import numpy as np
import cv2
from scene.app_model import AppModel

def render_set_no_mesh(model_path, name, iteration, views, scene, gaussians, pipeline, background, 
               app_model=None):
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_depth")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_normal")

    makedirs(gts_path, exist_ok=True)
    makedirs(render_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt, _ = view.get_image()
        out = render(view, gaussians, pipeline, background, app_model=app_model, image_scaling_coef=2)
        rendering = out["render"]

        # Depth and normal maps for feeding into PlanarSplatting (.npy files)        
        # Process and save depth map
        depth_map = out["plane_depth"].squeeze() # Shape: (H, W)
        depth_map_clamped = torch.clamp(depth_map, min=0, max=300) # Clamp values to (0, 300)
        depth_np = depth_map_clamped.cpu().numpy()
        np.save(os.path.join(render_depth_path, view.image_name + ".npy"), depth_np)

        # Process and save normal map
        normal_map = out["rendered_normal"]  # Shape: (3, H, W)
        normal_map = (normal_map + 1.0) / 2.0 # Remap from [-1, 1] to [0, 1]
        normal_np = normal_map.cpu().numpy()
        np.save(os.path.join(render_normal_path, view.image_name + ".npy"), normal_np)

        if name == 'test':
            torchvision.utils.save_image(gt, os.path.join(gts_path, view.image_name + ".png"))
            torchvision.utils.save_image(rendering, os.path.join(render_path, view.image_name + ".png"))
        else:
            rendering_np = (rendering.permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
            cv2.imwrite(os.path.join(render_path, view.image_name + ".jpg"), rendering_np)

def render_sets_no_mesh(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, 
                        skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            render_set_no_mesh(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), scene, 
                               gaussians, pipeline, background)
        if not skip_test:
            render_set_no_mesh(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), scene, 
                               gaussians, pipeline, background)

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max_depth", default=20.0, type=float, help='Will not function in this "no mesh" script')
    parser.add_argument("--voxel_size", default=0.002, type=float, help='Will not function in this "no mesh" script')
    parser.add_argument("--num_cluster", default=1, type=int, help='Will not function in this "no mesh" script')
    parser.add_argument("--use_depth_filter", action="store_true", help='Will not function in this "no mesh" script')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Will not function in this "no mesh" script')

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    print(f"multi_view_num {model.multi_view_num}")
    render_sets_no_mesh(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)