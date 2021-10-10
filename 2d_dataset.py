"""
Demo code to load the FLAME Layer and visualise the 3D landmarks on the Face 

Author: Soubhik Sanyal
Copyright (c) 2019, Soubhik Sanyal
All rights reserved.

Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights on this
computer program.
You can only use this computer program if you have closed a license agreement with MPG or you get the right to use
the computer program from someone who is authorized to grant you that right.
Any use of the computer program without a valid license is prohibited and liable to prosecution.
Copyright 2019 Max-Planck-Gesellschaft zur Foerderung der Wissenschaften e.V. (MPG). acting on behalf of its
Max Planck Institute for Intelligent Systems and the Max Planck Institute for Biological Cybernetics.
All rights reserved.

More information about FLAME is available at http://flame.is.tue.mpg.de.

For questions regarding the PyTorch implementation please contact soubhik.sanyal@tuebingen.mpg.de
"""
from itertools import product
import numpy as np
import torch
from FLAME import FLAME
import pyrender
import pyvista as pv
import trimesh
from config import get_config

radian = np.pi / 180.0
no_people = 1

# For 0 we can have a normal distribution
lips_int = 3.2
lips_stretch_param = torch.arange(-6, 10, lips_int)
mouth_int = 2.0
mouth_openness_param = torch.arange(0, 10, mouth_int)
x_grid, y_grid = torch.meshgrid(lips_stretch_param, mouth_openness_param)

# Creating a batch of neutral expressions
no_exps = x_grid.shape[0] * x_grid.shape[1]
expression_params = torch.zeros(no_exps, 50, dtype=torch.float32)
expression_params[:, 2] = x_grid.flatten() 
expression_params[:, 3] = y_grid.flatten()
expression_params = expression_params.repeat(no_people, 1)
# 0 is the stretch along the mouth line
# 1 is the lips puckering 
# 2 is position of lips, left or right
# 3 could be mouth opening and closing
# 9 is happy to sad

# Pose for each image
pose_params = torch.zeros(no_exps * no_people, 6, dtype=torch.float32)

# Creating a batch of mean shapes
shape_params = torch.randn(no_people, 100).repeat_interleave(no_exps, 0)

# Forward Pass of FLAME, one can easily use this as a layer in a Deep learning Framework 
config = get_config()
config.batch_size = no_exps * no_people
flamelayer = FLAME(config)
vertice, landmark = flamelayer(shape_params, expression_params, pose_params) # For RingNet project

faces = flamelayer.faces
plotter = pv.Plotter(shape=x_grid.shape, window_size=[1500, 2000])  # , lighting='three lights')

for i, (x, y) in enumerate(product(range(x_grid.shape[0]), range(x_grid.shape[1]))):
    plotter.subplot(x, y)
    vertices = vertice[i].detach().cpu().numpy().squeeze()
    tri_mesh = trimesh.Trimesh(vertices, faces)
    pv_mesh = pv.wrap(tri_mesh)
    pv_mesh.rotate_x(angle=90, point=axes.origin)
    pv_mesh.rotate_z(angle=-15, point=axes.origin)

    plotter.set_background('w')
    light = pv.Light(position=(0, 10, 0), intensity=0.5, light_type='scene light')
    plotter.add_light(light)
    # plotter.camera.Zoom(10)
    # plotter.camera_set = True
    plotter.add_mesh(pv_mesh, opacity=1.0, color="#759198", show_edges=False)

plotter.show(screenshot='meshes_sample.png', cpos='xz')
