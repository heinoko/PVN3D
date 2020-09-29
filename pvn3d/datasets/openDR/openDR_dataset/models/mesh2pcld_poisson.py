import open3d as o3d
import numpy as np
import argparse
import os


parser = argparse.ArgumentParser(description="Arg parser")


parser.add_argument(
    "-dataset",
    type=str,
    default='openDR',
    help="Define dataset you want to save the pointcloud to",
)


parser.add_argument(
    "-obj_id",
    type=int ,
    default=0,
    help="Class ID of the object ",
)

parser.add_argument(
    "-mesh_file",
    type=str ,
    default=0,
    help="path of the .stl .ply or .obj mesh file ",
)
args = parser.parse_args()


mesh = o3d.io.read_triangle_mesh(str(args.mesh_file))
poisson_pcld = mesh.sample_points_poisson_disk(number_of_points=30000)

o3d.io.write_point_cloud('./datasets/'+str(args.dataset)+str(args.dataset)+'_dataset/models/obj_'+str(args.obj_id), poisson_pcld )  


