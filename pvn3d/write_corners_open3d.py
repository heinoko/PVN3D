import sys
#.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')		#This is to remove ROS-python from the PYTHONPATH which messes up the Python 3 env this project works with
import pcl
#import lib.utils.basic_utils as bs_utl
import numpy as np
import argparse
import os
import struct
''' This script loads a pcd or ply file of an object model and extract the corners of it's bounding-box and saves it to a txt file '''

def XYZ_to_XYZRGB(XYZ_cloud, color):
	""" Converts a PCL XYZ point cloud to a PCL XYZRGB point cloud

        All returned points in the XYZRGB cloud will be the color indicated
        by the color parameter.

        Args:
            XYZ_cloud (PointCloud_XYZ): A PCL XYZ point cloud
            color (list): 3-element list of integers [0-255,0-255,0-255]

        Returns:
            PointCloud_PointXYZRGB: A PCL XYZRGB point cloud
    """
	XYZRGB_cloud = pcl.PointCloud_PointXYZRGB()
	points_list = []

	for i, data in enumerate(XYZ_cloud):
		float_rgb = rgb_to_float(color[i])
		points_list.append([data[0], data[1], data[2], float_rgb])

	XYZRGB_cloud.from_list(points_list)
	return XYZRGB_cloud


def rgb_to_float(color):
	""" Converts an RGB list to the packed float format used by PCL

        From the PCL docs:
        "Due to historical reasons (PCL was first developed as a ROS package),
         the RGB information is packed into an integer and casted to a float"

        Args:
            color (list): 3-element list of integers [0-255,0-255,0-255]

        Returns:
            float_rgb: RGB value packed as a float
    """
	hex_r = (0xff & int(color[0])) << 16
	hex_g = (0xff & int(color[1])) << 8
	hex_b = (0xff & int(color[2]))

	hex_rgb = hex_r | hex_g | hex_b

	float_rgb = struct.unpack('f', struct.pack('i', hex_rgb))[0]

	return float_rgb
parser = argparse.ArgumentParser(description="Arg parser")


parser.add_argument(
    "-dataset",
    type=str,
    default='openDR',
    help="Define dataset you want to save the pointcloud to",
)

parser.add_argument(
    "-cls_id",
    type=int ,
    default=2,
    help="Class ID of the object ",
)

args = parser.parse_args()



dataset_folder = {'openDR':'openDR_dataset', 'linemod':'Linemod_preprocessed', 'ycb':'ycb_dataset', 'CrankSlider': 'CrankSlider_dataset', 'EngineParts':'EngineParts_dataset'}
kps_folder = {'openDR':'openDR_object_kps', 'linemod':'lm_object_kps', 'ycb':'ycb_object_kps', 'CrankSlider': 'CrankSlider_object_kps', 'EngineParts':'EngineParts_object_kps'}




cls_ids = args.cls_id

if args.dataset == 'openDR':
	cls_ids = [1,2,3,4,5,6,7,8,9,10]
elif args.dataset == 'EngineParts':
	cls_ids = [1]



for cls_id in cls_ids:
	print('Writing corners for class id '+str(cls_id))
	pcd = pcl.load('./datasets/'+args.dataset+'/'+dataset_folder[args.dataset]+'/models/obj_'+str(cls_id)+'.ply')
	pts = np.array(pcd)
	#fps_idx = bs_utl.farthestPointSampling(pts, 8)
	corners = []
	for i in range(0,2):	
	
		if (i == 0):
			x = np.min(pts[:,0])		
		else:
			x = np.max(pts[:,0])

		for j in range(0,2):
		
			if (j == 0):
				y = np.min(pts[:,1])		
			else:
				y = np.max(pts[:,1])
		
			for k in range(0,2):
		
				if (k == 0):
					z = np.min(pts[:,2])		
				else:
					z = np.max(pts[:,2])
	
				corners.append([x,y,z])


	corners = np.array(corners)
	#print(corners.shape)
	pts = np.vstack(( pts, corners ))
	colors = 100*np.ones((pts.shape[0],3))
	colors[-8:-1,:] = np.array([255, 0 , 0])

	colors = colors.tolist()

	cloud_corners = XYZ_to_XYZRGB(pts, colors)


	np.savetxt('./datasets/'+args.dataset+'/'+kps_folder[args.dataset]+'/'+str(cls_id)+'/corners.txt', corners)	#corners
	pcd = pcl.save(cloud_corners,'./datasets/'+args.dataset+'/'+kps_folder[args.dataset]+'/'+str(cls_id)+'/corners.pcd')  ##PCD file with corners added to the actual cloud
