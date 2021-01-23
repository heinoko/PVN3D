import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')		#This is to remove ROS-python from the PYTHONPATH which messes up the Python 3 env this project works with
from lib.utils.basic_utils import Basic_Utils
from common import Config
import numpy as np
from PIL import Image
import pcl
from os import listdir

''' This script writes cloud, normals and sampled-points for all the training data so it does not have to be calculated from training images during the training '''

cfg = Config(dataset_name='CrankSlider')
bs_utils = Basic_Utils(cfg)

def get_normal( cld):
        cloud = pcl.PointCloud()
        cld = cld.astype(np.float32)
        cloud.from_array(cld)
        ne = cloud.make_NormalEstimation()
        kdtree = cloud.make_kdtree()
        ne.set_SearchMethod(kdtree)
        ne.set_KSearch(50)
        n = ne.compute()
        n = n.to_array()
        return n

## This script is only for openDR dataset ##

for i,_ in enumerate(listdir('./datasets/CrankSlider/CrankSlider_dataset/depth')):
#for i in range(2304, 3071):#2304):

    with Image.open('./datasets/CrankSlider/CrankSlider_dataset/depth/'+str(i)+'.png') as dpt_im:
        dpt = np.array(dpt_im)
        print(dpt.dtype)
        print(dpt.max())
        dpt = dpt/65535				#Scaling down between 0 - 3m
        
        dpt = dpt*3.0
    
    print(dpt.max())

    #Back-projection util function
    cld, choose = bs_utils.dpt_2_cld(dpt, 1, cfg.intrinsic_matrix['CrankSlider'])
    normals = get_normal(cld)
    all_arr = np.concatenate( (cld, choose.reshape(choose.shape[0],1), normals[:,:3]) , axis = 1)

    #Write cloud, choose-filter and normals in npy files for retrieval in dataloader
    ''' This is to prevent repeated calculations while Batch-loading during training,
            as it takes a considerable amount of time for each batch. '''

    with open('./datasets/CrankSlider/CrankSlider_dataset/'+str(i)+'.npy', 'wb') as f:

        print('Writing file '+str(i))
        np.save(f, all_arr)

		
