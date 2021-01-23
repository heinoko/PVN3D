#!/usr/bin/env python3
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import sys
#import pcl
import open3d as o3d
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from lib.utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
from cv2 import imshow, waitKey
from scipy.spatial.transform import Rotation as Rot
import time
config = Config(dataset_name='CrankSlider')
bs_utils = Basic_Utils(config)
DEBUG = False


class CrankSlider_Dataset():

    def __init__(self, dataset_name):
        self.dataset_name = dataset_name
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        #self.diameters = {}
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        #self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.cls_lst = bs_utils.read_lines(config.CrankSlider_cls_lst_p)
        #self.obj_dict = {'piston':11, 'round_peg':2, 'square_peg':3, 'pendulum':4, 'pendulum_head':5, 'separator':6, 'shaft':7, 'face_plate':8, 'valve_tappet':9, 'shoulder_bolt':10}
        '''
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id'''
        self.rng = np.random
        if dataset_name == 'train':
            self.add_noise = True
            self.path = 'datasets/CrankSlider/CrankSlider_dataset/train.txt'
            self.all_lst = bs_utils.read_lines(self.path)
            self.real_lst = []
            self.syn_lst = []
            for item in self.all_lst:
                #if item[:5] == 'data/':
                self.real_lst.append(item)
                #else:
                #self.syn_lst.append(item)
        else:
            self.pp_data = None
            if os.path.exists(config.preprocessed_testset_pth) and config.use_preprocess:
                print('Loading valtestset.')
                with open(config.preprocessed_testset_pth, 'rb') as f:
                    self.pp_data = pkl.load(f)
                self.all_lst = [i for i in range(len(self.pp_data))]
                print('Finish loading valtestset.')
            else:
                self.add_noise = False
                self.path = 'datasets/CrankSlider/CrankSlider_dataset/test.txt'
                self.all_lst = bs_utils.read_lines(self.path)
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))
        self.root = config.CrankSlider_root
        self.sym_cls_ids = config.CrankSlider_sym_cls_ids

    def real_syn_gen(self):
        if self.rng.rand() > 0.8:
            n = len(self.real_lst)
            idx = self.rng.randint(0, n)
            item = self.real_lst[idx]
        else:
            n = len(self.syn_lst)
            idx = self.rng.randint(0, n)
            item = self.syn_lst[idx]
        return item

    def real_gen(self):
        n = len(self.real_lst)
        idx = self.rng.randint(0, n)
        item = self.real_lst[idx]
        return item

    def rand_range(self, rng, lo, hi):
        return rng.rand()*(hi-lo)+lo

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, : ,1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def get_normal_or(self, cld):
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

    def get_normal(self, cld):
       ''' Open3d based normal estimation '''
       cloud = o3d.geometry.PointCloud()
       cloud.points = o3d.utility.Vector3dVector(cld)

       cloud.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))

       cloud.orient_normals_towards_camera_location()
       n = np.asarray(cloud.normals)
       return n

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back==0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <=0).astype(dpt.dtype)
        return rgb, dpt

    def get_item(self, item_name):
        start = time.time()*1000.0
        try:
            #print( self.root)

            '''with Image.open(os.path.join("./datasets/openDR/openDR_dataset/depth/{}.png".format(item_name) )) as di:
                dpt = np.array(di)
                dpt = dpt/65535
                dpt = dpt*3.0'''
            with Image.open(os.path.join("./datasets/CrankSlider/CrankSlider_dataset/mask/{}.png".format(item_name))) as li:
                labels = np.array(li)
                #cls_id_lst = np.unique(labels[labels.nonzero()])
            meta = scio.loadmat(os.path.join('./datasets/CrankSlider/CrankSlider_dataset/meta/', item_name+'-meta.mat'))
            #if item_name[:8] != 'data_syn' and int(item_name[5:9]) >= 60:
            #    K = config.intrinsic_matrix['openDR']
            #else:
            K = config.intrinsic_matrix['CrankSlider']

            with Image.open(os.path.join(self.root + '/rgb/{}.png'.format(item_name))) as ri:
                #ri = np.repeat(np.expand_dims( ri, axis = 2), 3, axis = 2) # For gray image with 1 channel
                '''
                if self.add_noise:
                    ri = self.trancolor(ri)'''
                rgb = np.array(ri)[:, :, :3]

            ######################## Grey value for each class i = (255*(i+1)/8)
 
            #In [7]: for i in range(0,7):
            #       ...:     print(1.05*255*(i+1)/8)
             
            #   33.46875
            #   66.9375
            #   100.40625
            #   133.875
            #   167.34375
            #   200.8125
            #   234.28125
            #########################################


            for i, val in enumerate([ 33,  67,  100, 134, 167, 201, 234]):
                labels[np.where(labels == val )] = i + 1
            rnd_typ = 'syn' if 'syn' in item_name else 'real'
            cam_scale = 1 #meta['factor_depth'].astype(np.float32)[0][0]
            #msk_dp = dpt > 1e-6

            if self.add_noise and rnd_typ == 'syn':
                rgb = self.rgb_add_noise(rgb)
                rgb_labels = labels.copy()
                rgb, dpt = self.add_real_back(rgb, rgb_labels, dpt, msk_dp)
                if self.rng.rand() > 0.8:
                    rgb = self.rgb_add_noise(rgb)

            #dpt = bs_utils.fill_missing(dpt, cam_scale, 1)
            #msk_dp = dpt > 1e-6

            rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw
            ## Depth-img back projection and normal estimation takes too long during dataloading and wastes
            ## useful training time before loading batches at the beginning of every epoch. So we precalcuated
            ## these and read them as npy files in this dataloader to save time.
            #cld, choose = bs_utils.dpt_2_cld(dpt, cam_scale, K)    ## Original calculation
            cld_choose_normals = np.load( os.path.join(self.root, 'cld_choose_norms/{}.npy'.format(item_name)) )
            #normal = self.get_normal(cld)[:, :3]
            cld = cld_choose_normals[:,:3]
            choose = np.array(cld_choose_normals[:,3], dtype= np.uint32)
            normal = cld_choose_normals[:,4:7]
            normal[np.isnan(normal)] = 0.0


            labels = labels.flatten()[choose]
            rgb_lst = []
            for ic in range(rgb.shape[0]):
                rgb_lst.append(
                    rgb[ic].flatten()[choose].astype(np.float32)
                )
            rgb_pt = np.transpose(np.array(rgb_lst), (1, 0)).copy()

            choose = np.array([choose])
            choose_2 = np.array([i for i in range(len(choose[0, :]))])


            if len(choose_2) < 400:
                return None
            if len(choose_2) > config.n_sample_points:
                c_mask = np.zeros(len(choose_2), dtype=int)
                c_mask[:config.n_sample_points] = 1
                np.random.shuffle(c_mask)
                choose_2 = choose_2[c_mask.nonzero()]
            else:
                choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')

            cld_rgb_nrm = np.concatenate((cld, rgb_pt, normal), axis=1)
            cld = cld[choose_2, :]
            cld_rgb_nrm = cld_rgb_nrm[choose_2, :]
            choose = choose[:, choose_2]
            labels = labels[choose_2].astype(np.int32)

            RTs = np.zeros((config.n_objects, 3, 4))
            kp3ds = np.zeros((config.n_objects, config.n_keypoints, 3))
            ctr3ds = np.zeros((config.n_objects, 3))
            cls_ids = np.zeros((config.n_objects, 1))
            kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3))
            ctr_targ_ofst = np.zeros((config.n_sample_points, 3))
            #cls_id_lst = meta['cls_indexes'].flatten().astype(np.uint32)
            opt2cam_R = Rot.from_euler('zyx',[1.57,0,1.57])
            opt2cam_R= opt2cam_R.as_dcm()

            for i, cls_id in enumerate(range(1,8)):
                r = meta['poses'][:, :, i][:, 0:3]
                r = np.matmul( opt2cam_R, r )
                t = np.array(meta['poses'][:, :, i][:, 3:4].flatten()[:, None])
                t = np.array([-t[1], -t[2], t[0]])
                RT = np.concatenate((r, t), axis=1)
                RTs[i] = RT

                ctr = bs_utils.get_ctr(i, ds_type='CrankSlider').copy()[:, None]
                ctr = np.dot(ctr.T, r.T) + t[:, 0]
                ctr3ds[i, :] = ctr[0]
                msk_idx = np.where(labels == cls_id)[0]

                target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
                ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
                cls_ids[i, :] = np.array([cls_id])

                key_kpts = ''
                if config.n_keypoints == 8:
                    kp_type = 'farthest_8'
                else:
                    kp_type = 'farthest_{}'.format(config.n_keypoints)
                kps = bs_utils.get_kps(
                    i, kp_type=kp_type, ds_type='CrankSlider'
                ).copy()
                kps = np.dot(kps, r.T) + t[:, 0]
                kp3ds[i] = kps

                target = []
                for kp in kps:
                    target.append(np.add(cld, -1.0*kp))
                target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
                kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
            '''
            print('kp3ds shape '+ str(kp3ds.shape))
            print('ctr3ds shape '+ str(ctr3ds.shape))
            print('labels NZ '+str(labels[labels.nonzero()].shape))
            print('cls_ids '+ str(cls_ids.T ))
            print('Labels shape '+str(labels.shape))
            print('kp_targ_ofst shape '+str(kp_targ_ofst.shape))
            print('RTs shape ' + str(RTs.shape) )
            print('rgb shape '+ str(rgb.shape))
            print('cld_rgb_nrm shape'+str(cld_rgb_nrm.shape))
            print('cls_ids shape'+str(cls_ids.shape))
            print('all Labels '+ str(np.unique(labels) ))
            print(' ')'''
            #print(rgb.dtype)
            #print('all Labels '+ str(cls_ids.T) )
            #print('all Labels '+ str(np.unique(labels) ))
            #print('Time norm: '+str(time.time()*1000.0-start)+' ms per item')

            # rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, ctr_targ_ofst, cls_ids, RTs, labels, kp_3ds, ctr_3ds
            if DEBUG:
                return  torch.from_numpy(rgb.astype(np.float32)), \
                        torch.from_numpy(cld.astype(np.float32)), \
                        torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
                        torch.LongTensor(choose.astype(np.int32)), \
                        torch.from_numpy(kp_targ_ofst.astype(np.float32)), \
                        torch.from_numpy(ctr_targ_ofst.astype(np.float32)), \
                        torch.LongTensor(cls_ids.astype(np.int32)), \
                        torch.from_numpy(RTs.astype(np.float32)), \
                        torch.LongTensor(labels.astype(np.int32)), \
                        torch.from_numpy(kp3ds.astype(np.float32)), \
                        torch.from_numpy(ctr3ds.astype(np.float32)), \
                        torch.from_numpy(K.astype(np.float32)), \
                        torch.from_numpy(np.array(cam_scale).astype(np.float32))

            return  torch.from_numpy(rgb.astype(np.float32)), \
                    torch.from_numpy(cld.astype(np.float32)), \
                    torch.from_numpy(cld_rgb_nrm.astype(np.float32)), \
                    torch.LongTensor(choose.astype(np.int32)), \
                    torch.from_numpy(kp_targ_ofst.astype(np.float32)), \
                    torch.from_numpy(ctr_targ_ofst.astype(np.float32)), \
                    torch.LongTensor(cls_ids.astype(np.int32)), \
                    torch.from_numpy(RTs.astype(np.float32)), \
                    torch.LongTensor(labels.astype(np.int32)), \
                    torch.from_numpy(kp3ds.astype(np.float32)), \
                    torch.from_numpy(ctr3ds.astype(np.float32)),
        except Exception as inst:
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    print('exception: '+str(inst)+' in '+ str(exc_tb.tb_lineno))
                    return None


    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        if self.dataset_name == 'train':
            item_name = self.real_gen()
            #print('Data Loader starting...')
            data = self.get_item(item_name)
            while data is None:
                item_name = self.real_gen()
                data = self.get_item(item_name)
            return data
        else:
            if self.pp_data is None or not config.use_preprocess:
                item_name = self.all_lst[idx]
                return self.get_item(item_name)
            else:
                data = self.pp_data[idx]
                return data


def main():
    # config.mini_batch_size = 1
    global DEBUG
    DEBUG = True
    ds = {}
    # ds['train'] = openDR_Dataset('train')
    # ds['val'] = openDR_Dataset('validation')
    ds['test'] = CrankSlider_Dataset('test')
    idx = dict(
        train=0,
        val=0,
        test=0
    )
    while True:
        # for cat in ['val', 'test']:
        for cat in ['test']:
        # for cat in ['train']:
            datum = ds[cat].__getitem__(idx[cat])
            idx[cat] += 1
            datum = [item.numpy() for item in datum]
            if cat == "train":
                rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, \
                    ctr_targ_ofst, cls_ids, RTs, labels, kp3ds, ctr3ds, K, cam_scale = datum
            else:
                rgb, pcld, cld_rgb_nrm, choose, kp_targ_ofst, \
                    ctr_targ_ofst, cls_ids, RTs, labels, kp3ds, ctr3ds = datum
                K = config.intrinsic_matrix['CrankSlider']
                cam_scale = 1.0
            nrm_map = bs_utils.get_normal_map(cld_rgb_nrm[:, 6:], choose[0])
            imshow('nrm_map', nrm_map)
            rgb = rgb.transpose(1, 2, 0)[...,::-1].copy()# [...,::-1].copy()
            for i in range(22):
                p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
                # rgb = bs_utils.draw_p2ds(rgb, p2ds)
                kp3d = kp3ds[i]
                if kp3d.sum() < 1e-6:
                    break
                kp_2ds = bs_utils.project_p3d(kp3d, cam_scale, K)
                rgb = bs_utils.draw_p2ds(
                    rgb, kp_2ds, 3, bs_utils.get_label_color(cls_ids[i], mode=1)
                )
                ctr3d = ctr3ds[i]
                ctr_2ds = bs_utils.project_p3d(ctr3d[None, :], cam_scale, K)
                rgb = bs_utils.draw_p2ds(
                    rgb, ctr_2ds, 4, (0, 0, 255)
                )
            imshow('{}_rgb'.format(cat), rgb)
            cmd = waitKey(0)
            if cmd == ord('q'):
                exit()
            else:
                continue


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
