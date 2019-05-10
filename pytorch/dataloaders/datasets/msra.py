import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
import torch
import sys
from random import randint
import struct
sys.path.append('/u/big/workspace_hodgkinsona/denseReg/pytorch')
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
from dataloaders.utils import CameraConfig
from collections import namedtuple

class MSRADataset(data.Dataset):
    # directory = '/u/big/trainingdata/MSRA/cvpr15_MSRAHandGestureDB/'
    def __init__(self, args, root=Path.db_root_dir('msra'), split="train"):
        self.cfg = CameraConfig(fx=241.42, fy=241.42, cx=160, cy=120, w=320, h=240)
        self.pose_list = '1 2 3 4 5 6 7 8 9 I IP L MP RP T TIP Y'.split()
        self.approximate_num_per_file = 85 
        self.max_depth = 1000.0
        self.pose_dim = 63 
        self.jnt_num = 21

        self.root = root
        self.split = split
        self.args = args
        # self.exclude = set(excludes)
        self.subjects = {
            'train': frozenset([0, 1, 2, 3, 4, 5, 6]),
            'val': frozenset([7]),
            'test': frozenset([8])
        }

        self.fnames = []

        for subject_num in self.subjects[split]:
            self.add_files(subject_num)

        # print(len(self.fnames))
        if len(self.fnames) == 0:
            raise Exception("No samples found for split=[{}] found in {}".format(split, self.root))
        print("Found {} {} samples in {}".format(len(self.fnames), split, self.root))

    def add_files(self, subject_num):
        for pose in self.pose_list:
            path = os.path.join(self.root, 'P{}'.format(subject_num), pose, 'joint.txt')
            with open(path, 'r') as f:
                n = int(f.readline().strip())
                for frm_idx in range(n):
                    base = os.path.split(path)[0]
                    self.fnames.append(os.path.join(base, "{:06d}_depth.bin".format(frm_idx)))


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        
        bin_path = self.fnames[index]
        base_path, bin_fname = os.path.split(bin_path)
        line_num = int(bin_fname[3:6])
        annotation_path = os.path.join(base_path, 'joint.txt')
        with open(annotation_path, 'r') as f:
            # first line of file just tells how many samples
            # need to offset so first sample isn't garbage
            annotation = f.readlines()[line_num+1].split()
            pose = np.empty(63, dtype=np.float32)
            for idx, d in enumerate(annotation):
                if idx % 3 == 0:
                    pose[idx] = float(d)
                elif idx % 3 == 1:
                    pose[idx] = -float(d)
                elif idx % 3 == 2:
                    pose[idx] = -float(d)            
            pose = pose.reshape((21, 3))

        img = self.convert_bin(bin_path)

        # config = np.array([self.cfg.fx/])
        center_of_mass = self.center_of_mass(img, config)
        sample = {'image': img,
                  'label': pose,
                  'center_of_mass': center_of_mass,
                  'config', config}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def convert_bin(self, bin_path):
        MSRA_size = namedtuple('MSRA_size', ['cols', 'rows', 'left', 'top', 'right', 'bottom'])
        with open(bin_path, 'rb') as f:
            shape = [struct.unpack('i', f.read(4))[0] for i in range(6)]
            shape = MSRA_size(*shape)
            crop_depth_map_data = np.fromfile(f, dtype=np.float32)
        
        crop_rows, crop_cols = shape.bottom - shape.top, shape.right - shape.left
        crop_depth_map_data = crop_depth_map_data.reshape(crop_rows, crop_cols)

        depth_map_data = np.zeros((shape.rows, shape.cols), np.float32)
        np.copyto(depth_map_data[shape.top : shape.bottom, shape.left : shape.right], crop_depth_map_data)
        return depth_map_data

    def center_of_mass(self, depth_map, config):
        c_h, c_w = depth_map.size()
        avg_u, avg_v = c_w / 2.0, c_h / 2.0
        avg_d = depth_map[depth_map > 0].mean()
        avg_d = np.max(avg_d, 200.0)
        
        avg_x = (avg_u-config[2])*avg_d/config[0]
        avg_y = (avg_v-config[3])*avg_d/config[1]
        avg_xyz = np.stack([avg_x, avg_y, avg_d], axis=0)
        return avg_xyz
        
        
    def transform_tr(self, sample):
        # composed_transforms = transforms.Compose([
        #     tr.RandomHorizontalFlip(),
        #     tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
        #     tr.RandomGaussianBlur(),
        #     tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        #     tr.ToTensor()])
        composed_transforms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation((-90, 90)),
            transforms.RandomResizedCrop(),

        ])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # args.base_size = 513
    # args.crop_size = 513
    for i in range(9):
        msra_train = MSRADataset(args, split='train')
        msra_test = MSRADataset(args, split='test')
        
        for j in range(500):
            samples = msra_train[j]
            img, pose = samples['image'], samples['label']

            fig = plt.figure()
            ax = fig.add_subplot(111)
            plt.imshow(img)
            plt.show()
    