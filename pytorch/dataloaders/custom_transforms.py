import torch
import random
import numpy as np
import math

from PIL import Image, ImageOps, ImageFilter
from skimage.transform import resize


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32)
        mask = np.array(mask).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        img = sample['image']
        mask = sample['label']
        img = np.array(img).astype(np.float32).transpose((2, 0, 1))
        mask = np.array(mask).astype(np.float32)

        img = torch.from_numpy(img).float()
        mask = torch.from_numpy(mask).float()

        return {'image': img,
                'label': mask}


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return {'image': img,
                'label': mask}


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        rotate_degree = random.uniform(-1*self.degree, self.degree)
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return {'image': img,
                'label': mask}


class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))

        return {'image': img,
                'label': mask}


class RandomScaleCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4, 4. / 3), interpolation=Image.BILINEAR):
        if isinstance(size, tuple):
            self.size = size
        else:
            self.size = (size, size)
        # self.base_size = base_size
        # self.crop_size = crop_size
        # self.fill = fill
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio


    def __call__(self, sample):
        img = sample['image']
        label = sample['label']

        m = torch.distributions.normal.Normal(torch.tensor([1.0]), torch.tensor([0.2]))
        edge_ratio = torch.clamp(m.sample(torch.Size([2])), 0.9, 1.1)
        height, width = img.size()
        target_height = height * edge_ratio[0]
        target_width = width * edge_ratio[1]
        img = img.resize((target_height, target_width), self.interpolation)
        label = label * torch.Tensor([edge_ratio[1], edge_ratio[0], 1.0]).repeat(label.size(0))

        # random scale (short edge)
        # short_size = random.randint(int(self.base_size * 0.5), int(self.base_size * 2.0))
        # w, h = img.size
        # if h > w:
        #     ow = short_size
        #     oh = int(1.0 * h * ow / w)
        # else:
        #     oh = short_size
        #     ow = int(1.0 * w * oh / h)
        # img = img.resize((ow, oh), Image.BILINEAR)
        # mask = mask.resize((ow, oh), Image.NEAREST)
        # # pad crop
        # if short_size < self.crop_size:
        #     padh = self.crop_size - oh if oh < self.crop_size else 0
        #     padw = self.crop_size - ow if ow < self.crop_size else 0
        #     img = ImageOps.expand(img, border=(0, 0, padw, padh), fill=0)
        #     mask = ImageOps.expand(mask, border=(0, 0, padw, padh), fill=self.fill)
        # # random crop crop_size
        # w, h = img.size
        # x1 = random.randint(0, w - self.crop_size)
        # y1 = random.randint(0, h - self.crop_size)
        # img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        # mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))


        return {'image': img,
                'label': mask}



class FixScaleCrop(object):
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = img.size
        if w > h:
            oh = self.crop_size
            ow = int(1.0 * w * oh / h)
        else:
            ow = self.crop_size
            oh = int(1.0 * h * ow / w)
        img = img.resize((ow, oh), Image.BILINEAR)
        mask = mask.resize((ow, oh), Image.NEAREST)
        # center crop
        w, h = img.size
        x1 = int(round((w - self.crop_size) / 2.))
        y1 = int(round((h - self.crop_size) / 2.))
        img = img.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))
        mask = mask.crop((x1, y1, x1 + self.crop_size, y1 + self.crop_size))

        return {'image': img,
                'label': mask}


class FixedResize(object):
    def __init__(self, size):
        self.size = (size, size)  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        assert img.size == mask.size

        img = img.resize(self.size, Image.BILINEAR)
        mask = mask.resize(self.size, Image.NEAREST)

        return {'image': img,
                'label': mask}

class transform:
    def __init__(self, degree=90):
        self.degree = degree

    def __call__(self, sample):
        img = sample['image']
        pose = sample['label']
        config = sample['config']
        center_of_mass = sample['center_of_mass']

        angle = random.uniform(-1 * self.degree, self.degree)
        rot_image = img.rotate(angle, Image.BILINEAR)

        uv_center_of_mass = xyz2uvd(center_of_mass, config)
        uvd_pose = xyz2uvd(pose, config) - np.tile(uv_center_of_mass, pose.shape[0] // 3)
        cos, sin = np.cos(angle), np.sin(angle)
        rot_mat = np.array([[cos, -sin, 0], [sin, cos, 0], [0, 0, 1]])
        uvd_pose = uvd_pose.reshape((-1, 3))

        rot_pose = np.matmul(uvd_pose, rot_mat).reshape((-1,))




class XYZ2UVD:
    def __init__(self):
        pass

    def __call__(self, sample):
        img = sample['image']
        label = sample['label']


def xyz2uvd_op(xyz_pts, cfg):
    '''xyz_pts: tensor of xyz points
       camera_cfg: constant tensor of camera configuration
    '''
    xyz_pts = tf.reshape(xyz_pts, (-1,3))
    xyz_list = tf.unstack(xyz_pts)
    uvd_list = [_pro(pt, cfg) for pt in xyz_list]
    uvd_pts = tf.stack(uvd_list)
    return tf.reshape(uvd_pts, shape=(-1,))
