import os
import numpy as np
import scipy.misc as m
from PIL import Image
from torch.utils import data
import sys
sys.path.append('/u/big/workspace_hodgkinsona/denseReg/pytorch')
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
# from basedataset import CameraConfig
from collections import namedtuple

CameraConfig = namedtuple('CameraConfig', 'fx,fy,cx,cy,w,h')
Annotation = namedtuple('Annotation', 'name,pose')

class ICVLDataset(data.Dataset):
    cfg = CameraConfig(fx=241.42, fy=241.42, cx=160, cy=120, w=320, h=240)
    approximate_num_per_file = 85 
    max_depth = 1000.0
    pose_dim = 63 
    jnt_num = 21
    pose_list = '1 2 3 4 5 6 7 8 9 I IP L MP RP T TIP Y'.split()
    # directory = '/u/big/trainingdata/MSRA/cvpr15_MSRAHandGestureDB/'
    def __init__(self, args, root=Path.db_root_dir('icvl'), split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.files = {i: {pose: [] for pose in self.pose_list} for i in range(9)}
        import pprint as pp
        pp.pprint(self.files)
        # for i in range(9):
        #     files = 
        # # self.images_base = os.path.join(self.root, "P{}".format(pid), self.split)
        # self.annotations_base = os.path.join(self.root, 'gtFine', self.split)

        # self.files[split] = self.recursive_glob(rootdir=self.images_base, suffix='.png')

        # if not self.files[split]:
        #     raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))

        # print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):

        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(self.annotations_base,
                                img_path.split(os.sep)[-2],
                                os.path.basename(img_path)[:-15] + 'gtFine_labelIds.png')

        _img = Image.open(img_path).convert('RGB')
        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
        _tmp = self.encode_segmap(_tmp)
        _target = Image.fromarray(_tmp)

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            return self.transform_tr(sample)
        elif self.split == 'val':
            return self.transform_val(sample)
        elif self.split == 'test':
            return self.transform_ts(sample)

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask

    def recursive_glob(self, rootdir='.', suffix=''):
        """Performs recursive glob with given suffix and rootdir
            :param rootdir is the root directory
            :param suffix is the suffix to be searched
        """
        return [os.path.join(looproot, filename)
                for looproot, _, filenames in os.walk(rootdir)
                for filename in filenames if filename.endswith(suffix)]

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

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
    from dataloaders.utils import decode_segmap
    from torch.utils.data import DataLoader
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    
    # args.base_size = 513
    # args.crop_size = 513
    msra_train = MSRADataset(args, split='train')
    
    # cityscapes_train = CityscapesSegmentation(args, split='train')

    # dataloader = DataLoader(cityscapes_train, batch_size=2, shuffle=True, num_workers=2)

    # for ii, sample in enumerate(dataloader):
    #     for jj in range(sample["image"].size()[0]):
    #         img = sample['image'].numpy()
    #         gt = sample['label'].numpy()
    #         tmp = np.array(gt[jj]).astype(np.uint8)
    #         segmap = decode_segmap(tmp, dataset='cityscapes')
    #         img_tmp = np.transpose(img[jj], axes=[1, 2, 0])
    #         img_tmp *= (0.229, 0.224, 0.225)
    #         img_tmp += (0.485, 0.456, 0.406)
    #         img_tmp *= 255.0
    #         img_tmp = img_tmp.astype(np.uint8)
    #         plt.figure()
    #         plt.title('display')
    #         plt.subplot(211)
    #         plt.imshow(img_tmp)
    #         plt.subplot(212)
    #         plt.imshow(segmap)

    #     if ii == 1:
    #         break

    # plt.show(block=True)

