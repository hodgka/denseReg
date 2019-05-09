import cv2
import numpy as np

import torch
from torch.nn import functional as F
from torchvision import transforms
from modeling.deeplab import *
from dataloaders import *
import dataloaders.utils as d_utils
import argparse
from skimage.transform import resize
import time

parser = argparse.ArgumentParser()
parser.add_argument('model')
parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
parser.add_argument('--out-stride', type=int, default=8,
                    help='network output stride (default: 8)')
parser.add_argument('--base-size', type=int, default=128,
                        help='base image size')
parser.add_argument('--crop-size', type=int, default=128,
                    help='crop image size')
parser.add_argument('--sync-bn', type=bool, default=None,
                    help='whether to use sync bn (default: auto)')
parser.add_argument('--freeze-bn', type=bool, default=False,
                    help='whether to freeze bn parameters (default: False)')
parser.add_argument('--loss-type', type=str, default='ce',
                    choices=['ce', 'focal'],
                    help='loss func type (default: ce)')
parser.add_argument('--dataset', type=str, default='pascal',
                        choices=['pascal', 'coco', 'cityscapes', 'doorknobs'],
                        help='dataset name (default: pascal)')
parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
parser.add_argument('--gpu-ids', type=str, default='0',
                    help='use which gpu to train, must be a \
                    comma-separated list of integers only (default=0)')
args = parser.parse_args()
args.size = 240
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

# setup
cap = cv2.VideoCapture(0)
xres = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
yres = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


MEAN=[0.3535316924463259, 0.36619692064482556, 0.3704869546679205]
STD=[0.1689161790400891, 0.17802213630682928, 0.17895873909382565]
composed_transforms = transforms.Compose([
            # tr.DemoFixScaleCrop(crop_size=args.crop_size),
            tr.DemoNormalize(mean=MEAN, std=STD),
            tr.DemoToTensor(),
            ])

class NewDeepLab(DeepLab):
    def forward(self, x):
        x, low_level_feat = self.backbone(x)
        x = self.aspp(x)
        return x
# initialize model
model = NewDeepLab(num_classes=13,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn).cuda()


# # loading logic                        
checkpoint = torch.load(args.model, map_location=lambda storage, loc: storage)
args.start_epoch = checkpoint['epoch']
if args.cuda:
    model.load_state_dict(checkpoint['state_dict'])    
else:
    model.load_state_dict(checkpoint['state_dict'])
best_pred = checkpoint['best_pred']
print("=> loaded checkpoint '{}' (epoch {})".format(args.model, checkpoint['epoch']))
model = model.eval()
while True:
    t = time.time()
    ret, frame = cap.read()
    if not ret: break
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = composed_transforms(img).unsqueeze(0).cuda()
    with torch.no_grad():
        pred = model(img).squeeze()
        # decoded = d_utils.decode_segmap(torch.max(pred, 0)[1].cpu().numpy(), dataset='doorknobs')
        # decoded = (255 * decoded).astype(np.uint8)
    # print(decoded.shape)
    # output = frame
    # decoded = resize(decoded, (480,480), preserve_range=True)
    # output = np.hstack([frame, decoded])
    # cv2.imshow('frame', output)
    # cv2.waitKey(1)
    print("FPS: ", time.time() - t)

cv2.release()
cv2.destroyAllWindows()