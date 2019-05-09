import os
import torch
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
# from dataloaders.utils import decode_seg_map_sequence

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    # def visualize_image(self, writer, dataset, image, target, output, global_step, **kwargs):
    #     grid_image1 = make_grid(image[:3].clone().cpu().data, 3, normalize=True)
    #     writer.add_image('Image', grid_image1, global_step)
    #     grid_image2 = make_grid(decode_seg_map_sequence(torch.max(output[:3], 1)[1].detach().cpu().numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image('Predicted label', grid_image2, global_step)
    #     grid_image3 = make_grid(decode_seg_map_sequence(torch.squeeze(target[:3], 1).detach().cpu().numpy(),
    #                                                    dataset=dataset), 3, normalize=False, range=(0, 255))
    #     writer.add_image('Groundtruth label', grid_image3, global_step)
    #     grid_diff = grid_image2 - grid_image3
    #     writer.add_image("Label Diff", grid_diff, global_step)