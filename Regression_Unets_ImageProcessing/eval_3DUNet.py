import os
import torch
import argparse
import numpy as np
from dataloader import ImageDataset, VideoDataset, Video3DDataset
from torch.utils.data import DataLoader
import cv2
import matplotlib.pyplot as plt

def get_3d_dataloader(hparams):
    trainset = Video3DDataset(video_path=hparams.video_path, train=True)
    data_loader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=False, drop_last=False,
                             num_workers=hparams.workers)
    return data_loader


def params():
    parser = argparse.ArgumentParser(description='UNet filter training')
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--eval', action='store_true', help='Train or Evaluate the network.')
    parser.add_argument('--store_results', action='store_false', help='Store evaluated images.')
    parser.add_argument('--store_video', action='store_true', help='Store results as a video')
    parser.add_argument('--use_3d_data', action='store_false', help='Use 3D Video data')
    parser.add_argument('--use_video_data', action='store_true', help='Choose this option to train with video_data')

    # Training Settings.
    parser.add_argument('--exp_name', default='exp_unet')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=0.001, help='Specify learning rate of optimizer.')
    parser.add_argument('--epochs', default=2, type=int, help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, help='manual epoch number (useful on restarts)')

    # Network Settings.
    parser.add_argument('--n_channels', type=int, default=3)
    parser.add_argument('--n_classes', type=int, default=3)

    # Dataset Settings.
    parser.add_argument('--img_dir', default='dataset/images', type=str,
                        help='directory of training images')
    parser.add_argument('--mask_dir', default='./dataset/contour/train_masks/', type=str,
                        help='directory of training labels')
    parser.add_argument('--video_path', default='dataset/videos', type=str, help='directory of videos.')
    parser.add_argument('--mask_filter', default='CannyEdgeDetection', type=str, help='Choose mask filter',
                        choices=['CannyEdgeDetection', 'GaussianBlur', 'Emboss', 'Blur', 'Sharpen'])
    parser.add_argument('--batch_size', default=1, type=int,
                        metavar='N', help='mini-batch size (default: 32)')
    parser.add_argument('--test_batch_size', default=1, type=int,
                        metavar='N', help='test-mini-batch size (default: 8)')
    parser.add_argument('--workers', default=0, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')

    # Use of pretrained models.
    parser.add_argument('--resume', default='', type=str,
                        metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
    parser.add_argument('--pretrained',
                        default='D:/Regression-Unets-RGB-video-processing-/Regression-Unets-RGB-video-processing-/checkpoints/exp_unet/models/best_model.t7',
                        type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')

    hparams = parser.parse_args()
    return hparams


if __name__ == '__main__':
    # Get the hyper-parameters.
    hparams = params()
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(hparams.seed)
    torch.cuda.manual_seed_all(hparams.seed)
    np.random.seed(hparams.seed)

    # Get the dataloaders.
    data_loader = get_3d_dataloader(hparams)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Create UNet model.
    from unets.unet import UNet3d2d

    model = UNet3d2d(in_channels=hparams.n_channels, n_labels=1)
    model.eval()
    model.to(device)
    model.load_state_dict(torch.load(hparams.pretrained, map_location='cpu'))

for idx, data in enumerate(data_loader):
    image, mask = data
    image = image.to(device)
    mask = mask.to(device)
    predicted_mask = model(image)
    image = image[0]
    image = image.permute(0, 3, 1, 2)
    image = image.permute(1, 0, 2, 3)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    size = (320, 240)
    PROJECT = 'D:/Regression-Unets-RGB-video-processing-/Regression-Unets-RGB-video-processing-/checkpoints/exp_unet/results/'
    filename = str(PROJECT) + 'input_{}.avi'.format(idx)
    cap = cv2.VideoWriter(filename, fourcc, fps, size)
    for im in image:
        if im.shape[0] == 3:
            im = im.permute(1, 2, 0)
            im = im.detach().cpu().numpy() * 255.0
            cap.write(im.astype(np.uint8))

    if mask.shape[0] == 1 or mask.shape[0] == 3:
        mask = mask[0].permute(1, 2, 0)
    mask = mask.detach().cpu().numpy()

    if predicted_mask.shape[0] == 1 or predicted_mask.shape[0] == 3:
        predicted_mask = predicted_mask[0].permute(1, 2, 0)
    predicted_mask = predicted_mask.detach().cpu().numpy()

    mask = mask[0][0]
    mask = np.dstack([mask, mask, mask])
    plt.imsave(str(PROJECT) + 'gt_{}.png'.format(idx), mask)

    predicted_mask = predicted_mask[0][0]
    predicted_mask = np.dstack([predicted_mask, predicted_mask, predicted_mask])
    plt.imsave(str(PROJECT) + 'prediction_{}.png'.format(idx),predicted_mask)
