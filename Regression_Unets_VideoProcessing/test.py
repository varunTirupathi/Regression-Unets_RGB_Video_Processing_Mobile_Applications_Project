import argparse
import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy.ndimage import gaussian_filter1d
from torch.utils.data import DataLoader, Dataset

from unets.unet import UNet3d, UNet3d2d
from temporal_median import temporal_median_filter_multi2

# Only used for input image conversion (in VideoDataLoader)
def cv2pil_input(image):
    return Image.fromarray(image)

imsize = (128, 128)
img_transform = transforms.Compose(
    [transforms.Resize(imsize), transforms.ToTensor()]  # scale imported image
)

class VideoWriter:
    def __init__(self, filename, fps=30, size=imsize):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter(filename, fourcc, fps, size)

    def write(self, frame):
        if len(frame.shape)>3:
            for i in range(frame.shape[0]):
                frame_write = (frame[i])*255
                self.video.write(frame_write.astype(np.uint8))
        self.release()
        return True

    def release(self):
        self.video.release()

class SampleEntropy:
    def __init__(self, m=2):
        self.r = 100
        self.m = m

    @staticmethod
    def torch2ndarray(input):
        return input.numpy()

    @staticmethod
    def ndarray2torch(input):
        return torch.from_numpy(input).float()

    # Referred the function from https://en.wikipedia.org/wiki/Sample_entropy
    def sample_entropy(self, L):
        N = len(L)
        B = 0.0
        A = 0.0

        # Split time series and save all templates of length m
        xmi = np.array([L[i : i + self.m] for i in range(N - self.m)])
        xmj = np.array([L[i : i + self.m] for i in range(N - self.m + 1)])

        # Save all matches minus the self-match, compute B
        B = np.sum([np.sum(np.abs(xmii - xmj).max(axis=1) <= self.r) - 1 for xmii in xmi])

        # Similar for computing A
        self.m += 1
        xm = np.array([L[i : i + self.m] for i in range(N - self.m + 1)])
        A = np.sum([np.sum(np.abs(xmi - xm).max(axis=1) <= self.r) - 1 for xmi in xm])
        self.m -= 1

        # Return SampEn
        return -np.log(A / B)

    def __call__(self, image):
        image = image.permute(1, 0, 2, 3)
        image = self.torch2ndarray(image)
        _, _, height, width = image.shape
        output = np.zeros((height, width))

        # Apply SampleEntropy on each pixel on sequential data.
        for i in range(height):
            for j in range(width):
                output[i, j] = self.sample_entropy(image[:, :, i, j] * 255.0)
        return self.ndarray2torch(output)  # A class to apply filter.


class TemporalGaussian:
    def __init__(self, sigma=2):
        self.sigma = sigma

    @staticmethod
    def torch2ndarray(input):
        return input.numpy()

    @staticmethod
    def ndarray2torch(input):
        return torch.from_numpy(input).float()

    def __call__(self, image):
        image = image.permute(1, 0, 2, 3)
        image = self.torch2ndarray(image)
        frames, channels, width, height = image.shape
        output = np.zeros((frames, channels, width, height))

        for ww in range(width):
            for hh in range(height):
                for cc in range(channels):
                    output[:, cc, ww, hh] = gaussian_filter1d(output[:, cc, ww, hh], self.sigma)

        return self.ndarray2torch(output)


class TemporalMedian:
    def __init__(self):
        self.temporal_median = temporal_median_filter_multi2

    @staticmethod
    def torch2ndarray(input):
        return input.numpy()

    @staticmethod
    def ndarray2torch(input):
        return torch.from_numpy(input).float()

    def __call__(self, image):
        # Shape of input:                       # [channel, height, width, frames]
        image = image.permute(3, 1, 2, 0)       # [frames, height, width, channel]
        image = self.torch2ndarray(image)
        output = self.temporal_median(image)
        return self.ndarray2torch(output)      

class Video3DDataset(Dataset):
    def __init__(self, video_path, train=False, evaluate=False, mask="TemporalGaussian"):
        self.img_transform = img_transform
        self.train = train
        self.video_path = video_path
        self.eval = evaluate
        self.mask = mask

        self.min_size = 32
        self.index_links = self.find_index_links()
        mt_obj = globals()[mask]
        if mask == "SampleEntropy":
            self.mask_transform = mt_obj(self.min_size - 1)
        else:
            self.mask_transform = mt_obj()

    def find_index_links(self):
        videos = [x for x in os.listdir(self.video_path) if '_mask' not in x]  # List all the videos at given path.

        # Split the data into 8:2.
        training_indices = int(len(videos) * 0.8)
        if self.train:
            videos = videos[:training_indices]
        else:
            videos = videos[training_indices:]

        self.videos_full_path = [
            os.path.join(self.video_path, x) for x in videos
        ]  # Find the full path.


        if self.mask == 'TemporalMedian':
            # Find the full path for ground truth videos.
            self.videos_full_path_masks = [
                os.path.join(self.video_path, x.split('.')[0]+'_mask.avi') for x in videos
            ]

        index_links = []
        count = 0
        for vid, vfp in enumerate(self.videos_full_path):
            cap = cv2.VideoCapture(vfp)
            size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in range(int(size / self.min_size)):
                index_links.append([count, vid, i])
                count += 1
        return np.array(index_links)

    def __len__(self):
        return self.index_links.shape[0]

    def __getitem__(self, index):
        vid = self.index_links[index, 1]
        vid_i = self.index_links[index, 2]

        frame = []
        cap = cv2.VideoCapture(self.videos_full_path[vid])
        if self.mask == 'TemporalMedian':
            cap_mask = cv2.VideoCapture(self.videos_full_path_masks[vid])           # read ground truth data point from the video.
            masks = []

        for ii in range(vid_i + 1):
            for i in range(self.min_size):
                _, frame_i = cap.read()
                if self.mask == 'TemporalMedian':
                    _, mask_i = cap_mask.read()
                if ii == vid_i:
                    frame.append(frame_i)
                    if self.mask == 'TemporalMedian':
                        masks.append(mask_i)

        frame = [self.img_transform(cv2pil_input(x)).unsqueeze(0) for x in frame]

        frame = torch.cat(frame, axis=0)        # [frames, channels, height, width]
        frame = frame.permute(1, 0, 2, 3)       # [channel, frames, height, width]
        # frame = frame.permute(0, 2, 3, 1)		# [channel, height, width, frames]
        frame = frame.permute(0, 2, 3, 1)       # [channel, height, width, frames]

        if self.mask == 'TemporalMedian':
            masks = [self.img_transform(cv2pil_input(x)).unsqueeze(0) for x in masks]       # pre-process the ground truth data points.
            masks = torch.cat(masks, axis=0)        # [masks, channels, height, width]
            masks = masks.permute(1, 0, 2, 3)       # [channel, maskss, height, width]
            masks = masks.permute(0, 2, 3, 1)       # [channel, height, width, frames]
            return frame, masks
        else:
            mask = self.mask_transform(frame)
            mask = mask.permute(1, 0, 2, 3)
            return frame, mask

def post_process(data):
    return data[0].permute(3, 1, 2, 0).detach().cpu().numpy()

def save_results(idx, frame, mask, output, folder='results'):
    if not os.path.exists(folder): os.mkdir(folder)
    frame = post_process(frame)
    mask = post_process(mask)
    output = post_process(output)
    frame_name = f"{folder}/{idx}_frame.avi"
    mask_name = f"{folder}/{idx}_mask.avi"
    output_name = f"{folder}/{idx}_output.avi"
    video_writer = VideoWriter(frame_name)
    video_writer.write(frame)
    video_writer = VideoWriter(mask_name)
    video_writer.write(mask)
    video_writer = VideoWriter(output_name)
    video_writer.write(output)


def params():
    parser = argparse.ArgumentParser(description="UNet filter training")
    parser.add_argument("--seed", type=int, default=1234)

    # Network Settings.
    parser.add_argument("--n_channels", type=int, default=3)

    # Dataset Settings.
    parser.add_argument(
        "--video_path", default="dataset/videos", type=str, help="directory of videos."
    )
    parser.add_argument(
        "--mask",
        default="TemporalGaussian",
        type=str,
        choices=["TemporalGaussian", "SampleEntropy", "TemporalMedian"],
    )

    parser.add_argument(
        "--test_batch_size",
        default=1,
        type=int,
        metavar="N",
        help="test-mini-batch size (default: 8)",
    )
    parser.add_argument(
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument('--pretrained', default='D:/Regression-Unets-RGB-video-processing-PL/Regression-Unets-RGB-video-processing-PL/weights.pth', type=str,
                        metavar='PATH', help='path to pretrained model file (default: null (no-use))')

    hparams = parser.parse_args()
    return hparams


if __name__ == "__main__":
    hparams = params()
    if hparams.mask == "TemporalGaussian" or hparams.mask == "TemporalMedian":
        model = UNet3d(in_channels=hparams.n_channels, n_labels=3)
    else:
        model = UNet3d2d(in_channels=hparams.n_channels, n_labels=1)

    model.load_state_dict(torch.load(hparams.pretrained, map_location='cpu'))

    valset = Video3DDataset(
            video_path=hparams.video_path, train=False, mask=hparams.mask
        )
    val_dataloader = DataLoader(
            valset,
            batch_size=hparams.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=hparams.workers,
        )

    for idx, data in enumerate(val_dataloader):
        frame, mask = data
        output = model(frame)
        save_results(idx, frame, mask, output)
