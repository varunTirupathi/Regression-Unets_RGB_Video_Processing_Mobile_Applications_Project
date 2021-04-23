import argparse
import os
import cv2
import numpy as np
import pytorch_lightning as pl
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
        self.min_size = 32
        self.index_links = self.find_index_links()
        self.mask = mask
        mt_obj = globals()[mask]
        if mask == "SampleEntropy":
            self.mask_transform = mt_obj(self.min_size - 1)
        else:
            self.mask_transform = mt_obj()

    def find_index_links(self):
        videos = [x for x in os.listdir(self.video_path)]  # List all the videos at given path.

        # Split the data into 8:2.
        training_indices = int(len(videos) * 0.8)
        if self.train:
            videos = videos[:training_indices]
        else:
            videos = videos[training_indices:]

        self.videos_full_path = [
            os.path.join(self.video_path, x) for x in videos
        ]  # Find the full path.

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
        for ii in range(vid_i + 1):
            for i in range(self.min_size):
                _, frame_i = cap.read()
                if ii == vid_i:
                    frame.append(frame_i)

        frame = [self.img_transform(cv2pil_input(x)).unsqueeze(0) for x in frame]

        frame = torch.cat(frame, axis=0)        # [frames, channels, height, width]
        frame = frame.permute(1, 0, 2, 3)       # [channel, frames, height, width]
        frame = frame.permute(0, 2, 3, 1)       # [channel, height, width, frames]
        mask = self.mask_transform(frame)

        if self.mask == 'TemporalMedian':
            mask = mask.permute(3, 1, 2, 0)
        else:
            mask = mask.permute(1, 0, 2, 3)
        return frame, mask

def params():
    parser = argparse.ArgumentParser(description="UNet filter training")
    parser.add_argument("--seed", type=int, default=1234)

    # Training Settings.
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Specify learning rate of optimizer."
    )

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
        "--batch_size", default=1, type=int, metavar="N", help="mini-batch size (default: 32)"
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
        default=4,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )

    hparams = parser.parse_args()
    return hparams

class UNet3D(pl.LightningModule):
    def __init__(self):
        super(UNet3D, self).__init__()
        self.hparams = params()
        if self.hparams.mask == "TemporalGaussian" or self.hparams.mask == "TemporalMedian":
            self.model = UNet3d(in_channels=self.hparams.n_channels, n_labels=3)
        else:
            self.model = UNet3d2d(in_channels=self.hparams.n_channels, n_labels=1)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch
        predicted_mask = self(image)
        loss = torch.nn.functional.mse_loss(mask, predicted_mask)
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch
        mask = mask
        predicted_mask = self(image)
        loss = torch.nn.functional.mse_loss(mask, predicted_mask)
        PROJECT = str(os.environ.get("HOME")) + "/Regression-Unets-RGB-video-processing-PL/results/" #compute canada cluster path
        torch.save(self.model.state_dict(), str(PROJECT), 'weights.pth')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer

    def train_dataloader(self):
        trainset = Video3DDataset(
            video_path=self.hparams.video_path, train=True, mask=self.hparams.mask
        )
        return DataLoader(
            trainset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.workers,
        )

    def val_dataloader(self):
        valset = Video3DDataset(
            video_path=self.hparams.video_path, train=False, mask=self.hparams.mask
        )
        return DataLoader(
            valset,
            batch_size=self.hparams.test_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.workers,
        )

if __name__ == "__main__":
    model = UNet3D()
    trainer = pl.Trainer(max_epochs=5, gpus=0)
    trainer.fit(model)
