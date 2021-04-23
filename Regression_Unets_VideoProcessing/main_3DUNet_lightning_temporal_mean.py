import argparse
import os
import cv2
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from unets.unet import UNet3d, UNet3d2d

def cv2pil_input(image):
    # numpy array to PIL Image (required in torch dataloader)
    return Image.fromarray(image)

imsize = (128, 128)

# Resize the image and convert it to a tensor.
img_transform = transforms.Compose(
    [transforms.Resize(imsize), transforms.ToTensor()]
)

class Video3DDataset(Dataset):
    def __init__(self, video_path, train=False, evaluate=False):
        # video_path:       path of the folder containing videos.

        self.img_transform = img_transform
        self.train = train
        self.video_path = video_path
        self.eval = evaluate
        self.min_size = 32
        self.index_links = self.find_index_links()

    # Function to find number of data points in the dataset. Each data point has shape [3, 128, 128, 32]
    def find_index_links(self):
        videos = [x for x in os.listdir(self.video_path) if '_mask' not in x]  # List all the videos at given path.

        # Split the data into 8:2 ratio for training and testing.
        training_indices = int(len(videos) * 0.8)
        if self.train:
            videos = videos[:training_indices]
        else:
            videos = videos[training_indices:]

        # Find the full path for input videos.
        self.videos_full_path = [
            os.path.join(self.video_path, x) for x in videos
        ]
        # Find the full path for ground truth videos.
        self.videos_full_path_masks = [
            os.path.join(self.video_path, x.split('.')[0]+'_mask.avi') for x in videos
        ]

        index_links = []                                        # stores info to retrieve data points.
        count = 0
        for vid, vfp in enumerate(self.videos_full_path):
            cap = cv2.VideoCapture(vfp)
            size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))       # total frames in a video.
            for i in range(int(size / self.min_size)):          # divide videos in data points [3,128,128,32]
                index_links.append([count, vid, i])
                count += 1
        return np.array(index_links)

    def __len__(self):
        return self.index_links.shape[0]            # len of the dataset.

    def __getitem__(self, index):
        vid = self.index_links[index, 1]
        vid_i = self.index_links[index, 2]
        frame = []
        masks = []
        cap = cv2.VideoCapture(self.videos_full_path[vid])                      # read input data point from the video.
        cap_mask = cv2.VideoCapture(self.videos_full_path_masks[vid])           # read ground truth data point from the video.

        for ii in range(vid_i + 1):
            for i in range(self.min_size):
                _, frame_i = cap.read()
                _, mask_i = cap_mask.read()
                if ii == vid_i:
                    frame.append(frame_i)
                    masks.append(mask_i)

        frame = [self.img_transform(cv2pil_input(x)).unsqueeze(0) for x in frame]       # pre-process the input data points.
        frame = torch.cat(frame, axis=0)        # [frames, channels, height, width]
        frame = frame.permute(1, 0, 2, 3)       # [channel, frames, height, width]
        frame = frame.permute(0, 2, 3, 1)       # [channel, height, width, frames]     
        masks = [self.img_transform(cv2pil_input(x)).unsqueeze(0) for x in masks]       # pre-process the ground truth data points.
        masks = torch.cat(masks, axis=0)        # [masks, channels, height, width]
        masks = masks.permute(1, 0, 2, 3)       # [channel, maskss, height, width]
        masks = masks.permute(0, 2, 3, 1)       # [channel, height, width, frames]
        return frame, masks

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
        "--batch_size", default=4, type=int, metavar="N", help="mini-batch size (default: 4)"
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
        self.hparams = params()                                                 # create parameters for training.
        self.model = UNet3d(in_channels=self.hparams.n_channels, n_labels=3)    # create UNet network.

    # forward pass of neural network.
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, mask = batch                                                 # find the data.
        predicted_mask = self(image)                                        # compute the prediction.
        loss = torch.nn.functional.mse_loss(mask, predicted_mask)           # calculate Mean Squared Error b/w prediction and ground truth.
        return loss

    def validation_step(self, batch, batch_idx):
        image, mask = batch                                                 # find the data.
        predicted_mask = self(image)                                        # compute the prediction.
        loss = torch.nn.functional.mse_loss(mask, predicted_mask)           # calculate MSE b/w prediction and ground truth.
        PROJECT = str(os.environ.get("HOME")) + "/Regression-Unets-RGB-video-processing-PL/results/" #compute canada cluster path
        torch.save(self.model.state_dict(), str(PROJECT), 'weights.pth')
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)     # define optimizer.
        return optimizer

    # Define training dataset and dataloader.
    def train_dataloader(self):
        trainset = Video3DDataset(
            video_path=self.hparams.video_path, train=True
        )
        return DataLoader(
            trainset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.workers,
        )

    # Define testing dataset and dataloader.
    def val_dataloader(self):
        valset = Video3DDataset(
            video_path=self.hparams.video_path, train=False
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
