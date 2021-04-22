import os
import torch
import argparse
import numpy as np
from model import UNet
from trainer import Trainer
from dataloader import VideoDataset
from torch.utils.data import DataLoader

# Create dataloader
def get_dataloader(hparams):
	testset = VideoDataset(video_path=hparams.video_path, mask_filter=hparams.mask_filter)
	test_loader = DataLoader(testset, batch_size=hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers)
	return testset, test_loader

def params():
	parser = argparse.ArgumentParser(description='UNet filter training')
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--eval', action='store_false', help='Train or Evaluate the network.')
	parser.add_argument('--store_results', action='store_true', help='Store evaluated images.')
	parser.add_argument('--store_video', action='store_false', help='Store results as a video')
	
	# Network Settings.
	parser.add_argument('--n_channels', type=int, default=3)
	parser.add_argument('--n_classes', type=int, default=3)

	# Dataset Settings.
	parser.add_argument('--video_path', default='output.avi', type=str,
						help='directory of training images')
	parser.add_argument('--mask_filter', default='CannyEdgeDetection', type=str, help='Choose mask filter', 
						choices=['CannyEdgeDetection', 'GaussianBlur', 'Emboss', 'Blur', 'Sharpen'])
	parser.add_argument('--test_batch_size', default=1, type=int,
						metavar='N', help='test-mini-batch size (default: 8)')
	parser.add_argument('--workers', default=0, type=int,
						metavar='N', help='number of data loading workers (default: 4)')

	# Use of pretrained models.
	parser.add_argument('--resume', default='', type=str,
						metavar='PATH', help='path to latest checkpoint (default: null (no-use))')
	parser.add_argument('--pretrained', default='', type=str,
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
	testset, test_loader = get_dataloader(hparams)
	hparams.fps = testset.fps

	# Create UNet model.
	model = UNet(hparams)

	if hparams.eval:
		trainer = Trainer(model, hparams)		# Create trainer instance.
		trainer.test(test_loader)				# Test a pretrained network for only a single epoch.