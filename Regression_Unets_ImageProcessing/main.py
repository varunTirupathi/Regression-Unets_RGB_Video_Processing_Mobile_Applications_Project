import os
import torch
import argparse
import numpy as np
from model import UNet
from trainer import Trainer
from dataloader import ImageDataset, VideoDataset, Video3DDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

# Just to take the backup of files used while training. (like a version history)
def _init_(hparams):
	if not os.path.exists('checkpoints'):
		os.makedirs('checkpoints')
	if not os.path.exists('checkpoints/' + hparams.exp_name):
		os.makedirs('checkpoints/' + hparams.exp_name)
	if not os.path.exists('checkpoints/' + hparams.exp_name + '/' + 'models'):
		os.makedirs('checkpoints/' + hparams.exp_name + '/' + 'models')
	os.system('cp main.py checkpoints' + '/' + hparams.exp_name + '/' + 'main.py.backup')
	os.system('cp trainer.py checkpoints' + '/' + hparams.exp_name + '/' + 'trainer.py.backup')
	os.system('cp model.py checkpoints' + '/' + hparams.exp_name + '/' + 'model.py.backup')
	os.system('cp dataloader.py checkpoints' + '/' + hparams.exp_name + '/' + 'dataloader.py.backup')

# Create a text file to store loss of each epoch.
class IOStream:
	def __init__(self, path):
		self.f = open(path, 'a')

	def cprint(self, text):
		print(text)
		self.f.write(text + '\n')
		self.f.flush()

	def close(self):
		self.f.close()

def get_video_dataloader(hparams):
	if not hparams.eval:
		trainset = VideoDataset(video_path=hparams.video_path, train=True, mask_filter=hparams.mask_filter)
		train_loader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers)
		testset = VideoDataset(video_path=hparams.video_path, train=False, mask_filter=hparams.mask_filter)
		test_loader = DataLoader(testset, batch_size=hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers)
		return train_loader, test_loader
	else:
		testset = VideoDataset(video_path=hparams.video_path, train=False, mask_filter=hparams.mask_filter, eval=hparams.eval)
		test_loader = DataLoader(testset, batch_size=hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers, eval=hparams.eval)		
		return None, test_loader

def get_3d_dataloader(hparams):
	trainset = Video3DDataset(video_path=hparams.video_path, train=True)
	train_loader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers)
	testset = Video3DDataset(video_path=hparams.video_path, train=False)
	test_loader = DataLoader(testset, batch_size=hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers)
	return train_loader, test_loader

# Create dataloader
def get_dataloader(hparams):
	testset = ImageDataset(img_dir=hparams.img_dir, mask_dir=hparams.mask_dir, train=False, mask_filter=hparams.mask_filter)
	test_loader = DataLoader(testset, batch_size=hparams.test_batch_size, shuffle=False, drop_last=False, num_workers=hparams.workers)

	# Create train loader only during training.
	if not hparams.eval:
		trainset = ImageDataset(img_dir=hparams.img_dir, mask_dir=hparams.mask_dir, train=True, mask_filter=hparams.mask_filter)
		train_loader = DataLoader(trainset, batch_size=hparams.batch_size, shuffle=True, drop_last=True, num_workers=hparams.workers)
		return train_loader, test_loader
	else:
		return None, test_loader

def params():
	parser = argparse.ArgumentParser(description='UNet filter training')
	parser.add_argument('--seed', type=int, default=1234)
	parser.add_argument('--eval', action='store_true', help='Train or Evaluate the network.')
	parser.add_argument('--store_results', action='store_false', help='Store evaluated images.')
	parser.add_argument('--store_video', action='store_true', help='Store results as a video')
	parser.add_argument('--use_3d_data', action='store_true', help='Use 3D Video data')
	parser.add_argument('--use_video_data', action='store_true', help='Choose this option to train with video_data')

	# Training Settings.
	parser.add_argument('--exp_name', default='exp_unet')
	parser.add_argument('--optimizer', type=str, default='Adam')
	parser.add_argument('--lr', type=float, default=0.001, help='Specify learning rate of optimizer.')
	parser.add_argument('--epochs', default=50, type=int, help='number of total epochs to run')
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
	parser.add_argument('--batch_size', default=8, type=int,
						metavar='N', help='mini-batch size (default: 32)')
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
	if hparams.use_video_data:
		train_loader, test_loader = get_video_dataloader(hparams)
	elif hparams.use_3d_data:
		train_loader, test_loader = get_3d_dataloader(hparams)
	else:
		train_loader, test_loader = get_dataloader(hparams)

	# Create UNet model.
	if hparams.use_3d_data:
		from unets.unet import UNet3d2d
		model = UNet3d2d(in_channels=hparams.n_channels, n_labels=1)
	else:
		model = UNet(hparams)

	if hparams.eval:
		trainer = Trainer(model, hparams)		# Create trainer instance.
		trainer.test(test_loader)				# Test a pretrained network for only a single epoch.
	else:
		# Create log to store training data.
		tensorboard_logger = SummaryWriter(log_dir='checkpoints/' + hparams.exp_name)
		_init_(hparams)
		text_logger = IOStream('checkpoints/' + hparams.exp_name + '/run.log')
		text_logger.cprint(str(hparams))

		# Create trainer instance.
		trainer = Trainer(model, hparams, tensorboard_logger, text_logger)
		# Start the training.
		trainer.train(train_loader, test_loader)