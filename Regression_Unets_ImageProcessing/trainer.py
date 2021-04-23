import os
import torch
import numpy as np
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2

class VideoWriter:
	def __init__(self, filename, fps, size):
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		self.video = cv2.VideoWriter(filename, fourcc, fps, size)

	def write(self, frame):
		if len(frame.shape)>3:
			for i in range(frame.shape[0]):
				frame_write = (frame[i].permute(1, 2, 0).detach().cpu().numpy())*255
				self.video.write(frame_write.astype(np.uint8))
		else:
			frame_write = (frame.permute(1, 2, 0).detach().cpu().numpy())*255
			self.video.write(frame_write.astype(np.uint8))
		return True

	def release(self):
		self.video.release()

# Class for the training of UNet network.
class Trainer:
	def __init__(self, model, hparams, tensorboard_logger=None, text_logger=None):
		self.hparams = hparams
		self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		self.model = model.to(self.device)
		self.tensorboard_logger = tensorboard_logger
		self.text_logger = text_logger
		self.is_training = True

		# Will be used only if we restart the training from a certain pretrained model.
		if self.hparams.resume:
			assert os.path.isfile(self.hparams.resume)
			self.checkpoint = torch.load(self.hparams.resume)
			self.hparams.start_epoch = checkpoint['epoch']
			self.model.load_state_dict(checkpoint['model'])
		else:
			self.checkpoint = None

		# Load the pretrained model for evaluation.
		if self.hparams.pretrained:
			assert os.path.isfile(self.hparams.pretrained)
			model.load_state_dict(torch.load(self.hparams.pretrained, map_location='cpu'))		

	# Define an optimizer for training. (only called during training).
	def create_optimizer(self):
		learnable_params = filter(lambda p: p.requires_grad, self.model.parameters())
		if self.hparams.optimizer == 'Adam':
			self.optimizer = torch.optim.Adam(learnable_params, lr=self.hparams.lr)
		else:
			self.optimizer = torch.optim.SGD(learnable_params, lr=self.hparams.lr)

		if self.checkpoint is not None:
			self.optimizer.load_state_dict(self.checkpoint['optimizer'])

	# Here we can define various loss functions as per the requirement.
	@staticmethod
	def calculate_loss(gt_mask, predicted_mask):
		return torch.nn.functional.mse_loss(predicted_mask, gt_mask)

	# Training for each episode takes place.
	def train_one_epoch(self, train_loader):
		self.model.train()			# Just to ensure that all tensors have right gradient connections.
		train_loss = 0.0
		count = 0
		for i, data in enumerate(tqdm(train_loader)):
			image, mask = data 					# Get the inputs, masks from the dataset.

			# Transfer the tensor to either cpu or gpu as per the available device.
			image = image.to(self.device)
			mask = mask.to(self.device)

			# Run the UNet and get a prediction.
			predicted_mask = self.model(image)
			loss = self.calculate_loss(mask, predicted_mask)

			# Do backpropagation.
			self.optimizer.zero_grad()
			loss.backward()
			self.optimizer.step()

			train_loss += loss.item()
			count += 1

		train_loss = float(train_loss)/count
		return train_loss

	# Stores the input image and prediction of network as a result with a spacing between them.
	def store_results(self, image, predicted_mask, batch_idx):
		gap = torch.zeros(image.shape[0], image.shape[1], image.shape[2], 20).to(image.device)
		results = torch.cat([image, gap, predicted_mask], axis=3).permute(0, 2, 3, 1)
		for idx in range(self.hparams.test_batch_size):
			result_path = os.path.join(self.result_dir, '%0.5d.jpg'%(batch_idx*self.hparams.test_batch_size + idx))
			plt.imsave(result_path, results[idx].detach().cpu().numpy())

	# Stores the ground truth mask and prediction of network as a result with a spacing between them.
	def store_comparisons(self, mask, predicted_mask, batch_idx):
		gap = torch.zeros(mask.shape[0], mask.shape[1], mask.shape[2], 20).to(mask.device)
		results = torch.cat([mask, gap, predicted_mask], axis=3).permute(0, 2, 3, 1)
		for idx in range(self.hparams.test_batch_size):
			result_path = os.path.join(self.result_dir, '%0.5d_comparison.jpg'%(batch_idx*self.hparams.test_batch_size + idx))
			plt.imsave(result_path, results[idx].detach().cpu().numpy())

	# Test the network for each episode.
	def test_one_epoch(self, test_loader):
		self.model.eval()			# Very useful in case of batch-norm and ensures that network is in evaluation mode.
		test_loss = 0.0
		count = 0

		if self.is_training:
			self.result_dir = 'checkpoints/{}/results/{}'.format(self.hparams.exp_name, self.epoch)
		else:
			self.result_dir = 'log/eval/'

		# Creates a director to store the results.
		if self.hparams.store_results and self.epoch%20 == 0:
			if not os.path.exists(self.result_dir):
				os.makedirs(self.result_dir)

		if self.hparams.store_video:
			if not os.path.exists(self.result_dir):
				os.makedirs(self.result_dir)
			
			video_size = (320, 240)
			image_video = VideoWriter(os.path.join(self.result_dir, 'image_video.avi'), self.hparams.fps, video_size)
			mask_video = VideoWriter(os.path.join(self.result_dir, 'mask_video.avi'), self.hparams.fps, video_size)
			result_video = VideoWriter(os.path.join(self.result_dir, 'result_video.avi'), self.hparams.fps, video_size)

		for i, data in enumerate(tqdm(test_loader)):
			image, mask = data

			image = image.to(self.device)
			mask = mask.to(self.device)

			predicted_mask = self.model(image)
			loss = self.calculate_loss(mask, predicted_mask)

			# Store the results at the interval of 20 epochs after first epoch.
			if self.hparams.store_results and self.epoch%20 == 0 and not self.hparams.use_3d_data:
				self.store_results(image, predicted_mask, i)
				self.store_comparisons(mask, predicted_mask, i)				

			if self.hparams.store_video:
				image_video.write(image)
				mask_video.write(mask)
				result_video.write(predicted_mask)

			test_loss += loss.item()
			count += 1

		if self.hparams.store_video:
			image_video.release()
			mask_video.release()
			result_video.release()

		test_loss = float(test_loss)/count
		return test_loss

	def train(self, train_loader, test_loader):
		self.create_optimizer()				# Define optimizer.
		# Keep a track of best_test_loss.
		try:
			self.best_test_loss = checkpoint['min_loss']
		except:
			self.best_test_loss = np.inf

		# Start the training loop.
		for self.epoch in range(self.hparams.start_epoch, self.hparams.epochs):
			train_loss = self.train_one_epoch(train_loader)
			test_loss = self.test_one_epoch(test_loader)

			self.save_checkpoint() 					# Store checkpoint/weights in each epoch.
			if test_loss < self.best_test_loss:
				self.best_test_loss = test_loss
				self.save_checkpoint(best=True)		# Store checkpoint/weights only if best_test_loss decreases.

			log_data = {'train_loss':train_loss, 'test_loss': test_loss, 'best_test_loss': self.best_test_loss}
			self.tensorboard_log(log_data)
			self.text_log(log_data)

	# Only called for evaluation when a pretrained model is given.
	def test(self, test_loader):
		self.is_training = False
		self.epoch = 0
		test_loss = self.test_one_epoch(test_loader)
		print('Evaluation Loss: {}'.format(test_loss))

	# Used to store the weights and checkpoints.
	def save_checkpoint(self, best=False):
		snap = {'epoch': self.epoch + 1,
				'model': self.model.state_dict(),
				'min_loss': self.best_test_loss,
				'optimizer' : self.optimizer.state_dict(),}

		if best:
			torch.save(snap, 'checkpoints/%s/models/best_model_snap.t7' % (self.hparams.exp_name))
			torch.save(self.model.state_dict(), 'checkpoints/%s/models/best_model.t7' % (self.hparams.exp_name))
		else:
			torch.save(snap, 'checkpoints/%s/models/model_snap.t7' % (self.hparams.exp_name))
			torch.save(self.model.state_dict(),'checkpoints/%s/models/model.t7' % (self.hparams.exp_name))

	def tensorboard_log(self, log_data):
		for key, val in log_data.items():
			self.tensorboard_logger.add_scalar(key, val, self.epoch+1)

	def text_log(self, log_data):
		text = 'EPOCH:: {}'.format(self.epoch+1)
		for key, val in log_data.items():
			text += ' {}: {}'.format(key, val)
		self.text_logger.cprint(text)
