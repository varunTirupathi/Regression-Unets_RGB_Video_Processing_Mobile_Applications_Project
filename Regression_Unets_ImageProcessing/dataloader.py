import os
import glob
from tqdm import tqdm
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# PIL Image stores image in (RGB) and cv2 stores image in (BGR)
# Converts PIL Image to cv2 image.
def pil2cv(image):
	image = np.asarray(image)
	return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
		
# Converts cv2 image to PIL image.
def cv2pil(image):
	if len(image.shape)==3:
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	elif len(image.shape)==2:
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	return Image.fromarray(image)

# Only used for input image conversion (in VideoDataLoader)
def cv2pil_input(image):
	return Image.fromarray(image)

# A class to apply filter.
class Filter:
	def __call__(self, image):
		image = pil2cv(image)
		image = self.apply_filter(image)
		return cv2pil(image)


class CannyEdgeDetection(Filter):
	@staticmethod
	def apply_filter(image, sigma=0.33):
		# Ref. https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
		v = np.median(image)
		lower = int(max(0, (1.0 - sigma) * v))
		upper = int(min(255, (1.0 + sigma) * v))
		edged = cv2.Canny(image, lower, upper)
		edged = 255 - edged
		return edged


class Blur(Filter):
	@staticmethod
	def apply_filter(image, kernel_size=(5,5)):
		# Ref. https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
		return cv2.blur(image, kernel_size)


class Emboss(Filter):
	@staticmethod
	def apply_filter(image, combine=True):
		# Ref. https://medium.com/dataseries/designing-image-filters-using-opencv-like-abode-photoshop-express-part-2-4479f99fb35
		kernel_left = np.array([[0, -1, -1], # kernel for embossing bottom left side
					[1, 0, -1],
					[1, 1, 0]])
		kernel_right = np.array([[-1, -1, 0], # kernel for embossing bottom right side
							[-1, 0, 1],
							[0, 1, 1]])

		height, width, _ = image.shape
		y = np.ones((height, width), np.uint8) * 128

		# you can generate kernels for embossing top as well
		gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		if combine:
			output_left = cv2.add(cv2.filter2D(gray_image, -1, kernel_left), y) # emboss on bottom left side
			output_right = cv2.add(cv2.filter2D(gray_image, -1, kernel_right), y) # emboss on bottom right side
			result = np.maximum(output_left, output_right)
		else:
			result = cv2.add(cv2.filter2D(gray_image, -1, kernel_left), y) # emboss on bottom left side
		return result


class GaussianBlur(Filter):
	# Ref. https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
	@staticmethod
	def apply_filter(image, kernel_size=(5,5), sigma=0):
		return cv2.GaussianBlur(image, kernel_size, sigma)


class Sharpen(Filter):
	@staticmethod
	def apply_filter(image, kernel=None, sharpening_factor=1):
		# sharpening_factor:		It should be in the range of 0 to 5.
		if kernel is None:	
			kernel = np.array([[-1, -1, -1],
							   [-1, 9+sharpening_factor, -1], 
							   [-1, -1, -1]])
		return cv2.filter2D(image, -1, kernel)


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
				output[i, j] = self.sample_entropy(image[:, :, i, j]*255.0)
		return self.ndarray2torch(output)

imsize = (240, 320)
img_transform = transforms.Compose([
	transforms.Resize(imsize),  # scale imported image
	transforms.ToTensor()
])

mask_transform = [
	transforms.Resize(imsize),  # scale imported image
	transforms.ToTensor()
]


class ImageDataset(Dataset):
	def __init__(self, img_dir, mask_dir='', train=True, img_transform=img_transform, mask_transform=mask_transform, mask_filter='CannyEdgeDetection', use_masks_from_dir=False):
		self.img_dir = img_dir
		self.mask_dir = mask_dir
		self.img_transform = img_transform
		self.use_masks_from_dir = use_masks_from_dir

		# Insert the filter transform after the Resize transform in mask_transforms.
		# globals() gives you a directory of all the present classes in this file. ({'Name of Class': 'Class Constructor'})
		try:
			mask_transform.insert(1, globals()[mask_filter]())
		except:
			raise Exception(f'Transform named {mask_filter} is not available')

		self.mask_transform = transforms.Compose(mask_transform)

		try:
			self.ids = [s.split('.')[0] for s in os.listdir(self.img_dir)]
			self.ids.sort()                                                 # sort the images according to indices.
			training_indices = int(len(self.ids)*0.8)
			# Split the data into training and testing in the ratio 8:2.
			if train:
				self.ids = self.ids[:training_indices]
			else:
				self.ids = self.ids[training_indices:]
		except FileNotFoundError:
			self.ids = []

	def __len__(self):
		return len(self.ids)

	def __getitem__(self, index):
		idx = self.ids[index]

		img_files = glob.glob(os.path.join(self.img_dir, idx+'.*'))

		assert len(img_files) == 1, f'{idx}: {img_files}'

		img = Image.open(img_files[0])

		# If we want to use masks from saved dataset.
		if self.use_masks_from_dir:
			mask_files = glob.glob(os.path.join(self.mask_dir, idx+'.*'))
			assert len(mask_files) == 1, f'{idx}: {mask_files}'
			mask = Image.open(mask_files[0])
			assert img.size == mask.size, "image shape: {} & mask shape: {}".format(img.shape, mask.shape)
			return self.img_transform(img), self.img_transform(mask)

		return self.img_transform(img), self.mask_transform(img)


class VideoDataset(Dataset):
	def __init__(self, video_path, train=False, img_transform=img_transform, mask_transform=mask_transform, mask_filter='CannyEdgeDetection', evaluate=False):
		try:
			mask_transform.insert(1, globals()[mask_filter]())
		except:
			raise Exception(f'Transform named {mask_filter} is not available')

		self.mask_transform = transforms.Compose(mask_transform)
		self.img_transform = img_transform
		self.train = train
		self.video_path = video_path
		self.eval = evaluate

		if not self.train and self.eval:
			self.find_video_fps(video_path)
			self.cap = cv2.VideoCapture(video_path)
		else:
			self.total_frames = self.find_total_frames()
		frames = []

	def find_total_frames(self):
		videos = [x for x in os.listdir(self.video_path)]		# List all the videos at given path.

		# Split the data into 8:2.
		training_indices = int(len(videos)*0.8)
		if self.train:
			videos = videos[:training_indices]
		else:
			videos = videos[training_indices:]

		self.videos_full_path = [os.path.join(self.video_path, x) for x in videos]	# Find the full path.
		self.videos_start_idxs = {}									# Store the starting index of each video.
		self.videos_end_idxs = {}									# Store the last index of each video.
		size = 0

		# Go through all the videos.
		for vfp in self.videos_full_path:
			cap = cv2.VideoCapture(vfp)
			self.videos_start_idxs.update({vfp:size})
			size += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))			# Find the number of frames in video.
			self.videos_end_idxs.update({vfp:size-1})
			cap.release()
		return size

	# Find the video path and index of the frame.
	def find_video_idx(self, index):
		for vfp in self.videos_full_path:
			if self.videos_start_idxs[vfp] <= index <= self.videos_end_idxs[vfp]:
				return vfp, index-self.videos_start_idxs[vfp]

	# Find the frame from the given video path and index.
	def find_frame(self, video_full_path, index):
		cap = cv2.VideoCapture(video_full_path)
		cap.set(cv2.CAP_PROP_POS_FRAMES, index)
		_, frame = cap.read()
		return frame

	# Find frame rate of a given video.
	def find_video_fps(self, video_path):
		cap = cv2.VideoCapture(video_path)
		(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
		if int(major_ver) < 3:
			self.fps = int(cap.get(cv2.cv.CV_CAP_PROP_FPS))
			print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(self.fps))
		else:
			self.fps = int(cap.get(cv2.CAP_PROP_FPS))
			print ("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(self.fps))

	def __len__(self):
		if not self.train:
			return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
		else:
			return self.total_frames

	def __getitem__(self, index):
		if not self.train and self.eval:
			_, frame = self.cap.read()				# Find the frame of a video for evaluation.
		else:
			vfp, index = self.find_video_idx(index)		# Find the video path and its index.
			frame = self.find_frame(vfp, index)			# Find the frame from given details.
		return self.img_transform(cv2pil_input(frame)), self.mask_transform(cv2pil(frame))


class Video3DDataset(Dataset):
	def __init__(self, video_path, train=False, img_transform=img_transform, mask_transform=mask_transform, mask_filter='CannyEdgeDetection', evaluate=False):
		try:
			mask_transform.insert(1, globals()[mask_filter]())
		except:
			raise Exception(f'Transform named {mask_filter} is not available')

		self.img_transform = img_transform
		self.train = train
		self.video_path = video_path
		self.eval = evaluate

		self.min_size = 20
		self.index_links = self.find_index_links()
		self.mask_transform = SampleEntropy(self.min_size-1)

	def find_index_links(self):
		videos = [x for x in os.listdir(self.video_path)]		# List all the videos at given path.

		# Split the data into 8:2.
		training_indices = int(len(videos)*0.8)
		if self.train:
			videos = videos[:training_indices]
		else:
			videos = videos[training_indices:]

		self.videos_full_path = [os.path.join(self.video_path, x) for x in videos]	# Find the full path.

		index_links = []
		count = 0
		for vid, vfp in enumerate(self.videos_full_path):
			cap = cv2.VideoCapture(vfp)
			size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
			for i in range(int(size/self.min_size)):
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
		for ii in range(vid_i+1):
			for i in range(self.min_size):
				_, frame_i = cap.read()
				if ii == vid_i:
					frame.append(frame_i)

		frame = [self.img_transform(cv2pil_input(x)).unsqueeze(0) for x in frame]

		frame = torch.cat(frame, axis=0)
		frame = frame.permute(1, 0, 2, 3)		# [channel, frames, height, width]
		# frame = frame.permute(0, 2, 3, 1)		# [channel, height, width, frames]
		return frame.permute(0, 2, 3, 1), self.mask_transform(frame).unsqueeze(0)

if __name__ == '__main__':
	img_dir = './dataset/contour/train/'
	mask_dir = './dataset/contour/train_masks/'
	video_path = 'output_2.avi'
	video_path = "D:/Mobile_research_ project/dataset/dataset/vid"
	dataset = Video3DDataset(video_path, train=True, mask_filter='CannyEdgeDetection')

	from torch.utils.data import DataLoader
	train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, num_workers=4)
	from tqdm import tqdm
	
	for data in tqdm(train_loader):
		img, mask = data
		import pdb; pdb.set_trace()