# Regression-UNets-RGB-video-processing
This Project contains both the Image processing and video processing code which is developed in Pytorch framework.

## Training:
> python main.py --exp_name exp_unet

## Testing:
> python main.py --eval --pretrained 'checkpoints/exp_unet/models/best_model.t7'

## Arguments:
### Training Settings:
1. eval: Evaluate the network.
2. exp_name: Name of the log file created during training.
3. optimizer: Select optimizer from 'Adam' and 'SGD'.
4. lr: Set learning rate.
5. epochs: Total number of epochs for training.

### Network Settings:
1. n_channels: No of channels in the input image.
2. n_classes: No of classes in the output.

### Dataset Settings:
1. img_dir: Directory of input images.
2. mask_dir: Directory of ground truth labels.
3. mask_filter: Choose a filter to create a ground truth mask. ('CannyEdgeDetection', 'GaussianBlur', 'Blur', 'GaussianBlur', 'Sharpen')
4. batch_size: Batch size while training the network.
5. test_batch_size: Batch size during evaluation and testing.
6. workers: Number of workings to load dataset.

### Pretrained Model Settings:
1. resume: Path of checkpoint to resume the training from that checkpoint.
2. pretrained: Path of pretrained model to evaluate the network.
