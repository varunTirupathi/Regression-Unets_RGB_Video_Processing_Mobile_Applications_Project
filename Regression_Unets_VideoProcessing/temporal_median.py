from __future__ import print_function

import argparse
import numpy
import os
import glob
import sys
import copy
import numpy as np

from multiprocessing import Process, Pool
import matplotlib.pyplot as plt
import cv2

# Store videos in a file.
class VideoWriter:
    # Create video writer in opencv.
    def __init__(self, filename, size, fps=10):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video = cv2.VideoWriter(filename, fourcc, fps, size)

    # Write frame in the video writer.
    def write(self, frame):
        self.video.write(frame.astype(np.uint8))
        return True

    # release the video writer.
    def release(self):
        self.video.release()


def read_data(root_dir):
    input_data = [os.path.join(root_dir, x) for x in os.listdir(root_dir)]
    input_data = sorted(input_data, key=key_func)
    images = []
    for path in input_data:
        images.append(plt.imread(path))
    return np.array(images)

def get_number_of_frames(input_data):
    return input_data.shape[0]

def key_func(element):
    return int(element.split('/')[1].split('.')[0])

def get_frame_limit(limit_frames, globsize):
    """ Determines a limit on the number of frames

    Args:
        limit_frames:
        globsize:

    Returns:
        int: total frames to run TMF on
    """
    if limit_frames != -1:
        if globsize > limit_frames:
            total_frames = limit_frames
            print("Frames limited to ", limit_frames)
        else:
            print("Frame limit of ", limit_frames, "is higher than total # of frames: ", globsize)
            total_frames = globsize
    else:
        total_frames = globsize

    return total_frames


def temporal_median_filter_multi2(images, limit_frames=-1, output_format="JPEG", frame_offset=8, simultaneous_frames=8):
    """
    Uses multiprocessing to efficiently calculate a temporal median filter across set of input images.

    DIAGRAM:
    f.o = offset (you actually get 2x what you ask for)
    s.o = simultaneous offset (cuz we do multiple frames at the SAME TIME)
    randframes = we make some random frames for before/after so that we don't run out of frames to use

                   |_____________________total frames______________________|
    randframes_----0      |--f.o----|s.o|---f.o---|


    Args:
        images: 32 images stacked together in an array
        limit_frames: put a limit on the number of frames
        output_format: select PNG, TIFF, or JPEG (default)
        frame_offset: Number of frames to use for median calculation
        simultaneous_frames: Number of frames to process simultaneously
    Returns:
        final_results: 32 images stacked in an array after applying temporal median filter
    """

    height, width = images[0].shape[0], images[0].shape[1]
    total_frames = get_frame_limit(limit_frames, get_number_of_frames(images))

    median_array = numpy.zeros((frame_offset+simultaneous_frames+frame_offset, height, width, 3),numpy.uint8)

    for frame in range(frame_offset):
        median_array[frame, :, :, :] = numpy.random.randint(low=0, high=255, size=(height, width, 3))

    # read all the frames into big ol' array
    for frame_number in range(simultaneous_frames+frame_offset):
        next_array = numpy.array(images[frame_number], numpy.uint8)
        median_array[frame_offset+frame_number, :, :, :] = next_array
        del next_array

    #                |_____________________total frames______________________|
    # randframes_----0      |--f.o----|s.o|---f.o---|
    # whole_array = numpy.zeros((total_frames, height, width, 3), numpy.uint8)

    p = Pool(processes=8)
    current_frame = 0
    filtered_array = numpy.zeros((simultaneous_frames, height, width, 3), numpy.uint8)

    final_results = []

    while current_frame < total_frames:
        if current_frame == 0:
            pass
        else:
            median_array = numpy.roll(median_array, -simultaneous_frames, axis=0)
            for x in range(simultaneous_frames):
                if (current_frame+frame_offset+x) >= total_frames:
                    next_array = numpy.random.randint(low=0, high=255, size=(height, width, 3))
                else:
                    next_array = numpy.array(images[frame_offset+current_frame+x], numpy.uint8)
                median_array[frame_offset+frame_offset+x, :, :, :] = next_array

        slice_list = []
        for x in range(simultaneous_frames):
            if (x+current_frame) >= total_frames:
                break
            else:
                slice_list.append(median_array[x:(x+frame_offset+frame_offset)])

        # calculate medians in our multiprocessing pool
        results = p.map(median_calc, slice_list)

        for frame in range(len(results)):
            filtered_array[frame, :, :, 0] = results[frame][0]
            filtered_array[frame, :, :, 1] = results[frame][1]
            filtered_array[frame, :, :, 2] = results[frame][2]

            final_results.append(copy.deepcopy(filtered_array[frame, :, :, :]))
        current_frame += simultaneous_frames
    return np.array(final_results)


def median_calc(median_array):
    return numpy.median(median_array[:, :, :, 0], axis=0), \
           numpy.median(median_array[:, :, :, 1], axis=0), \
           numpy.median(median_array[:, :, :, 2], axis=0)

# Save the results.
def save(final_results):
    results = 'result'
    os.mkdir(results)
    print(len(final_results))
    for idx, img in enumerate(final_results):
        plt.imsave(f"{results}/{idx}.png", img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser._optionals.title = 'arguments'

    parser.add_argument("-i", "--input_dir",
                        help="Input directory. Set of frames or a single video. Tests for frames first.", required=True)

    args = parser.parse_args()
    
    # Find full path of videos.
    video_paths = [os.path.join(args.input_dir, x) for x in os.listdir(args.input_dir)]

    for idx, path in enumerate(video_paths):
        # read each video and process it to apply temporal mean.
        cap = cv2.VideoCapture(path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap. get(cv2. CAP_PROP_FPS)
        size = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # store the ground truth temporal mean filter applied videos.
        mask_writer = VideoWriter(path.split('.')[0]+'_mask.avi', size=(width, height), fps=fps)

        for j in range(int(size/32)):
            try:
                frames = []
                for i in range(32):
                    _, frame_i = cap.read()
                    frames.append(frame_i)

                frames = np.array(frames)
                masks = temporal_median_filter_multi2(frames)           # apply filter.

                for i in range(32):
                    mask_writer.write(masks[i])
            except:
                break
        mask_writer.release()