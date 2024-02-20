"""
    Efficient adaptable dataset class for loading videos
    Instead of loading every frame of a video, 
    this class loads x RGB frames of a video (sparse temporal sampling) and eventually
    chooses those frames from start to end of the video, returning 
    a list of tensors of shape (F, C, H, W) -> (Frames, Channels, Height, Width)
    References:
        - Temporal Segment Networks for Action Recognition in Videos: https://arxiv.org/abs/1608.00859
"""
# system libraries
import os
import os.path

# data processing libraries
import pandas as pd
import numpy as np
import cv2

# pytorch libraries
import torch
import torchvision as tv
from torchvision import transforms

from typing import List, Union, Tuple, Any


class VideoRecord(object):
    def __init__(self, root_path, row):
        """
            VideoRecord Class Constructor
            This class represent a video sample's metadata
            -----------------------------------------------
            Inputs:
                root_path {str}: the system path to the root directory containing the videos
                row {List}: a list containing records of video elements
                    1. video_name {str}: the name of the video
                    2. start_frame {int}: the starting frame of the video
                    3. end_frame {int}: the ending frame of the video
                    4. label {int}: the label/annotation of the video
        """
        self._path = os.path.join(root_path, f"{row[0]}.mp4")
        self._data = row


    @property
    def path(self) -> str:
        return self._path
    

    @property
    def num_frames(self) -> int:
        return np.floor(self.end_frame) - np.ceil(self.start_frame) + 1 # inclusive
    

    @property
    def start_frame(self) -> int:
        return int(self._data[1])
    

    @property
    def end_frame(self) -> int:
        return int(self._data[2])
    

    @property
    def label(self) -> Any:
        return self._data[3]
        

class VideoFrameDataset(torch.utils.data.Dataset):
    def __init__(self, 
                root_path: str,
                metadata: np.ndarray,
                num_segments: int,
                frames_per_segment: int,
                transform=None,
                test_mode: bool = False):
        """
            DataLoader Class Constructor
            Notes:
                This class 
            -------------------------------------------
            Inputs:
                root_path {str}: path to the root directory containing the videos
                metadata {str}: 2D array containing the metadata of the videos in the following format:
                                1. video_name {str}: the name of the video  (column 0)
                                2. start_frame {int}: the starting frame of the video (column 1)
                                3. end_frame {int}: the ending frame of the video (column 2)
                                4. label {int}: the label/annotation of the video (column 3)
                num_segments {int}: number of segments to split the video to sample frames from
                frames_per_segment {int}: number of frames to be loaded per segment. 
                                          For each segment's frame-range, a random start index or 
                                          center is chosen, from which frames_per_segment 
                                          consecutive frames are loaded.
                transform {torchvision.transforms}: transform pipeline that receives a list of PIL images/frames.
                test_mode {bool}: If True, frames are taken from the center of each
                                  segment, instead of a random location in each segment.
        """
        super(VideoFrameDataset, self).__init__()

        # load input parameters into constructor
        self.root_path = root_path
        self.metadata = metadata
        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment
        self.transform = transform
        self.test_mode = test_mode

        # parse metadata into a list of video records
        self._parse_metadata()


    def _parse_metadata(self):
        """
            Method to parse the metadata numpy array into a list of video record
        """
        self.video_list = [VideoRecord(self.root_path, row) for row in self.metadata]


    def _get_video_frames(self, record: VideoRecord) -> "np.ndarray":
        """
            Method to get the frames of the video
            -------------------------------------------
            Inputs:
                record {VideoRecord}: a video sample record
            Outputs:
                frames {np.ndarray}: a numpy array of frames of the video
        """
        vc = cv2.VideoCapture(record.path)
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
        frames = list()
        i=0
        
        if total_frames != 0:
            # as long as there are frames to read
            while i < total_frames:
                # read the frame
                ok, frame = vc.read()
                if ok:
                    # convert frame to color and to tensor
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(frame)

        # release the video object
        vc.release()
        return frames



    def _get_start_indicies(self, record: VideoRecord) -> "np.ndarray[int]":
        """
            Method to get the start indices of the segments
            For each segment, choose a start index from where frames are the be loaded
            -------------------------------------------
            Inputs:
                record {VideoRecord}: a video sample record
            Outputs:
                start_indicies {np.ndarray}: a numpy array of 

        """
        # choose start indices that are perfectly spread across the video frames
        if self.test_mode:
            # in test mode, frames are taken from the center of each segment
            # essentially, the start_indicies is a list denoting the center frame of each segment
            distance_between_indices = (record.num_frames - self.frames_per_segment + 1) / float(self.num_segments)
            start_indicies = np.array([int(np.ceil(distance_between_indices * i)) for i in range(self.num_segments)])
        else:
            # if not in test mode, randomly sample start indices that are approximately evenly spread across the video frames.
            # the start_indicies is a list of random start indices for each segment
            # essentially, the start_index could be on any frame of the segment
            max_valid_start_index = (record.num_frames - self.frames_per_segment + 1) // self.num_segments
            start_indicies = np.multiply(list(range(self.num_segments)), max_valid_start_index) \
                            + np.random.randint(max_valid_start_index, size=self.num_segments)

        return start_indicies



    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Any]:
        """
            Method to get the item at the given index
            -------------------------------------------
            Inputs:
                index {int}: index of the item
            Outputs:
                video {tensor}: tensor of the video
                label {Any}: label of the video
        
        """
        record: VideoRecord = self.video_list[idx]
        frame_start_indices: 'np.ndarray[int]' = self._get_start_indicies(record)

        return self._get(record, frame_start_indices)


    def _get(self, 
             record: VideoRecord, 
             frame_start_indices: 'np.ndarray[int]') -> Tuple[torch.Tensor, Any]:
        """
            Method to get the video frames and label at the corresponding frame start indices
            -------------------------------------------
            Inputs:
                record {VideoRecord}: a video sample record
                frame_start_indices {np.ndarray}: a numpy array of start indices for each segment
            Outputs:
                video {tensor}: tensor of the video
                label {int}: label of the video
        """
        # load the start frame to each indicies
        frame_start_indices = frame_start_indices + record.start_frame
        video_frames = self._get_video_frames(record)

        if len(video_frames) == 0:
            return (torch.zeros(1), None)
        

        frames = list()
        # for each start index, load self.frames_per_segment consecutive frames
        for start_index in frame_start_indices:
            frame_index = int(start_index)

            # load self.frames_per_segment consecutive frames from the video
            for _ in range(self.frames_per_segment):
                
                frame = video_frames[frame_index]
                frame = tv.transforms.ToTensor()(frame)
                frames.append(frame)
                
                if frame_index < record.end_frame:
                    frame_index += 1

            if self.transform is not None:
                frames = self.transform(frames)
            
        return (torch.stack(frames), record.label)

            

    def __len__(self):
        return len(self.video_list)
        
if __name__ == "__main__":
    pass