# system libraries
import os
import os.path
from pathlib import Path

# data processing libraries
import pandas as pd
import numpy as np
import cv2

# pytorch libraries
import torch
import torchvision as tv

from typing import Dict, Any


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


class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, 
                 root_path: str,
                 metadata: np.ndarray):
        """
            DataLoader Class Constructor
            -------------------------------------------
            Inputs:
                root_path  {list}: directory with all the videos
                metadata {str}: 2D array containing the metadata of the videos in the following format:
                                1. video_name {str}: the name of the video  (column 0)
                                2. start_frame {int}: the starting frame of the video (column 1)
                                3. end_frame {int}: the ending frame of the video (column 2)
                                4. label {int}: the label/annotation of the video (column 3)
        """
        self.root_path = root_path
        self.metadata = metadata
    
        # parse metadata into a list of video records
        self._parse_metadata()


    def _parse_metadata(self):
        """
            Method to parse the metadata numpy array into a list of video record
        """
        self.video_list = [VideoRecord(self.root_path, row) for row in self.metadata]


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
            Method to get the item at the given index
            -------------------------------------------
            Inputs:
                index {int}: index of the item
            Outputs:
                Dictionary Containing:
                    video {tensor}: tensor of the video
                    label {int}: label of the video
        """
        # get the video name and the associated label
        record: VideoRecord = self.video_list[idx]
        video_name = record._path
        label = record.label

        # intialize video capture object
        vc = cv2.VideoCapture(video_name)

        # calculate total number of frames in the video
        total_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))

        # fix cases when the video is not available
        if total_frames != 0:
            # as long as there are frames to read
            # start reading frame by frame
            i = 0
            frames = []
            while i < total_frames:
                ok, f = vc.read()
                if ok:
                    # convert frame to tensor
                    f = tv.transforms.ToTensor()(f)
                    frames.append(f)

                    # increment frame counter
                    vc.set(cv2.CAP_PROP_POS_FRAMES, i)
                    i += 1
                else:
                    break
            
            # release the video capture object
            vc.release()
            return {'video': torch.stack(frames), 'label': label}
        else:
            return {'video': torch.zeros(1), 'label': 'None'}

    def __len__(self): return len(self.data_path)



if __name__ == "__main__":
    print("VideoLoader.py is being run directly")