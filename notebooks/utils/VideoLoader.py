"""
    Module to load videos from a given path 
    then create a dataset using pytorch framework
"""
# imoport necessary libraries
import torch
import torchvision as tv
import cv2

import pandas as pd
import numpy as np

from pathlib import Path

class VideoLoader(torch.utils.data.Dataset):
    def __init__(self, csv_path, root_dir):
        """
            DataLoader Class Constructor
            -------------------------------------------
            Inputs:
                csv_path {str}: path to csv file metadata
                root_dir  {list}: directory with all the videos
        """
        self.metadata = pd.read_csv(csv_path)
        self.root_dir = root_dir


    def __getitem__(self, idx):
        """
            Method to get the item at the given index
            -------------------------------------------
            Inputs:
                index {int}: index of the item
            Outputs:
                video {tensor}: tensor of the video
                label {int}: label of the video
        """
        # get the link and id of the video
        link = self.metadata['url'].iloc[idx]
        id = link.split('=')[1]
        label = self.metadata['clean_text'].iloc[idx]

        # intialize video capture object
        vc = cv2.VideoCapture(self.root_dir + id + '.mp4')

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