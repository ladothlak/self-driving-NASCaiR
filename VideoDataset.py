## Code slightly modified from this article: https://medium.com/howtoai/video-classification-with-cnn-rnn-and-pytorch-abe2f9ee031

from torch.utils.data import Dataset
import glob
from PIL import Image
import torch
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, ids, labels, telemetry, transform, timesteps):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.telemetry = telemetry
        self.timesteps = timesteps
        
        path2imgs = []
        path2labels = []
        path2telemetry = []
        
        for directory in self.ids:
            cur_img_path = sorted(glob.glob(directory+"\\*.jpg"), key=len)
            
            runout = -(len(cur_img_path)%self.timesteps)
            
            if runout != 0:
                cur_img_path = cur_img_path[:runout]
            
            path2imgs = path2imgs + cur_img_path
            
        for directory in self.labels:
            cur_label_path = sorted(glob.glob(directory+"\\*.npy"), key=len)
            
            runout = -(len(cur_label_path)%self.timesteps)
            
            if runout != 0:
                cur_label_path = cur_label_path[:runout]
            
            path2labels = path2labels + cur_label_path
            
        for directory in self.telemetry:
            cur_label_path = sorted(glob.glob(directory+"\\*.npy"), key=len)
            
            runout = -(len(cur_label_path)%self.timesteps)
            
            if runout != 0:
                cur_label_path = cur_label_path[:runout]
            
            path2telemetry = path2telemetry + cur_label_path
        
        self.path2imgs = path2imgs
        self.path2labels = path2labels
        self.path2telemetry = path2telemetry
        
        self.length = int(len(self.path2imgs)/self.timesteps)
        #self.length = int(len(self.path2imgs)-self.timesteps)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx): 

        start = idx*self.timesteps
        stop = (idx+1)*self.timesteps   
        
        #start = idx
        #stop = idx+self.timesteps
        
        temp_path2imgs = self.path2imgs[start:stop]
        temp_path2labels = self.path2labels[start:stop]
        temp_path2telemetry = self.path2telemetry[start:stop]
        
        frames = []

        for p2i in temp_path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)
        
        labels = []
        for p2l in temp_path2labels:
            label = np.load(p2l)
            labels.append(label)
            
        telemetries = []
        for p2t in temp_path2telemetry:
            telemetry = np.load(p2t)
            telemetries.append(telemetry)
           
        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        return frames_tr, telemetries, labels
    


