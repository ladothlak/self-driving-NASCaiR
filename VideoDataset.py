## Code slightly modified from this article: https://medium.com/howtoai/video-classification-with-cnn-rnn-and-pytorch-abe2f9ee031

from torch.utils.data import Dataset
from torchvision import transforms
import glob
from PIL import Image
import torch
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, data, transform, timesteps):      
        self.transform = transform
        self.data = data
        self.timesteps = timesteps
        self.windows_per_minibatch = 256-self.timesteps
        
        directory_structure = []
        
        for directory in self.data:
            cur_img_path = sorted(glob.glob(directory+"\\img\\*.jpg"), key=len)[:256]
            cur_label_path = sorted(glob.glob(directory+"\\input\\*.npy"), key=len)[:256]
            cur_tel_path = sorted(glob.glob(directory+"\\telemetry\\*.npy"), key=len)[:256]
            
            if len(cur_img_path) >= 256:
                directory_structure.append([cur_img_path, cur_label_path, cur_tel_path])  
        
        self.directory_structure = np.array(directory_structure)
        self.length = len(self.directory_structure)
        
    def __len__(self):
        return (self.length)*(self.windows_per_minibatch)
    
    def __getitem__(self, idx): 

        current_minibatch_set = (idx // self.windows_per_minibatch)
        horizontal_flip = np.random.rand(0, 1)
        
        idx = idx - current_minibatch_set*self.windows_per_minibatch
            
        #minibatch_start = current_minibatch_set*self.minibatch_size
        #minibatch_stop = (current_minibatch_set+1)*self.minibatch_size
        window_start = idx
        window_stop = idx+self.timesteps
        
        temp_minibatch_directory = self.directory_structure[current_minibatch_set]
        
        temp_path2imgs = temp_minibatch_directory[0,window_start:window_stop]
        temp_path2labels = temp_minibatch_directory[1,window_start:window_stop]        
        temp_path2telemetry = temp_minibatch_directory[2,window_start:window_stop]
        
        frames_tr = []
        labels = []
        telemetries = []
            
        for p2i, p2l, p2t in list(zip(temp_path2imgs, temp_path2labels, temp_path2telemetry)):
            frame = Image.open(p2i)
            label = torch.as_tensor(np.load(p2l), dtype=torch.float16)
            telemetry = torch.as_tensor(np.load(p2t), dtype=torch.float16)
            
            if horizontal_flip >= 0.5:
                frame = transforms.functional.hflip(frame)
                label[0] = label[0]*-1
                telemetry[1] = telemetry[1]*-1
                pass

            
            frames_tr.append(self.transform(frame).type(torch.FloatTensor))
            labels.append(label)
            telemetries.append(telemetry)
            
        return torch.stack(frames_tr), torch.stack(telemetries), torch.stack(labels)


