## Code slightly modified from this article: https://medium.com/howtoai/video-classification-with-cnn-rnn-and-pytorch-abe2f9ee031

from torch.utils.data import Dataset
import glob
from PIL import Image
import torch
import numpy as np

class VideoDataset(Dataset):
    def __init__(self, ids, labels, transform, timesteps):      
        self.transform = transform
        self.ids = ids
        self.labels = labels
        self.timesteps = timesteps
        
        path2imgs = []
        path2labels = []
        
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
        
        self.path2imgs = path2imgs
        self.path2labels = path2labels
        self.length = int(len(self.path2imgs)/self.timesteps)
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx): 

        start = idx*self.timesteps
        stop = (idx+1)*self.timesteps      
        
        temp_path2imgs = self.path2imgs[start:stop]
        temp_path2labels = self.path2labels[start:stop]
        
        frames = []

        for p2i in temp_path2imgs:
            frame = Image.open(p2i)
            frames.append(frame)
        
        labels = []
        for p2l in temp_path2labels:
            label = np.load(p2l)
            labels.append(label)
           
        frames_tr = []
        for frame in frames:
            frame = self.transform(frame)
            frames_tr.append(frame)
        if len(frames_tr)>0:
            frames_tr = torch.stack(frames_tr)
        return frames_tr, labels
        
        # batch_frames = torch.Tensor()
        # batch_labels = torch.Tensor()
        
        # for x in range(int(len(path2imgs)/self.timesteps)-1):
            
        #     temp_path2imgs = path2imgs[x*self.timesteps:(x+1)*self.timesteps]
        #     temp_path2labels = path2labels[x*self.timesteps:(x+1)*self.timesteps]
            
        #     frames = []
            
        #     for p2i in temp_path2imgs:
        #         frame = Image.open(p2i)
        #         frames.append(frame)
            
        #     labels = []
            
        #     for p2l in temp_path2labels:
        #         label = np.load(p2l)
        #         labels.append(label)
                
        #     frames_tr = []
                
        #     for frame in frames:
        #         frame = self.transform(frame)
        #         frames_tr.append(frame)
           
        #     if len(batch_frames)>0:
        #         frames_tr = torch.stack(frames_tr).unsqueeze(0)
        #         labels = torch.Tensor(labels).unsqueeze(0)
                
        #         print('frames',batch_frames.shape)
        #         print('labels',batch_labels.shape)
                
        #         batch_frames = torch.cat((batch_frames, frames_tr))
        #         batch_labels = torch.cat((batch_labels, labels))
        #     else:
        #         batch_frames = torch.stack(frames_tr).unsqueeze(0)
        #         batch_labels = torch.Tensor(labels).unsqueeze(0)
        
        # return batch_frames, batch_labels


    
    
    
# path2imgs = sorted(glob.glob(self.ids[idx]+"\\*.png"), key=len)
# path2labels = sorted(glob.glob(self.labels[idx]+"\\*.npy"), key=len)

# path2imgs = path2imgs[:self.timesteps]
# path2labels = path2labels[:self.timesteps]

# frames = []
# print(path2imgs)
# print(path2labels)
# for p2i in path2imgs:
#     frame = Image.open(p2i)
#     frames.append(frame)

# labels = []
# for p2l in path2labels:
#     label = np.load(p2l)
#     labels.append(label)
   
# frames_tr = []
# for frame in frames:
#     frame = self.transform(frame)
#     frames_tr.append(frame)
# if len(frames_tr)>0:
#     frames_tr = torch.stack(frames_tr)
# return frames_tr, labels


