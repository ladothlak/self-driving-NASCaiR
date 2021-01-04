#Train the RNN using actual inputs as the labeled data to be predicted
import os
import torch
import numpy as np

from torch import optim, nn
from torchvision import transforms
from VideoDataset import VideoDataset
from test_rnn import Resnt18Rnn
from time import time

#Clear out any memory in use
torch.cuda.empty_cache()

#Model params
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DIMS = [224, 224]
SEQUENCE_LENGTH = 15
#Latest training was with BATCH_SIZE = 24
BATCH_SIZE = 128
LSTM_INPUT = 6
EPOCHS = 15

params_model={
        "num_classes": 4,
        "dr_rate": 0.1,
        "pretrained" : True,
        "rnn_num_layers": 2,
        "rnn_hidden_size": 200,}

MODEL = Resnt18Rnn(params_model).train()
CRITERION = nn.BCEWithLogitsLoss()
OPTIMIZER = optim.Adam(MODEL.parameters(), lr=1e-4)

device = torch.device('cuda')

## Load in recordings of videos and inputs ##

#Get path to data directories
full_path = 'D:\\UsersRedirect\\Josh Cardosi\\Desktop\\Code\\TrackMania Driver'
data_directories = [f.path for f in os.scandir('data') if f.is_dir()]

#Capture all the different image and label directories we want to use for our 
#training data
img_directories = []
label_directories = []
for directory in data_directories:
    img_directories.append(f'{directory}\\img')
    label_directories.append(f'{directory}\\input')

#Define the transform we want for the images. Normalize is based on AlexNet
#specifications
full_transform = transforms.Compose([ 
        transforms.Resize(DIMS),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])

#Create training dataset
train_ds = VideoDataset(img_directories, label_directories, full_transform, 4)

#Put the data in dataloaders
def collate_fn_rnn(batch):
    imgs_batch, label_batch = list(zip(*batch))
    imgs_batch = [imgs for imgs in imgs_batch if len(imgs)>0]
    label_batch = [torch.tensor(l) for l, imgs in zip(label_batch, imgs_batch) if len(imgs)>0]
    imgs_tensor = torch.stack(imgs_batch)
    labels_tensor = torch.stack(label_batch)
    return imgs_tensor, labels_tensor

train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, collate_fn=collate_fn_rnn,
                          pin_memory=True)

#Define the training loop
def train(model, epochs, loss_fn, optimizer, train_loader):
    #Train that bad boy
    loss_history = []
    
    model.to(device, non_blocking=True)
    model.train()
    
    for epoch in range(1,epochs+1):
        start_epoch = time()
        print(f'Epoch {epoch}')
        
        for batch, (x,y) in enumerate(train_loader):
            
            x = x.to(device, non_blocking=True)
            y = (y.type(torch.FloatTensor).to(device, non_blocking=True))
            
            model.zero_grad()
            
            train_prediction = model.forward(x)#[0].unsqueeze(0)
            
            loss = loss_fn(train_prediction, y[:,-1,:])
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            
            loss_history.append(loss.data.item())
            
            if batch%20 == 0:
                print(batch)
            
        print(f'Epoch time: {time()-start_epoch}')
        print(f'Average batch error: {round(np.mean(loss_history), 5)}')
        loss_history = []
    
    return model

def pretty_size(size):
	"""Pretty prints a torch.Size object"""
	assert(isinstance(size, torch.Size))
	return " × ".join(map(str, size))

def dump_tensors(gpu_only=True):
	"""Prints a list of the Tensors being tracked by the garbage collector."""
	import gc
	total_size = 0
	for obj in gc.get_objects():
		try:
			if torch.is_tensor(obj):
				if not gpu_only or obj.is_cuda:
					print("%s:%s%s %s" % (type(obj).__name__, 
										  " GPU" if obj.is_cuda else "",
										  " pinned" if obj.is_pinned else "",
										  pretty_size(obj.size())))
					total_size += obj.numel()
			elif hasattr(obj, "data") and torch.is_tensor(obj.data):
				if not gpu_only or obj.is_cuda:
					print("%s → %s:%s%s%s%s %s" % (type(obj).__name__, 
												   type(obj.data).__name__, 
												   " GPU" if obj.is_cuda else "",
												   " pinned" if obj.data.is_pinned else "",
												   " grad" if obj.requires_grad else "", 
												   " volatile" if obj.volatile else "",
												   pretty_size(obj.data.size())))
					total_size += obj.data.numel()
		except Exception as e:
			pass        
	print("Total size:", total_size)

#dump_tensors()

trained_model = train(MODEL, EPOCHS, CRITERION, OPTIMIZER, train_dl)

torch.save(trained_model, f'models\\trained_model_{time()}.obj')

print('All done!')

torch.cuda.empty_cache()

#dump_tensors()