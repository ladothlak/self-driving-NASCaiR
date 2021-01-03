import cv2 as cv
import numpy as np
import torch
import os
import win32com.client
import matplotlib.pyplot as plt

from time import time
from screenshot import take_screenshot
from torchvision import transforms
from pynput import keyboard

window = 'Assetto Corsa'
img_window_name = 'NASCaiR'
board = keyboard.Controller()
client = win32com.client.Dispatch("WScript.Shell")

switcher = {
    0:'w',
    1:'a',
    2:'s',
    3:'d'
    }

FPS_TARGET = 15

RNN_LAYER_SIZE = [200, 200]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DIMS = [224, 224]
MODEL_PATH = 'models\\trained_model_1609635973.6221635.obj'
MODEL = torch.load(MODEL_PATH).eval()

def time_to_loop(loop_time, show_fps = False):
    fps = 1 / (time() - loop_time)
    
    if show_fps:
        print('FPS {}'.format(fps))
        
    return fps

def on_press(key):
    if key == keyboard.Key.f10:
        # Stop listener
        return False

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Start our non-blocking key listener
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

torch.cuda.empty_cache()
def run_main():
    loop_time = time()
    time_stats = []
    
    model_inputs = [0, 0, 0, 0]
    
    hidden = None
    fps = 0
    while(listener.running):
        img_stack = []
        while True:
            client.AppActivate(window)
            # get an updated image of the game
            screenshot = np.array(take_screenshot(window))
            
            full_transform = transforms.Compose([
                    transforms.ToPILImage(mode=None), 
                    transforms.Resize(DIMS),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN, STD)
                    ])
            
            img = full_transform(screenshot).unsqueeze(0).cuda()
            img_stack.append(img)
            
            if len(img_stack) >= 4:
                break

        img = torch.stack(img_stack, 1)
        
        
        
        #Get model prediction
        if hidden == None:
            model_inputs, hidden = MODEL.predict(img, None)
        else:
            model_inputs, hidden = MODEL.predict(img, hidden)
            #del old_hidden
        
        #Take predicted input and convert back to list we can work with here
        model_inputs = model_inputs.detach().cpu().numpy()[0][0]
        print(model_inputs)
        #model_inputs = [1 if keypress>0.5 else 0 for keypress in model_inputs]
        
        #Enact model key presses
        for keypress in range(len(model_inputs)):
            client.AppActivate(window)
            cur_key = switcher.get(keypress)
            
            if model_inputs[keypress] == 0:
                board.release(cur_key)
            else:
                board.press(cur_key)
                pass
        
        #screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2RGB)
        #screenshot = cv.putText(screenshot, f'{model_inputs},\n {fps}', (50,50), cv.FONT_HERSHEY_SIMPLEX,
        #                        1, (255, 0, 0), 5, cv.LINE_AA)
        #screenshot = cv.resize(screenshot, (256, 180))
        
        #cv.imshow(img_window_name, screenshot)
        #cv.waitKey(20)
        
        #plt.imshow(screenshot)
        #plt.show()
        print(model_inputs)
        
        # determine fps
        fps = time_to_loop(loop_time)
        
        #Limt loop to TARGET_FPS
        while fps > FPS_TARGET:
            fps = time_to_loop(loop_time)
        
        loop_time = time()
        try:
            time_stats.append(fps)
        except:
            pass
        
        #Clean up variables
        del model_inputs, img, img_stack, screenshot
    
    cv.destroyWindow(img_window_name)
    min_fps = min(time_stats)
    max_fps = max(time_stats)
    mean_fps = np.mean(time_stats)
    
    print(min_fps, mean_fps, max_fps)
    
    print('Done.')
    
run_main()
torch.cuda.empty_cache()