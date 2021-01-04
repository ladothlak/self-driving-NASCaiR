import cv2 as cv
import numpy as np
import torch
import os
import win32com.client

from time import time
from screenshot import take_screenshot
from torchvision import transforms
from pynput import keyboard

## Parameters related to window
window = 'Assetto Corsa'
img_window_name = 'DaiLE'
board = keyboard.Controller()
client = win32com.client.Dispatch("WScript.Shell")

## Plot parameters
#pw = pg.ImageView()
#pw.show()

font = cv.FONT_HERSHEY_SIMPLEX
org = (250, 700)
fontScale = 8
color = (255, 0, 0) 
spacing = 200
thickness = 12

## Key parameters
switcher = {
    0:'w',
    1:'a',
    2:'s',
    3:'d'
    }

## Model parameters
FPS_TARGET = 15
RNN_LAYER_SIZE = [200, 200]
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DIMS = [224, 224]
MODEL_PATH = 'models\\trained_model_1609697904.8732946.obj'
MODEL = torch.load(MODEL_PATH).eval()

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

## Utility functions

screenshot = np.ones([1,1])

def update_function(frame, cam_id):
    frame[...] = screenshot[...]
    
def time_to_loop(loop_time, show_fps = False):
    fps = 1 / (time() - loop_time)
    
    if show_fps:
        print('FPS {}'.format(fps))
        
    return fps

def on_press(key):
    if key == keyboard.Key.f10:
        # Stop listener
        return False

def get_predictions(img, hidden):
    #Get model prediction
    if hidden == None:
        model_inputs, hidden = MODEL.predict(img, None)
    else:
        model_inputs, hidden = MODEL.predict(img, hidden)
        #del old_hidden
    
    #Take predicted input and convert back to list we can work with here
    model_inputs = model_inputs.detach().cpu().numpy()[0][0]
    logits = model_inputs
    print(f'Output logits: {logits}')
    model_inputs = [1 if keypress>0.5 else 0 for keypress in model_inputs]
    
    return model_inputs, hidden, logits

def input_network_recommendation(model_inputs):
    for keypress in range(len(model_inputs)):
        client.AppActivate(window)
        cur_key = switcher.get(keypress)
    
        if model_inputs[keypress] == 0:
            board.release(cur_key)
        else:
            board.press(cur_key)
            pass

def run_main():
    loop_time = time()
    time_stats = []
    
    model_inputs = [0, 0, 0, 0]
    
    hidden = None
    fps = 0
    while(listener.running):
        # img_stack = []
        # while True:
        #     client.AppActivate(window)
        #     # get an updated image of the game
        #     screenshot = np.array(take_screenshot(window))
            
        #     full_transform = transforms.Compose([
        #             transforms.ToPILImage(mode=None), 
        #             transforms.Resize(DIMS),
        #             transforms.ToTensor(),
        #             transforms.Normalize(MEAN, STD)
        #             ])
            
        #     img = full_transform(screenshot).unsqueeze(0).cuda()
        #     img_stack.append(img)
            
        #     if len(img_stack) >= 4:
        #         break

        # img = torch.stack(img_stack, 1)
        
        # get an updated image of the game
        screenshot = np.array(take_screenshot(window))
        
        full_transform = transforms.Compose([
                transforms.ToPILImage(mode=None), 
                transforms.Resize(DIMS),
                transforms.ToTensor(),
                transforms.Normalize(MEAN, STD)
                ])
        
        img = full_transform(screenshot).unsqueeze(0).unsqueeze(0).cuda()
        'ss'
        
        #for cur_key in ['a', 'd']:
        #    board.release(cur_key)
            
        model_inputs, hidden, logits = get_predictions(img, hidden)
        
        #Enact model key presses
        input_network_recommendation(model_inputs)
        
        print(f'thresholded output: {model_inputs}')
        
        org_x, org_y = org
        #Change screenshot to show DaiLE's decision
        for keypress in range(len(model_inputs)):
            
            if logits[keypress] < 0.5:
                color = (127.5+127.5*(1-logits[keypress]), 0, 0)
            else:
                color = (0, 127.5+127.5*(logits[keypress]), 0)
                
            screenshot = cv.putText(screenshot, switcher.get(keypress),
                                    (org_x, org_y), font, fontScale, color, 
                                    thickness, cv.LINE_AA)
            
            org_x += spacing
        
        #Make live plot showing DaiLE's vision
        screenshot = cv.resize(screenshot, (256, 180))
        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2RGB)
        cv.imshow(img_window_name, screenshot)
        cv.moveWindow(img_window_name,1500,700)
        cv.waitKey(1)
        
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
        del model_inputs, img, screenshot
    
    min_fps = min(time_stats)
    max_fps = max(time_stats)
    mean_fps = np.mean(time_stats)
    
    print(min_fps, mean_fps, max_fps)
    
    for cur_key in ['w', 'a', 's', 'd']:
        board.release(cur_key)
        
    #board.press(keyboard.Key.esc)
    cv.destroyWindow(img_window_name)
    
    print('Done.')

#Start our non-blocking key listener
listener = keyboard.Listener(
    on_press=on_press)
listener.start()

torch.cuda.empty_cache()

try:
    run_main()
except:
    for cur_key in ['w', 'a', 's', 'd']:
        board.release(cur_key)
    
    try:
        cv.destroyWindow(img_window_name)
    except:
        print('No window to be destroyed')
    print('Done.')
    
torch.cuda.empty_cache()

'a'