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
board = keyboard.Controller()

## Key parameters
switcher = {
    0:'w',
    1:'a',
    2:'s',
    3:'d'
    }

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

## Utility functions
    
def time_to_loop(loop_time, show_fps = False):
    fps = 1 / (time() - loop_time)
    
    if show_fps:
        print('FPS {}'.format(fps))
        
    return fps

def on_press(key):
    if key == keyboard.Key.f10:
        # Stop listener
        return False

def get_predictions(model, img, hidden):
    #Get model prediction
    model_inputs, hidden = model.predict(img, hidden)
    
    #Take predicted input and convert back to list we can work with here
    model_inputs = model_inputs.detach().cpu().numpy()[0][0]
    logits = model_inputs
    model_inputs = [1 if keypress>0.5 else 0 for keypress in model_inputs]
    
    return model_inputs, hidden, logits

def input_network_recommendation(model_inputs, window):
    client = win32com.client.Dispatch("WScript.Shell")
    for keypress in range(len(model_inputs)):
        client.AppActivate(window)
        cur_key = switcher.get(keypress)
    
        if model_inputs[keypress] == 0:
            board.release(cur_key)
        else:
            board.press(cur_key)
            pass

def draw_cv_window(screenshot, logits, fps):
    img_window_name = 'DaiLE'
    font = cv.FONT_HERSHEY_SIMPLEX
    org_x, org_y = (250, 700)
    fontScale = 8
    color = (255, 0, 0) 
    spacing = 200
    thickness = 12
    
    #Change screenshot to show DaiLE's decision
    for keypress in range(len(logits)):
        
        if logits[keypress] < 0.5:
            color = (2*255*(0.5-logits[keypress]), 0, 0)
        else:
            color = (0, 2*255*(logits[keypress]-0.5), 0)
            
        screenshot = cv.putText(screenshot, switcher.get(keypress),
                                (org_x, org_y), font, fontScale, color, 
                                thickness, cv.LINE_AA)
        
        
        org_x += spacing
        
    screenshot = cv.putText(screenshot, str(round(fps)),
                        (100,100), font, 6, (255, 0, 0), 
                        thickness, cv.LINE_AA)
    
    #Make live plot showing DaiLE's vision
    screenshot = cv.resize(screenshot, (256, 180))
    screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2RGB)
    cv.imshow(img_window_name, screenshot)
    cv.moveWindow(img_window_name,2580,700)
    cv.waitKey(1)
    
def prep_DaiLE_img(screenshot):
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    DIMS = [224, 224]
    
    full_transform = transforms.Compose([
        transforms.ToPILImage(mode=None), 
        transforms.Resize(DIMS),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
        ])
    
    img = full_transform(screenshot).unsqueeze(0).unsqueeze(0).cuda()
    
    return img

def run_action_loop(model, hidden, window, fps_target=15, take_action=True):
    loop_time = time()

    screenshot = np.array(take_screenshot(window))
    
    img = prep_DaiLE_img(screenshot)
        
    model_inputs, hidden, logits = get_predictions(model, img, hidden)
    
    #Enact model key presses
    if take_action:
        input_network_recommendation(model_inputs, window)
    
    fps = time_to_loop(loop_time)
    
    draw_cv_window(screenshot, logits, fps)
    
    if fps_target!=None:
        #Limit loop to TARGET_FPS
        while fps > fps_target:
            fps = time_to_loop(loop_time)
    
    #Clean up variables
    del model_inputs, img, screenshot
    
    return hidden, fps


if __name__ == '__main__':
    #Get window to control
    window = 'Assetto Corsa'
    
    #Start our non-blocking key listener
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    
    #Create time stats variable
    time_stats = []
    
    #Initialize hidden state
    hidden = None
    
    #Empty cuda cache in case we have any old stuff pinned in there
    torch.cuda.empty_cache()
    
    #Load in a model
    MODEL_PATH = 'models\\trained_model_1609858299.4669878.obj'
    MODEL = torch.load(MODEL_PATH).eval()
    
    try:
        while listener.running:
            hidden, fps = run_action_loop(MODEL, hidden, window)
            time_stats.append(fps)
    except:
        pass
        
    try:
        cv.destroyAllWindows()
    except:
        print('No window to be destroyed')
        
    min_fps = min(time_stats)
    max_fps = max(time_stats)
    mean_fps = np.mean(time_stats)
    
    print(min_fps, mean_fps, max_fps)
    
    for cur_key in ['w', 'a', 's', 'd']:
        board.release(cur_key)
        
    print('Done.')
        
    torch.cuda.empty_cache()