import cv2 as cv
import numpy as np
import torch
import os
import win32com.client
import pyvjoy

from time import time
from screenshot import take_screenshot
from torchvision import transforms
from pynput import keyboard
from assetto_corsa_telemetry_reader import AssettoCorsaData

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

switcher = {0:'<', 1:'^', 2:'_'}

j = pyvjoy.VJoyDevice(1)
j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
j.set_axis(pyvjoy.HID_USAGE_X, 0x1)
j.reset()
j.reset_buttons()
j.reset_data()
j.reset_povs()

vjoy_const = 32768

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

def get_predictions(model, img, tel, hidden):
    #Get model prediction
    model_inputs, hidden = model.predict(img, tel, hidden)
    
    #Take predicted input and convert back to list we can work with here
    model_inputs = model_inputs.detach().cpu().numpy()[0][0]
    logits = model_inputs
    #model_inputs = [1 if keypress>0.5 else 0 for keypress in model_inputs]
    
    print(logits)
    
    return model_inputs, hidden, logits

def input_network_recommendation(model_inputs, window):
    client = win32com.client.Dispatch("WScript.Shell")
    client.AppActivate(window)
    
    steering = int((model_inputs[0]+1)*vjoy_const)
    throttle = int(model_inputs[1]*vjoy_const)
    brakes = int(model_inputs[2]*vjoy_const)
    
    print(steering)
    
    #Steering
    j.set_axis(pyvjoy.HID_USAGE_Z, steering)
    #Throttle
    j.set_axis(pyvjoy.HID_USAGE_X, throttle)
    #Brake
    j.set_axis(pyvjoy.HID_USAGE_Y, brakes)

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
    cv.moveWindow(img_window_name,1500,700)
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

def run_action_loop(model, hidden, window, reader, fps_target=15, take_action=True):
    loop_time = time()

    screenshot = np.array(take_screenshot(window))
    
    img = prep_DaiLE_img(screenshot)
    
    telemetry_data = reader.getData()
    speed, steering_angle = telemetry_data.get('speed')/300, telemetry_data.get('steerAngle')
    tel = torch.Tensor([speed, steering_angle]).unsqueeze(0).cuda()
        
    model_inputs, hidden, logits = get_predictions(model, img, tel, hidden)
    
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
    assettoReader = AssettoCorsaData()
    assettoReader.start()
    
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
    MODEL_PATH = 'models\\trained_model_1609973973.8520324.obj'
    MODEL = torch.load(MODEL_PATH).eval()
    
    try:
        while listener.running:
            hidden, fps = run_action_loop(MODEL, hidden, window, assettoReader)
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
    
    j.set_axis(pyvjoy.HID_USAGE_Z, int(vjoy_const/2))
    j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
    j.set_axis(pyvjoy.HID_USAGE_X, 0x1)
    
    j.reset()
    j.reset_buttons()
    j.reset_data()
    j.reset_povs()
        
    print('Done.')
        
    torch.cuda.empty_cache()