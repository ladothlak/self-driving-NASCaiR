from pynput import keyboard
import numpy as np
import os

from time import time
from PIL import Image
from screenshot import take_screenshot
from assetto_corsa_telemetry_reader import AssettoCorsaData
from XboxController import XboxController

window = 'Assetto Corsa'

#The framerate we will limit data collection to
target_fps = 15
sequence_length = 60

#The input device
controller = XboxController()

def time_to_loop(loop_time, show_fps = False):
    fps = 1 / (time() - loop_time)
    loop_time = time()
    
    if show_fps:
        print('FPS {}'.format(fps))
        
    return fps

def on_release(key):  
    if key == keyboard.Key.f10:
        # Stop listener
        return False



def sample_data(reader, target_fps=15):
    #As long as the listener is running, continue taking screenshots
    loop_time = time()
    telemetry_data = reader.getData()
    
    speed, steering_angle = telemetry_data.get('speed')/300, telemetry_data.get('steerAngle')
    
    #Collect screenshot
    screenshot = np.array(take_screenshot(window))
    
    input_for_screenshot = controller.read()
    
    new_data = [screenshot, input_for_screenshot, [speed, steering_angle]]

    #Determine fps
    fps = time_to_loop(loop_time)
    
    if target_fps != None:
        #Force program to wait until we are beneath the target_fps
        while fps > target_fps:
            fps = time_to_loop(loop_time)
        
    #print(fps)
    return new_data

def save_data(data):
    print('Saving...')
    save_time = time()
    #Make directory structure
    os.mkdir(f'data\\{save_time}')
    os.mkdir(f'data\\{save_time}\\img')
    os.mkdir(f'data\\{save_time}\\input')
    os.mkdir(f'data\\{save_time}\\telemetry')
    #Save each data point we've collected
    for point in range(len(data)):
        #img_to_save = cv.cvtColor(data[point][0], cv.COLOR_BGR2RGB)
        img_to_save = data[point][0]
        input_to_save = data[point][1]
        telemetry_to_save = data[point][2]
        filename = f'{save_time}_{point}'
        
        Image.fromarray(img_to_save).resize([640, 360]).save(f'data\\{save_time}\\img\\{filename}.jpg')
        np.save(f'data\\{save_time}\\input\\{filename}', np.array(input_to_save))
        np.save(f'data\\{save_time}\\telemetry\\{filename}', np.array(telemetry_to_save))
        
    print('Data collection successful!')

if __name__ == "__main__":
    #Start our non-blocking key listener
    listener = keyboard.Listener(
        on_release=on_release)
    listener.start()
    
    assettoReader = AssettoCorsaData()
    assettoReader.start()
    
    #The combined data--both images and key presses--that we will collect with
    #this script
    data = []
    
    start_time = time()
    
    print('Recording...')
    while listener.running:
        new_data = sample_data(assettoReader)
        data.append(new_data)
        
        if time()-start_time >= 20: 
            save_data(data)
            start_time = time()
            data = []
            print('Recording...')
            
    if len(data) > 0:
        save_data(data)