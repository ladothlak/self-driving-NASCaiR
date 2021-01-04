from pynput import keyboard
import numpy as np
from time import time
from PIL import Image
import os
from screenshot import take_screenshot

window = 'Assetto Corsa'

#The framerate we will limit data collection to
target_fps = 15
sequence_length = 60

#The current input w a s d
cur_input = [0, 0, 0, 0]

#Hash table for quick keypress lookup
switcher = {
    'w':0,
    'a':1,
    's':2,
    'd':3
    }

def time_to_loop(loop_time, show_fps = False):
    fps = 1 / (time() - loop_time)
    loop_time = time()
    
    if show_fps:
        print('FPS {}'.format(fps))
        
    return fps
    

def on_press(key):
    try:      
        result = switcher.get(key.char, '')
        if (result != ''):
            cur_input[result] = 1
            
    except AttributeError:
        pass

def on_release(key):
    try:
        result = switcher.get(key.char, '')
        if (result != ''):
            cur_input[result] = 0

    except:
        pass
    
    if key == keyboard.Key.f10:
        # Stop listener
        return False



def sample_data(input_data=None, target_fps=15):
    #As long as the listener is running, continue taking screenshots
    loop_time = time()
    
    #Collect screenshot
    screenshot = np.array(take_screenshot(window))
    
    if input_data == None:
        #Collect input at that time
        input_for_screenshot = cur_input.copy()
    else:
        input_for_screenshot = input_data
    
    new_data = [screenshot, input_for_screenshot]

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
    #Save each data point we've collected
    for point in range(len(data)):
        #img_to_save = cv.cvtColor(data[point][0], cv.COLOR_BGR2RGB)
        img_to_save = data[point][0]
        input_to_save = data[point][1]
        filename = f'{save_time}_{point}'
        
        Image.fromarray(img_to_save).resize([640, 360]).save(f'data\\{save_time}\\img\\{filename}.jpg')
        np.save(f'data\\{save_time}\\input\\{filename}', np.array(input_to_save))
        
    print('Data collection successful!')

if __name__ == "__main__":
    #Start our non-blocking key listener
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    
    #The combined data--both images and key presses--that we will collect with
    #this script
    data = []
    
    start_time = time()
    
    print('Recording...')
    while listener.running:
        new_data = sample_data()
        data.append(new_data)
        
        if time()-start_time >= 20: 
            save_data(data)
            start_time = time()
            data = []
            print('Recording...')
            
    if len(data) > 0:
        save_data(data)