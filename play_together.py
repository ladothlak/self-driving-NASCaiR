############################################################################
# Using this script, you should be able to pass off play between yourself  #
# and DaiLE. DaiLE will control the vehicle at random intervals, and then  #
# you will control the car for a random interval--back and forth forever.  #
# This way, we can collect data for areas where DaiLE still seems to be    #
# struggling (e.g. getting stuck in the grass or taking a corner too       #
# aggressively)                                                            #
############################################################################

#Import required functionality from other scripts

import DaiLE
import record_data
import torch
import cv2 as cv

from random import randint
from pynput import keyboard
from time import time

#Define lower and upper bounds of control time
TIME_LB = 5
TIME_UB = 10

#Hash table for quick keypress lookup
switcher = {
    'w':0,
    'a':1,
    's':2,
    'd':3
    }

cur_input = [0, 0, 0, 0]

def on_press(key):
    if key == keyboard.Key.f10:
        # Stop listener
        return False
    
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

if __name__ == "__main__":
    #Get window to control
    window = 'Assetto Corsa'
    
    #Start our key listener
    listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()
    
    #Create keyboard controller
    board = keyboard.Controller()
    
    #Load in a model
    MODEL_PATH = 'models\\trained_model_1609697904.8732946.obj'
    MODEL = torch.load(MODEL_PATH).eval()
    
    #Initialize hidden input
    hidden = None  
    data = []
    
    #while the listener is running
    try:
        while listener.running:
            #generate random number for time interval between 5-8 seconds
            time_to_control = randint(TIME_LB, TIME_UB)
            loop_start = time()
            warned = False
            
            print(f'You will have {time_to_control} seconds!')
            
            print('DaiLE\'s turn!')
            #give control to DaiLE
            while time()-loop_start < time_to_control:
                #Do DaiLE things (pressing keys, being scared of grass, etc.)
                hidden, _ = DaiLE.run_action_loop(MODEL, hidden, window)
                
                if ((time()-loop_start >= time_to_control-1) and warned == False):
                    warned = True
                    print('Get ready!')
                
            #Unpress any keys pressed by DaiLE
            for cur_key in ['w', 'a', 's', 'd']:
                board.release(cur_key)
                
            loop_start = time()
            print('Your turn!')
                
            #give control to player
            while time()-loop_start < time_to_control:
                #Keep on updating the hidden state for DaiLE so he doesn't forget
                #context when we hand control back over to him
                hidden, _ = DaiLE.run_action_loop(MODEL, hidden, window, fps_target=None, take_action=False)
                #Record player screen and inputs
                new_data = record_data.sample_data(cur_input, target_fps=None)
                data.append(new_data)
                
            record_data.save_data(data)
            data = []
                             
    except:
        pass
    
    cv.destroyAllWindows()
        
        