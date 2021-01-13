############################################################################
# Using this script, you should be able to pass off play between yourself  #
# and DaiLE. DaiLE will control the vehicle at random intervals, and then  #
# you will control the car for a random interval--back and forth forever.  #
# This way, we can collect data for areas where DaiLE still seems to be    #
# struggling (e.g. getting stuck in the grass or taking a corner too       #
# aggressively)                                                            #
############################################################################

# Import required functionality from other scripts

from DaiLE import DaiLE
import record_data
import sys
import pyvjoy
import numpy as np
import warnings

from pynput import keyboard
from time import time, sleep
from XboxController import XboxController
from assetto_corsa_telemetry_reader import AssettoCorsaData


def DaiLE_loop(model_path, controller, listener):
    DaiLE_obj = DaiLE(model_path, fps_target=None,
                      debug_mode=False, lightweight_mode=True)
    RT, LT = 0, 0
    data = []

    while ((RT <= 0.01) and (LT <= 0.01) and listener.running):
        # Get player controller inputs
        _, RT, LT = controller.read()

        # Do DaiLE things (pressing keys, being scared of grass, etc.)
        model_inputs, screenshot, telemetry = DaiLE_obj.run_action_loop()

        new_data = [screenshot, model_inputs, telemetry]

        data.append(new_data)

        if len(data) >= 60:
            data = data[-60:]

    return data, DaiLE_obj._return_controller()


def main_loop(listener, model_path):
    # Get window to control
    window = 'Assetto Corsa'

    # Create controller object to check state of player controller
    controller = XboxController()

    try:
        print('DaiLE\'s turn!')
        # give control to DaiLE

        data, j = DaiLE_loop(MODEL_PATH, controller, listener)

        print('Interrupting DaiLE!')

        recorder = record_data.data_recorder(window, controller, async_sample=True)
        loop_start = time()
        # give control to player
        while ((len(data) < 300) and listener.running):

            controller.write(j)

            fps = recorder.time_to_loop(loop_start)

            # Record player screen and inputs
            if fps <= 15:
                data.append(recorder.package_data(scale_speed=True))
                loop_start = time()

            if(len(data) == 256):
                print('Reset inputs now')

        j.data.wAxisX = 0
        j.data.wAxisY = 32767//2
        j.data.wAxisZ = 32767
        j.update()

        controller.stop()
        sleep(1)

        if not listener.running:
            return data[:256], recorder
        else:
            recorder.stop()
            del recorder
            return data[:256], ''

    except:
        print("Unexpected error:", sys.exc_info())

    try:
        j.data.wAxisX = 0
        j.data.wAxisY = 32767//2
        j.data.wAxisZ = 32767
        j.update()
    except:
        pass


if __name__ == "__main__":
    warnings.simplefilter("ignore")

    def on_press(key):
        if key == keyboard.Key.f10:
            # Stop listener
            return False
    # Start our key listener
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Load in a model
    MODEL_PATH = 'models\\trained_model_1610237203.9295883.obj'

    all_data = []
    while listener.running:
        data, recorder = main_loop(listener, MODEL_PATH)
        all_data.append(data)

    for collection_period in range(len(all_data)):
        if len(all_data[collection_period]) >= 256:
            recorder.save_data(all_data[collection_period])

    print('Stopping recorder')
    del recorder
