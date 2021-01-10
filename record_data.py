from pynput import keyboard
import numpy as np
import os
import threading

from time import time
from PIL import Image
from screenshot import take_screenshot
from assetto_corsa_telemetry_reader import AssettoCorsaData
from XboxController import XboxController


class data_recorder():
    def __init__(self, window, controller, target_fps=15, full_path='data', show_fps=False, async_sample=False):
        # The window to track
        self.window = window

        # The framerate we will limit data collection to
        self.target_fps = target_fps

        # The input device
        self.controller = controller

        # The relative path where we will store data
        self.full_path = 'data'

        self.show_fps = show_fps

        # Start asseto corsa telemetry reader
        self.assettoReader = AssettoCorsaData()
        self.assettoReader.start()

        self.async_sample = async_sample
        self._is_running = True

        # Keep track of when we took our last sample
        self.last_sample_time = time()

        if self.async_sample:
            self._sample_thread = threading.Thread(
                target=self.sample_data, args=())
            self._sample_thread.daemon = True
            self._sample_thread.start()

        self.speed = 0
        self.steering_angle = 0
        self.screenshot = []
        self.input_for_screenshot = [0, 0, 0]

    def __del__(self):
        self.stop()

    def stop(self):
        self._is_running = False
        self._sample_thread.join()

        print('Recorder thread closed')

    def time_to_loop(self, loop_time):
        fps = 1 / (time() - loop_time-0.0000001)

        if self.show_fps:
            print('FPS {}'.format(fps))

        return fps

    def package_data(self, scale_speed=False):
        if scale_speed:
            return [self.screenshot, self.input_for_screenshot, [self.speed, self.steering_angle]]
        else:
            return [self.screenshot, self.input_for_screenshot, [self.speed/300, self.steering_angle]]

    def sample_data(self, input_for_screenshot=[]):
        while self._is_running:
            # For asyncronous, wait until just about Nyquist frequency to try sampling again
            # For syncronous, just let sample_data do its thing
            if ((self.time_to_loop(self.last_sample_time) <= 2*self.target_fps+5) or not (self.async_sample)):
                # Reset our last sample time
                self.last_sample_time = time()

                telemetry_data = self.assettoReader.getData()

                self.speed, self.steering_angle = telemetry_data.get(
                    'speed')/300, telemetry_data.get('steerAngle')

                # Collect screenshot
                self.screenshot = np.array(take_screenshot(self.window))

                # If we weren't given data to attach to this data point, assume that we want the player's controller data
                if input_for_screenshot == []:
                    self.input_for_screenshot = self.controller.read()
                else:
                    self.input_for_screenshot = input_for_screenshot

                new_data = self.package_data()

                # Do this to emulate a do while loop. If user does not want async sampling, this will break the loop and return data as normal
                if self.async_sample == False:
                    break

        return new_data

    def save_data(self, data):
        print('Saving...')
        save_time = time()
        # Make directory structure
        os.mkdir(f'{self.full_path}\\{save_time}')
        os.mkdir(f'{self.full_path}\\{save_time}\\img')
        os.mkdir(f'{self.full_path}\\{save_time}\\input')
        os.mkdir(f'{self.full_path}\\{save_time}\\telemetry')
        # Save each data point we've collected
        for point in range(len(data)):
            # img_to_save = cv.cvtColor(data[point][0], cv.COLOR_BGR2RGB)
            img_to_save = data[point][0]
            input_to_save = data[point][1]
            telemetry_to_save = data[point][2]
            filename = f'{save_time}_{point}'

            Image.fromarray(img_to_save).resize([640, 360]).save(
                f'{self.full_path}\\{save_time}\\img\\{filename}.jpg')
            np.save(f'{self.full_path}\\{save_time}\\input\\{filename}',
                    np.array(input_to_save))
            np.save(f'{self.full_path}\\{save_time}\\telemetry\\{filename}',
                    np.array(telemetry_to_save))

        print('Data collection successful!')


if __name__ == "__main__":

    def on_release(key):
        if key == keyboard.Key.f10:
            # Stop listener
            return False

    # Start our non-blocking key listener
    listener = keyboard.Listener(
        on_release=on_release)
    listener.start()

    controller = XboxController()
    recorder = data_recorder('Assetto Corsa', controller)

    # The combined data--both images and key presses--that we will collect with
    # this script
    data = []

    start_time = time()

    print('Recording...')
    while listener.running:
        new_data = recorder.sample_data()
        data.append(new_data)

        if len(data) == 56:
            print("About to predict next frame")

        if len(data) >= 256:
            recorder.save_data(data)
            start_time = time()
            data = []
            print('Recording...')
