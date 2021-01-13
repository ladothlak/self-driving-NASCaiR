import cv2 as cv
import numpy as np
import torch
import os
import win32com.client
import pyvjoy

from time import time, sleep
from screenshot import take_screenshot
from torchvision import transforms
from pynput import keyboard
from assetto_corsa_telemetry_reader import AssettoCorsaData

# Change the working directory to the folder this script is in.
# Doing this because I'll be putting the files from each video in their own folder on GitHub
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Utility functions


class DaiLE():

    def __init__(self, model_path, controller=None, fps_target=15, window='Assetto Corsa', debug_mode=True, lightweight_mode=False):
        # Get window to control
        self.window = window
        self.fps_target = fps_target
        self.debug_mode = debug_mode
        # If true will disable draw_cv_window
        self.lightweight_mode = lightweight_mode

        # Initialize assettoReader object
        self.assettoReader = AssettoCorsaData()
        self.assettoReader.start()

        # Characters to be used with draw_cv_window
        self.switcher = {0: '<', 1: '^', 2: '_'}

        # Initialize vjoy variables
        if controller == None:
            self.j = pyvjoy.VJoyDevice(1)
        else:
            self.j = controller

        self.vjoy_const = 32767

        # Initialize vjoy state
        self.j.data.wAxisX = 0
        self.j.data.wAxisY = 32767//2
        self.j.data.wAxisZ = 0
        self.j.update()

        # Set model to use in eval mode
        self.model = torch.load(model_path).eval()

        # Initialize model hidden layer
        self.hidden = None

    def __del__(self):
        self.j.data.wAxisX = 0
        self.j.data.wAxisY = 0
        self.j.data.wAxisZ = 0
        self.j.update()

        try:
            cv.destroyAllWindows()
        except:
            pass

        torch.cuda.empty_cache()

        print('DaiLE instance deleted')

    def _diagnose_gamepad(self):
        num = self.vjoy_const

        while num > 0:
            self.j.data.wAxisX = num
            # Steering
            self.j.data.wAxisY = num
            # Brakes
            self.j.data.wAxisZ = num
            self.j.update()
            sleep(0.1)
            num -= 1000
            print(num)

        while num <= 32768:
            # Throttle
            self.j.data.wAxisX = num
            # Steering
            self.j.data.wAxisY = num
            # Brakes
            self.j.data.wAxisZ = num
            self.j.update()
            sleep(0.1)
            num += 1000
            print(num)

    def _return_controller(self):
        return self.j

    def time_to_loop(self, loop_time, show_fps=False):
        fps = 1 / (time() - loop_time)

        if show_fps:
            print('FPS {}'.format(fps))

        return fps

    def get_predictions(self, img, tel, hidden):
        # Get model prediction
        model_inputs, self.hidden = self.model.predict(img, tel, hidden)

        # Take predicted input and convert back to list we can work with here
        model_inputs = model_inputs.detach().cpu().numpy()[0][0]

        return model_inputs

    def input_network_recommendation(self, model_inputs):
        client = win32com.client.Dispatch("WScript.Shell")
        client.AppActivate(self.window)

        steering = max(int(((model_inputs[0])*self.vjoy_const)), 0)
        throttle = max(int(model_inputs[1]*self.vjoy_const), 0)
        brakes = max(int(model_inputs[2]*self.vjoy_const), 0)

        if self.debug_mode:
            print(
                f'Steering: {steering}, Throttle: {throttle}, Brakes {brakes}')

        self.j.data.wAxisX = throttle
        self.j.data.wAxisZ = brakes
        self.j.data.wAxisY = steering
        self.j.update()

    def draw_cv_window(self, screenshot, logits, fps):
        img_window_name = 'DaiLE'
        font = cv.FONT_HERSHEY_SIMPLEX
        org_x, org_y = (250, 700)
        fontScale = 8
        color = (255, 0, 0)
        spacing = 200
        thickness = 12

        if logits[0] < 0.5:
            self.switcher[0] = '<'
        else:
            self.switcher[0] = '>'

        # Change screenshot to show DaiLE's decision
        for keypress in range(len(logits)):

            # Let the potential colors range from 255*[0.2, 1.0] so it is easy to
            # tell which way the model is favoring
            logit_to_color = min(abs(logits[keypress]-0.5)*2 + 0.2, 1)

            if logits[keypress] < 0.45:
                color = (255*logit_to_color, 0, 0)
            elif logits[keypress] > 0.55:
                color = (0, 255*logit_to_color, 0)
            else:
                color = (0, 0, 255)

            screenshot = cv.putText(screenshot, self.switcher.get(keypress),
                                    (org_x, org_y), font, fontScale, color,
                                    thickness, cv.LINE_AA)

            org_x += spacing

        screenshot = cv.putText(screenshot, str(round(fps)),
                                (100, 100), font, 6, (255, 0, 0),
                                thickness, cv.LINE_AA)

        # Make live plot showing DaiLE's vision
        screenshot = cv.resize(screenshot, (256, 180))
        screenshot = cv.cvtColor(screenshot, cv.COLOR_BGR2RGB)
        cv.imshow(img_window_name, screenshot)
        cv.moveWindow(img_window_name, 1500, 700)
        cv.waitKey(1)

    def prep_DaiLE_img(self, screenshot):
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

    def run_action_loop(self):
        loop_time = time()

        screenshot = np.array(take_screenshot(self.window))

        img = self.prep_DaiLE_img(screenshot)

        telemetry_data = self.assettoReader.getData()
        speed, steering_angle = telemetry_data.get(
            'speed')/300, telemetry_data.get('steerAngle')
        tel = torch.Tensor([speed, steering_angle]).unsqueeze(0).cuda()

        model_inputs = self.get_predictions(img, tel, self.hidden)

        # Enact model key presses
        self.input_network_recommendation(model_inputs)

        fps = self.time_to_loop(loop_time)

        if not self.lightweight_mode:
            self.draw_cv_window(screenshot, model_inputs, fps)

        if self.debug_mode and self.lightweight_mode:
            print(fps)

        if self.fps_target != None:
            # Limit loop to TARGET_FPS
            while fps > self.fps_target:
                fps = self.time_to_loop(loop_time)

        return model_inputs, screenshot, [speed, steering_angle]


if __name__ == '__main__':

    def on_press(key):
        if key == keyboard.Key.f10:
            # Stop listener
            return False

    # Start our non-blocking key listener
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    # Empty cuda cache in case we have any old stuff pinned in there
    torch.cuda.empty_cache()

    # Load in a model
    MODEL_NAME = 'trained_model_1610237203.9295883.obj'
    MODEL_PATH = f'models\\{MODEL_NAME}'

    DaiLE = DaiLE(MODEL_PATH)

    try:
        while listener.running:
            DaiLE.run_action_loop()
            # DaiLE.diagnose_gamepad()
    except:
        del DaiLE

    print('Done.')

    torch.cuda.empty_cache()
