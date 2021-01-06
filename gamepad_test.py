# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:05:42 2021

@author: Josh Cardosi
"""
import pyvjoy
from time import sleep
from XboxController import XboxController

controller = XboxController()

while 1:
    print(controller.read())

# j = pyvjoy.VJoyDevice(1)

#sleep(40)

# j.reset()
# j.reset_buttons()
# j.reset_data()
# j.reset_povs()

# sleep(2)
#Throttle
# j.set_axis(pyvjoy.HID_USAGE_X, 0x2000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x1)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x2000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x2000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x6000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x1)

#Braking
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x2000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x1000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x1)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x2000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x6000)

#Steering
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)



# j.set_axis(pyvjoy.HID_USAGE_Z, 0x4000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x5000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x4000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x5000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x5000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)