# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 20:05:42 2021

@author: Josh Cardosi
"""
import pyvjoy
from time import sleep
from XboxController import XboxController

j = pyvjoy.VJoyDevice(1)
j_const = 32768

j.reset()
j.reset_buttons()
j.reset_data()
j.reset_povs()

test_self = False

if test_self:
    controller = XboxController()

    while 1:
        X_Axis, RT, LT = controller.read()
        throttle = min(max(int(RT*j_const), 0), j_const)
        steering = min(max(int(X_Axis*j_const), 0), j_const)
        brakes = min(max(int(LT*j_const), 0), j_const)

        # Throttle
        j.data.wAxisX = throttle
        # Steering
        j.data.wAxisY = steering
        # Brakes
        j.data.wAxisZ = brakes

        j.update()

else:
    num = j_const

    while num > 0:
        j.data.wAxisX = num
        # Steering
        j.data.wAxisY = num
        # Brakes
        j.data.wAxisZ = num
        j.update()
        sleep(0.1)
        num -= 1000
        print(num)

    while num <= 32768:
        # Throttle
        j.data.wAxisX = num
        # Steering
        j.data.wAxisY = num
        # Brakes
        j.data.wAxisZ = num
        j.update()
        sleep(0.1)
        num += 1000
        print(num)

    # Throttle
    j.data.wAxisX = 10000
    # Steering
    j.data.wAxisY = 32768//2
    # Brakes
    j.data.wAxisZ = 0
    j.update()

    j.reset()
    j.reset_buttons()
    j.reset_data()
    j.reset_povs()

# j.set_axis(pyvjoy.HID_USAGE_RZ, 0x1)
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
# j.set_axis(pyvjoy.HID_USAGE_X, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_X, 0x8000)

# Braking
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
# j.set_axis(pyvjoy.HID_USAGE_Y, 0x3000)

# #Steering
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x3000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x3000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x2000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x1)

# sleep(0.1)

# j.set_axis(pyvjoy.HID_USAGE_Z, 0x4000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x5000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x4000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x5000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x8000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x5000)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0x6000)
# sleep(0.1)
# j.set_axis(pyvjoy.HID_USAGE_Z, 0)
