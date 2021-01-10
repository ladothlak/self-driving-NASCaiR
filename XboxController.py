# Taken from TensorKart project utils here: https://github.com/kevinhughes27/TensorKart/blob/master/utils.py

from inputs import get_gamepad
import math
import threading


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self.vjoy_const = 32768

        self._is_running = True

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller, args=())
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def __del__(self):
        self.stop()

    def stop(self):
        self._is_running = False
        self._monitor_thread.join()
        print('Gamepad monitor thread ended')

    def write(self, j):
        # Throttle, Steering, Brakes
        j.data.wAxisX, j.data.wAxisY, j.data.wAxisZ = int(
            self.RightTrigger * self.vjoy_const), int(self.LeftJoystickX * self.vjoy_const), int(self.LeftTrigger * self.vjoy_const)
        j.update()

        return [self.LeftJoystickX, self.RightTrigger, self.LeftTrigger]

    def read(self):
        return [self.LeftJoystickX, self.RightTrigger, self.LeftTrigger]

    def _monitor_controller(self):
        while self._is_running:
            events = get_gamepad()
            for event in events:
                if event.code == 'ABS_X':
                    # normalize between 0 and 1
                    self.LeftJoystickX = (
                        (event.state / XboxController.MAX_JOY_VAL) + 1) / 2
                elif event.code == 'ABS_Z':
                    self.LeftTrigger = event.state / \
                        XboxController.MAX_TRIG_VAL  # normalize between 0 and 1
                elif event.code == 'ABS_RZ':
                    self.RightTrigger = event.state / \
                        XboxController.MAX_TRIG_VAL  # normalize between 0 and 1


if __name__ == '__main__':
    import pyvjoy
    j = pyvjoy.VJoyDevice(1)
    controller = XboxController()
    test = []
    while True:
        test.append(controller.write(j))
