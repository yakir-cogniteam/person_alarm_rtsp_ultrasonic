from gpiozero import InputDevice, OutputDevice
from time import sleep

night_sensor = InputDevice(17)     
deterrence_leds = OutputDevice(27) 


import sys

mode = sys.argv[1]
try:

    if mode == "ON":
        if night_sensor.value == 0:
            print("Sensor: DARK 🌙  | Action: Turning LEDs ON!")
            deterrence_leds.on()

    elif mode =="OFF":
        deterrence_leds.off()

except KeyboardInterrupt:
    print("\nTest stopped. Turning off LEDs for safety...")
    deterrence_leds.off()