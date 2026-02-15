from rpi_ws281x import PixelStrip, Color
import sys

mode = sys.argv[1]
bri = float(sys.argv[2])

# LED strip configuration:
LED_COUNT = 8           # Number of LED pixels
LED_PIN = 18            # GPIO pin (must support PWM)
LED_FREQ_HZ = 800000    # LED signal frequency in Hz
LED_DMA = 10            # DMA channel
LED_BRIGHTNESS = int(bri * 255)  # 0-255
LED_INVERT = False
LED_CHANNEL = 0

# Create NeoPixel object
pixels = PixelStrip(LED_COUNT, LED_PIN, LED_FREQ_HZ, LED_DMA, LED_INVERT, LED_BRIGHTNESS, LED_CHANNEL)
pixels.begin()

if mode == "0":
    for i in range(8):
        pixels.setPixelColor(i, Color(0, 0, 0))  # White (GRB order)
    pixels.show()

elif mode == "1":
    for i in range(8):
        pixels.setPixelColor(i, Color(255, 255, 255))  # White (GRB order)
    pixels.show()

elif mode == "2":
    for i in range(8):
        pixels.setPixelColor(i, Color(255, 0, 0))  # White (GRB order)
    pixels.show()
