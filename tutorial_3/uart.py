#!/usr/bin/env python3
import serial
import sys
import platform

speed = 115200
dev = serial.Serial("/dev/ttyUSB0", speed)
#Other common names: ttyACM0, tty.usbserial-0001

print("> Returned data:", file=sys.stderr)

while True:
    x = dev.read()
    sys.stdout.buffer.write(x)
    sys.stdout.flush()
