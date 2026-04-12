from gpiozero import Robot, Motor
from gpiozero.pins.pigpio import PiGPIOFactory
import time
import sys

pwm_freq = 800

if __name__ == "__main__":
    if len(sys.argv) > 2:
        left_speed = float(sys.argv[1])
        right_speed = float(sys.argv[2])
        if len(sys.argv) == 4:
            pwm_freq = int(sys.argv[3])
    else:
        raise Exception("Didn't input appropriate variables")

# pin definitions (BCM numbering)
IN1 = 27
IN2 = 22
ENA = 18

IN3 = 23
IN4 = 24
ENB = 19

# use pigpio for hardware PWM backend
factory = PiGPIOFactory()
pi = factory._connection
pi.set_PWM_frequency(ENA, pwm_freq)
pi.set_PWM_frequency(ENB, pwm_freq)

robot = Robot(
    left=Motor(forward=IN3, backward=IN4, enable=ENB, pin_factory=factory),
    right=Motor(forward=IN1, backward=IN2, enable=ENA, pin_factory=factory),
    pin_factory=factory
)

def turn(speed, angle, dir):
    # time to complete one rev at max speed
    t = 1.5
    turn_time = (1 / speed) * t * (angle / 360)
    if (dir.lower() == "left"):
        robot.left(speed=speed)
    else:   
        robot.right(speed=speed)
    time.sleep(turn_time)
    robot.stop()

#turn(1, 360, "left")
robot.value = (left_speed, right_speed)
time.sleep(13)
robot.stop()