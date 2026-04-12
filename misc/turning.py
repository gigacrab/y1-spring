import pigpio
import time
import sys

pwm_freq = 100

# time to complete one rev at max speed
T_360 = 1.7
offset = 0.1

IN1, IN2, ENA = 27, 22, 18
IN3, IN4, ENB = 23, 24, 19

# __main__ is the script that was passed to execute
if __name__ == "__main__":
    if len(sys.argv) >= 6:
        left_speed = float(sys.argv[1])
        right_speed = float(sys.argv[2])
        dir = sys.argv[3]
        angle = float(sys.argv[4])
        offset = float(sys.argv[5])
        if len(sys.argv) == 7:
            T_360 = float(sys.argv[6])
    else:
        raise Exception("Didn't input appropriate variables")

pi = pigpio.pi()

pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)
pi.set_mode(ENA, pigpio.OUTPUT)

pi.set_mode(IN3, pigpio.OUTPUT)
pi.set_mode(IN4, pigpio.OUTPUT)
pi.set_mode(ENB, pigpio.OUTPUT)

pi.set_PWM_frequency(ENA, pwm_freq)
pi.set_PWM_frequency(ENB, pwm_freq)
print(f"ENA frequency: {pi.get_PWM_frequency(ENA)}")
print(f"ENB frequency: {pi.get_PWM_frequency(ENB)}")

def move(a, b):
    pi.write(IN1, a > 0)
    pi.write(IN2, a < 0)
    pi.write(IN3, b > 0)
    pi.write(IN4, b < 0)
    pi.set_PWM_dutycycle(ENA, int(abs(a) * 255))
    pi.set_PWM_dutycycle(ENB, int(abs(b) * 255))
        
def turn(T_360, a, b, angle, dir, offset):
    speed = (a + b) / 2
    turn_time = (0.79 / speed) * T_360 * (angle / 360) + offset
    print(f"The turn time is: {turn_time}")
    if (dir.lower() == "l"):
        move(-a, b)
    else:   
        move(a, -b)
    time.sleep(turn_time)
    move(0, 0)

start_time = time.perf_counter()

try:
    turn(T_360, left_speed, right_speed, angle, dir, offset)
except KeyboardInterrupt:
    print("It's stopped")
    move(0, 0, pwm_freq)
    pi.stop()
