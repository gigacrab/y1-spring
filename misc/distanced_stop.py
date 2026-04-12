import pigpio
import time
import sys

pwm_freq = 800

# time to complete one rev at max speed
T_360 = 1.5

IN1, IN2, ENA = 27, 22, 18
IN3, IN4, ENB = 23, 24, 19

# __main__ is the script that was passed to execute
if __name__ == "__main__":
    if len(sys.argv) > 3:
        left_speed = float(sys.argv[1])
        right_speed = float(sys.argv[2])
        distance = float(sys.argv[3])
        if len(sys.argv) == 5:
            pwm_freq = int(sys.argv[4])
        
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

def move(a, b, f):
    pi.write(IN1, a > 0)
    pi.write(IN2, a < 0)
    pi.write(IN3, b > 0)
    pi.write(IN4, b < 0)
    pi.set_PWM_dutycycle(ENA, int(abs(a) * 255))
    pi.set_PWM_dutycycle(ENB, int(abs(b) * 255))
        
def turn(speed, angle, dir):
    turn_time = (1 / speed) * T_360 * (angle / 360)
    if (dir.lower() == "left"):
        move(-left_speed, -right_speed, pwm_freq)
    else:   
        move(left_speed, -right_speed, pwm_freq)
    time.sleep(turn_time)
    move(0, 0, pwm_freq)

average_pwm = (left_speed + right_speed) / 2 * 100
speed = 2.22 + (1.07 * average_pwm) + (-4.42E-03 * (average_pwm)**2) / 1.1
run_time = distance / speed

start_time = time.perf_counter()

try:
    move(left_speed, right_speed, pwm_freq)
    time.sleep(run_time)
    print(f"Time elapsed: {run_time}")
    print("It's stopped")
    move(0, 0, pwm_freq)
    pi.stop()
except KeyboardInterrupt:
    end_time = time.perf_counter()
    print(f"Time elapsed: {end_time - start_time}")
    print("It's stopped")
    move(0, 0, pwm_freq)
    pi.stop()
