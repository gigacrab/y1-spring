import pigpio

pwm_freq = 100

IN1, IN2, ENA = 27, 22, 18
IN3, IN4, ENB = 23, 24, 19

pi = pigpio.pi()

pi.set_mode(IN1, pigpio.OUTPUT)
pi.set_mode(IN2, pigpio.OUTPUT)
pi.set_mode(ENA, pigpio.OUTPUT)

pi.set_mode(IN3, pigpio.OUTPUT)
pi.set_mode(IN4, pigpio.OUTPUT)
pi.set_mode(ENB, pigpio.OUTPUT)

pi.set_PWM_frequency(ENA, pwm_freq)
pi.set_PWM_frequency(ENB, pwm_freq)
#print(f"ENA frequency: {pi.get_PWM_frequency(ENA)}")
#print(f"ENB frequency: {pi.get_PWM_frequency(ENB)}")

def move(r, l):
    pi.write(IN1, r > 0)
    pi.write(IN2, r < 0)
    pi.write(IN3, l > 0)
    pi.write(IN4, l < 0)
    pi.set_PWM_dutycycle(ENA, int(abs(r) * 255))
    pi.set_PWM_dutycycle(ENB, int(abs(l) * 255))
