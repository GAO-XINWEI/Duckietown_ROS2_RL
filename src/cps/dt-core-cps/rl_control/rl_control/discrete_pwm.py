"""Discrete forward-driving presets for a driver calibrated to PWM 120..255."""
MIN_PWM = 120
MAX_PWM = 255
# (left, right); conventional differential-drive directions.
TARGET_PWM = {0: (230, 255), 1: (255, 230), 2: (240, 240)}


def pwm_to_command(pwm):
    """Use the quantization-bin midpoint so floor() gives the target PWM."""
    if pwm == 0:
        return 0.0
    if not MIN_PWM <= pwm <= MAX_PWM:
        raise ValueError('PWM target outside calibrated range')
    if pwm == MAX_PWM:
        return 1.0
    return (pwm + 0.5 - MIN_PWM) / (MAX_PWM - MIN_PWM)


def discrete_wheel_commands(action_id):
    """Reject invalid IDs; never turn an unknown action into forward motion."""
    if action_id not in TARGET_PWM:
        raise ValueError('Unknown discrete action ID: {}'.format(action_id))
    left, right = TARGET_PWM[action_id]
    return pwm_to_command(left), pwm_to_command(right)
