import time

def wait_until(condition, timeout=1):
    """
    Re-evaluate `condition()` until it either returns true or we've waited
    longer than `timeout`.
    """
    slept_for = 0
    sleep_interval = 0.2

    while slept_for < timeout and not condition():
        time.sleep(sleep_interval)
        slept_for += sleep_interval

    if slept_for >= timeout:
        raise TimeoutError("Condition timed out")
