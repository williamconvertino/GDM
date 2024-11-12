import time

def get_time_remaining(start_time, step, num_steps):
    step = max(1, step)
    elapsed_time = time.time() - start_time
    steps_remaining = num_steps - step
    time_per_step = elapsed_time / step
    time_remaining = steps_remaining * time_per_step
    time_remaining_formatted = time.strftime("%H:%M:%S", time.gmtime(time_remaining))
    return time_remaining_formatted