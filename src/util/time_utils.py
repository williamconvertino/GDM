import time

def get_time_remaining(start_time, step, num_steps):
    step = max(1, step)
    elapsed_time = time.time() - start_time
    steps_remaining = num_steps - step
    time_per_step = elapsed_time / step
    time_remaining = steps_remaining * time_per_step
    
    num_days = time_remaining // (24 * 3600)
    time_remaining = time_remaining % (24 * 3600)
    num_hours = time_remaining // 3600
    time_remaining %= 3600
    num_minutes = time_remaining // 60
    time_remaining %= 60
    num_seconds = time_remaining
    
    time_remaining_formatted = f'{int(num_seconds)}s'
    if num_minutes > 0:
        time_remaining_formatted = f'{int(num_minutes)}m ' + time_remaining_formatted
    if num_hours > 0:
        time_remaining_formatted = f'{int(num_hours)}h ' + time_remaining_formatted
    if num_days > 0:
        time_remaining_formatted = f'{int(num_days)}d ' + time_remaining_formatted
    
    return time_remaining_formatted