# utils.py
import config

def log_message(message):
    """Prints to console and appends to the log file."""
    print(message)
    with open(config.LOG_FILE, 'a') as f:
        f.write(message + '\n')

def clear_log():
    """Clears the log file at the start of a run."""
    with open(config.LOG_FILE, 'w') as f:
        f.write("=== Experiment Log ===\n")