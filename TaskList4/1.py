import time

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print(f"Time to execute function {func.__name__}:", elapsed_time)
        return result
    return wrapper

@measure_time
def waitingFunction():
    time.sleep(2)

waitingFunction()