from inspect import getfullargspec
from collections import defaultdict
import math

class Overload:
    def __init__(self):
        # Create an empty dictionary[function_name][arguments_num] for all
        # versions of the given function
        self.vers = defaultdict(dict)

    def __call__(self, func):
        # Fill the dictionary
        self.vers[func.__name__][len(getfullargspec(func).args)] = func

        # Return the result from dict
        def func_wrapper(*args, **kwargs):
            return self.vers[func.__name__][len(args)](*args, **kwargs)

        return func_wrapper

overload = Overload()

@overload
def norm(x,y):
    return math.sqrt(x*x + y*y)

@overload
def norm(x,y,z):
    return abs(x) + abs(y) + abs(z)

print(f"norm(2,4) = {norm(2,4)}")
print(f"norm(2,3,4) = {norm(2,3,4)}")