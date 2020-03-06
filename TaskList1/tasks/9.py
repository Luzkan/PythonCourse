from math import *
x = input('Podaj x = ')
try:
    print(eval(x))
except Exception as ex:
    print(ex)