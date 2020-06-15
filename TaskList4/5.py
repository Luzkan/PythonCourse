from cmath import exp
from math import pi
import time
import random
import numpy as np


# I/O Handler
def writeResult(res):
    with open("fastbignum_benchmark.txt", 'w') as f_out:
        f_out.write(res+'\n')

# For measuring time (copy-paste from first task)
def measure_time(func):
    def wrapper():
        start_time = time.time()
        result = func()
        elapsed_time = time.time() - start_time
        print(f"Time to execute function {func.__name__}:", elapsed_time)
        res = str(str(func.__name__) + " " + str(elapsed_time))
        writeResult(res)
        return result
    return wrapper

# DFT/FFT
def dft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)

def idft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * (-k) * n / N)
    return np.dot(M, x)

def fft(x):
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    if np.log2(N) % 1 > 0:
        raise ValueError("must be a power of 2")
 
    N_min = min(N, 2)
    
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    while X.shape[0] < N:
            X_even = X[:, :int(X.shape[1] / 2)]
            X_odd = X[:, int(X.shape[1] / 2):]
            terms = np.exp(-1j * np.pi * np.arange(X.shape[0])
                            / X.shape[0])[:, None]
            X = np.vstack([X_even + terms * X_odd,
                        X_even - terms * X_odd])
    return X.ravel()

# Test if written function match numpy implementation
# x = np.random.random(1024)
x = [1, 2, 3, 4]
print("My own DFT Check with Numpy:", np.allclose(dft(x), np.fft.fft(x)))
print("My own FFT Check with Numpy:", np.allclose(fft(x), np.fft.fft(x)))

# Functions given in the tasklist
def omegaZaw(k,n):
    return exp(-2j*k*pi/n)

def dftZaw(x,n):
    return [sum(x[i]*omegaZaw(i*k,n) if i<len(x) else 0 for i in range(n)) for k in range(n)]

def idftZaw(x,n):
    return [int(round(sum(x[i]*omegaZaw(-i*k,n) if i<len(x) else 0 for i in range(n)).real)/n) for k in range(n)]


class FastBigNumNumpy:
    # Get list of ints out of string input
    def __init__(self, num):
        self.num = list(map(int, num))

    def __mul__(self, other):
        
        return self.num

def numpy_test(A, B):
    Xs = np.fft.fft(A)       # X Star
    Ys = np.fft.fft(B)       # Y Star

    Zs = Xs * Ys             # Z Star
    print("[NUMPY] a * b", Zs)

    Z = np.fft.ifft(Zs)
    print("[NUMPY] dft^-1(a * b)", Z)

    result = 0
    for i, t in enumerate(Z):
        result += t*10**i

    print("[NUMPY] as value:", result, "\n")

def zaw_test(A, B):
    Xs = dftZaw(A, len(A)*2)    # X Star
    Ys = dftZaw(B, len(B)*2)    # Y Star
    Zs = []                     # Z Star

    for i in range(len(Xs)):
        Zs.append(Xs[i] * Ys[i])            
    print("[ZAW] a * b \t", Zs)

    Z = idftZaw(Zs, len(A)*2)   # Z
    print("[ZAW] dft^-1(a * b)", Z)

    result = 0
    for i, t in enumerate(Z):
        result += t*10**i

    print("[ZAW] as value:", result, "\n")

def my_test(A, B):
    Xs = dft(A)    # X Star
    Ys = dft(B)    # Y Star

    Zs = Xs * Ys             # Z Star
    print("[OWN] a * b", Zs)

    Z = idft(Zs)   # Z
    print("[OWN] dft^-1(a * b)", Z)

    result = 0
    for i, t in enumerate(Z):
        result += t*10**i

    print("[OWN] as value:", result, "\n")

def test():
    # A = ''.join([random.choice("0123456789") for i in range(500)])
    # B = ''.join([random.choice("0123456789") for i in range(500)])
    Atest = [1, 2, 3]
    Btest = [4, 5, 6]

    a = 123
    b = 456 
    print("[Answ]", a*b, "\n")

    zaw_test(Atest, Btest)
    my_test(Atest, Btest)
    numpy_test(Atest, Btest)
    
  
test()
