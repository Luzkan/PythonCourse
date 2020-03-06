def primes(n):
    # Range starts with 3 and only needs to go up the squareroot of n (for all odd numbers)
    primelist = lambda n : [x for x in range(2, n) if not 0 in map(lambda z : x % z, range(2, int(x**0.5+1)))]
    print(primelist(n))
primes(6)