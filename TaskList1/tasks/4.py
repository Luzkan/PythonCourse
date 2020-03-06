from collections import Counter
def prime_factors(n):
    if n == 1: return []
    i, factors = 2, []
    while i * i <= n:
        if n % i:
            i += 1
        else:
            n //= i
            factors.append(i)
    factors.append(n)
    return Counter(factors).most_common()
print(prime_factors(360))