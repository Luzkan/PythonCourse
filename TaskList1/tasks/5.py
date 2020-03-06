# Trailing zero is formed when a multiple of 5 is multiplied with a multiple of 2
# keep in mind that there are more twos than fives (always), so we just count 5's
def fraczero(n):
    if n not in range(0, 10000):
        return "N outside of range (0, 10000)"
    zeros = 0
    while n > 0:
        n //= 5
        zeros += n
    return zeros
print(fraczero(12)) # 12! == 479001600 -> answ: 2 