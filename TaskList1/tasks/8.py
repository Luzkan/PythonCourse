rom = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
def r_to_a(r):
    # Start with number equal to first roman letter
    a = rom[r[0]]
    for i in range(1, len(r)):
        # Check if current letter is a subtraction from another (like in IV = 4, where I subtracts from V)
        if rom[r[i]] > rom[r[i - 1]]:
            a += rom[r[i]] - 2 * rom[r[i - 1]]
        else:
            a += rom[r[i]]
    return a
print(r_to_a('MMLXXVII'))