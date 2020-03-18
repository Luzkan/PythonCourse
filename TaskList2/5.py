import sys
import random
from random import randrange
from random import getrandbits

# Greatest Common Divisor
def gcd(a, b):
    if (b == 0):
        return a
    else:
        return gcd(b, a % b)

# Performs the extended Euclidean algorithm (ax + by = gcd(a, b))
def extendedgcd(a, b):
    x, old_x = 0, 1
    y, old_y = 1, 0
    while (b != 0):
        quotient = a // b
        a, b = b, a - quotient * b
        old_x, x = x, old_x - quotient * x
        old_y, y = y, old_y - quotient * y
    return old_x

def chooseE(phi):
    # Retrieves a random number from range 1 < e < phi
    # Checks if it's gcd(e, phi) = 1
    while (True):
        e = random.randrange(2, phi)
        if (gcd(e, phi) == 1):
            return e

def is_prime(n, num_of_tests):
    # Quick check if its not even (reminder: '2' is prime)
    if n == 2 or n == 3:
        return True
    if n <= 1 or n % 2 == 0:
        return False

    # Miller-Rabin
    #   Goal: Find nontrivial square roots of 1 mod n
    #   We look for 'r' and 's' such that (n-1) = r * (2^s), with r - odd
    #       (Fermat Little Theorem: a^(n-1) = 1 mod n)
    #
    #   After that we pick 'a' from range [1, n-1] and:
    #       If (a^r != 1 mod n) and (a^((2^j)*r) != -1 (mod n))
    #       for all j such that 0 <= j <= s-1 then n is not prime
    #
    #       If (f a^r = 1 (mod n)) or (a^((2^j)r) = -1 (mod n))
    #       for some j such that 0 <= j <= s-1 then n is PSEUDO prime
    #       Which means it's good enough :3

    s = 0
    r = n - 1
    while r & 1 == 0:
        s += 1
        r //= 2

    for _ in range(num_of_tests):
        a = randrange(2, n - 1)
        x = pow(a, r, n)

        if x != 1 and x != n - 1:
            j = 1
            while j < s and x != n - 1:
                x = pow(x, 2, n)
                if x == 1:
                    return False
                j += 1
            if x != n - 1:
                return False

    return True

def generate_prime_candidate(length):
    p = getrandbits(length)
    # Mask to set Most Significant and Least Signifcant Bit to 1 (bitwise OR)
    p |= (1 << length - 1) | 1
    return p

def generate_prime_number(length):
    p = 4
    num_of_tests = 128
    while not is_prime(p, num_of_tests):
        p = generate_prime_candidate(length)
    return p

def generateKeys(bits):
    p = generate_prime_number(bits)
    q = generate_prime_number(bits)
    n = p * q
    phi = (p - 1) * (q - 1)
    e = chooseE(phi)

    # Compute d (1 < d < phi) such that e*d = 1 mod phi (d must be positive)
    if (extendedgcd(e, phi) < 0):
        d = extendedgcd(e, phi) + phi
    else:
        d = extendedgcd(e, phi)

    f_public = open('key.pub', 'w')
    f_public.write(str(n) + '\n' + str(e) + '\n')
    f_public.close()
    f_private = open('key.prv', 'w')
    f_private.write(str(n) + '\n' + str(d) + '\n')
    f_private.close()

def encrypt(message, file='key.pub', block_size=2):
    # User may want use someone elses public key
    # Check if the path for that key exist
    try:
        with open(file, 'r') as f:
            n = int(f.readline())
            e = int(f.readline())
            f.close()
    except FileNotFoundError:
        print("File not found.")
        exit(1)
    
    # Init cipher-to-ASCII as the first character of message
    cipher = ord(message[0])
    encrypted = []

    for i in range(1, len(message)):
        # Add cipher to the list if the max block size is reached
        if (i % block_size == 0):
            encrypted.append(cipher)
            cipher = 0

        # Shifting the digits over to the left by 3 places (*1000)
        #   - ASCII Codes: Max 3 digits in decimal
        cipher = cipher * 1000 + ord(message[i])

    # Add the last block to the list
    encrypted.append(cipher)

    # Encrypt the numbers
    for i in range(len(encrypted)):
        encrypted[i] = str((encrypted[i]**e) % n)
    return " ".join(encrypted)

def decrypt(blocks, block_size=2):
    with open ('key.prv', 'r') as f:
        n = int(f.readline())
        d = int(f.readline())
        f.close()

    # Create groups of ints out of the input
    cipher = [int(x.strip()) for x in blocks.split(' ')]
    
    # Converts each int in the list to block_size number of characters (int = 2 chars)
    msg = ""
    for i in range(len(cipher)):
        # Decrypt Message
        cipher[i] = (cipher[i]**d) % n
        
        # Getting ASCII codes for each character in a block
        tmp = ""
        for _ in range(block_size):
            tmp = chr(cipher[i] % 1000) + tmp
            cipher[i] //= 1000
        msg += tmp
    return msg

# I/O Handler
if __name__ == "__main__":
    modes = {'--gen-keys': generateKeys, '--encrypt': encrypt, '--decrypt': decrypt}
    if sys.argv[1] not in modes:
        print("Usage: 2.py [--gen-keys [bits] / --encrypt [input] /  --decrypt [input]")
        exit(1)
    if sys.argv[1] == "--gen-keys":
        print("Generating Keys...")
        generateKeys(int(sys.argv[2]))
        print("Success! Saved to key.pub and key.prv")
    elif sys.argv[1] == "--encrypt":
        print("Encrypting your message...")
        if sys.argv[2] == "--file":
            msg = (' '.join(sys.argv[4:])).replace('\n','')
            print(encrypt(msg, sys.argv[3]))
        else:
            msg = (' '.join(sys.argv[2:])).replace('\n','')
            print(encrypt(msg))
    elif sys.argv[1] == "--decrypt":
        print("Decrypting your message...")
        msg = (' '.join(sys.argv[2:])).replace('\n','')
        print(decrypt(msg))
    exit(1)