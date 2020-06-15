import sys

table = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"

def encode(txt):
    # Create binary string, ex:
    #   bin(ord(A)) = bin(65) = 0b1000001 -> lstrip -> 1000001 -> 01000001
    bit_s = ""
    for c in txt:
        bin_c = bin(ord(c)).lstrip("0b")
        bin_c = (8-len(bin_c))*"0" + bin_c
        bit_s += bin_c

    # Padding (groups of 6 needed to encode)
    while (((len(txt)) % 3) != 0):
        bit_s += "00000000"    
        txt += "0"
    
    # Split into 6-bit groups
    splits = [bit_s[i:i+6] for i in range(0, len(bit_s), 6)]

    #Encode the brackets
    b64_str = ""
    for split in splits:
        if split == "000000":
            b64_str += "="
        else:
            b64_str += table[int(split,2)]
    return b64_str

def decode(b64_txt):
    # Create binary string
    bit_s = ""
    for c in b64_txt:
        if c in table:
            bin_c = bin(table.index(c)).lstrip("0b")
            bin_c = (6-len(bin_c))*"0" + bin_c
            bit_s += bin_c
    
    # Split into 8-bit groups
    splits = [bit_s[i:i+8] for i in range(0, len(bit_s), 8)]

    # Decode char binary -> asciii
    txt = ""
    for bracket in splits:
        txt += chr(int(bracket,2))
    return txt

# I/O Handler
if __name__ == "__main__":
    modes = {'--encode': encode, '--decode': decode}
    if sys.argv[1] not in modes:
        print("Usage: 2.py [--encode / --decode] [input] [output]")
        exit(1)

    with open(sys.argv[2], "r") as f_in, open(sys.argv[3], "w") as f_out:
        f_out.write(modes[sys.argv[1]](f_in.read()))