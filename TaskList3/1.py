matrix = ["1.1 2.2 3.3",
          "4.4 5.5 6.6",
          "7.7 8.8 9.9"]

# First we split the string
#   then get m[0, 1, 2][0] (k) into one line
#   then we group them back into string with a space
#   then repeat for m[k][0, 1, 2] (n) times
def trn(m): return [' '.join([n.split(' ') for n in matrix][k][n]
                             for k in range(len(matrix)))
                             for n in range(len(matrix))]

print(trn(matrix))
