def sum_size(path):
    with open(path, 'r') as f:
        return sum(int(line.split(' ')[-1]) for line in f)

print("Całkowita liczba bajtów:", sum_size("test.txt"))