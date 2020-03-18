import os
import sys

def info(path):
    word_count = 0
    line_count = 0

    with open(path, 'r') as f:
        for line in f:
            words = line.split()
            word_count += len(words)
            line_count += 1

    print("Bytes:", os.path.getsize(path))
    print("Words:", word_count)
    print("Lines:", line_count)
    print("Longest Line:", len(max(open(path, 'r'), key=len)))


if __name__ == "__main__":
    path = sys.argv[1]
    info(path)