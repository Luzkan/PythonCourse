import os
import sys

def tolower(directory):
    def rename_all(root, items):
        for name in items:
            os.rename(os.path.join(root, name), 
                      os.path.join(root, name.lower()))

    for root, dirs, files in os.walk(directory, topdown=False):
        rename_all(root, dirs)
        rename_all(root, files)

if __name__ == "__main__":
    path = sys.argv[1]
    tolower(path)