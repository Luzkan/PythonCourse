import os
import sys
import hashlib
from collections import defaultdict

def findDups(directory):
    uniqueSize = {}
    # Creates an arbitrary "callable" object (default item) to prevent KeyError
    couldDups = defaultdict(list)

    # Checking Size first is faster
    # If file with same filesize is detected, then we'll check hash
    def checkSize(root, items):
        for name in items:
            absPath = os.path.join(root, name)
            size = os.path.getsize(absPath)
            if size not in uniqueSize:
                uniqueSize[size] = absPath
            else:
                checkHash(absPath)
                checkHash(uniqueSize[size])

    # Check the hash for those items
    def checkHash(absPath):
        hmd5 = hashlib.md5()
        with open(absPath, 'rb') as file:
            for fb in iter(lambda: file.read(1024), b''):
                hmd5.update(fb)
        couldDups[hmd5.hexdigest()].append(absPath)

    for root, _, files in os.walk(directory, topdown=False):
        checkSize(root, files)

    # If two files have the same hash -> list them
    for md5, pathfile in ((md, pf) for md, pf in couldDups.items() if len(pf) >= 2):
        print("---------------------------------")
        print("Hash:", md5)

        # Organize into set, remove duplicate entries
        pflist = set(pathfile)
        for f in pflist:
            print(f)
    print("---------------------------------")

if __name__ == "__main__":
    path = sys.argv[1]
    findDups(path)
