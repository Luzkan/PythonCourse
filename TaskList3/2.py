l = [[1, 2, ["a", 4, "b", 5, 5, 5]], [4, 5, 6 ], 7, [[9, [123, [[123]]]], 10]]

def flatten(l):
    # For Elements in the List
    for e in l:
        # If that Element is not a List
        if not isinstance(e, list):
            # Return it (& keep track where we are atm - "yield")
            yield e
        else:
            # Recursivelly call flatten on found list
            yield from flatten(e)

# Flatten is a generator
print(list(flatten(l)))
