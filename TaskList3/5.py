table = [1, 2, 3]

def ps(l):
    if len(l) <= 0:   
        return [[]]
    else:
        # Asteriks "*" unpacks item, otherwise we would get [] cascade
        return list(map(lambda i: [l[0], *i], ps(l[1:]))) + ps(l[1:])

print(ps(table))
