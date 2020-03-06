def unique(l):
    seen = set()
    return [x for x in l if not (x in seen or seen.add(x))]
l = [1,1,2,2,2,3,3,5,5,5,4,4,4,0]
print(unique(l))