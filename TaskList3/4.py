table = [5, 4.5, -5.5, 15.5, -15]

# filter(function, iterable):
# ~ (element for element in iterable if function(element))
#   (lt      for lt      in l[1:]    if lt < l[0])
# filter(lambda lt: lt < l[0], l[1:]):

def qsort(l):
    if len(l) <= 1: 
        return l
    return qsort(list(filter(lambda lt: lt < l[0], l[1:]))) + [l[0]] + qsort(
                 list(filter(lambda ge: ge >= l[0], l[1:])))

print(qsort(table))
