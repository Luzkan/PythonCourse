def pascal_triangle(n):
    row, y, prePrint = [1], [0], []
    for _ in range(n):
        prePrint += [row]
        row = [r+l for r, l in zip(row+y, y+row)]
    return prePrint

def pyramid(p):
    offset = len(p[-1])-1
    for a in p:
        print(' ' * offset, ' '.join([str(s) for s in a]))
        offset -= 1

pyramid(pascal_triangle(5))