
from random import randint

# ===== Tree Generation =====

def indexGenerator():
    n = 2
    while True:
        yield n
        n += 1

def treeGen(height):
    # Generate Index for our tree structure
    index = indexGenerator()

    # Starting tree
    tree = [1, None, None]

    # Get max depth in a tree to restart if height requirement was not 
    # satisfied. Notice: it's a filthy way to create a tree with given
    # height don't use it to create a tall tree
    depth = lambda L: isinstance(L, list) and max(map(depth, L))+1
    
    while depth(tree) <= height:
        tree = generateTree(tree, index, 1, height)
    return tree

def generateTree(tree, index, depth, height):
    for i, node in enumerate(tree):
        if node == None:
            if randint(0, 1) == 1:
                tree[i] = [next(index), None, None]
    
    if depth < height:
        if tree[1] != None:
            generateTree(tree[1], index, depth+1, height)
        if tree[2] != None:
            generateTree(tree[2], index, depth+1, height)

    return tree

# ===== Traversing Algorithms =====

tree = treeGen(4)
print(tree)

# -- DFS Generator --
def dfs(tree):
    if tree is not None:
        yield tree[0]
        yield from dfs(tree[1])
        yield from dfs(tree[2])

dfs_gen = dfs(tree)
print(f"DFS: {list(dfs_gen)}")

# -- BFS --
def bfs(tree):
    que = [tree]
    while len(que) > 0:
        curr = que.pop(0)
        if curr[1] is not None: que.append(curr[1])
        if curr[2] is not None: que.append(curr[2])
        yield curr[0]

bfs_gen = bfs(tree)
print(f"BFS: {list(bfs_gen)}")
