
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
visited_dfs = []
def dfs(visited, tree, node):
    if node not in visited:
        visited.append(node)
        yield node
        if tree[1] != None:
            dfs(visited, tree[1], tree[1][0])
        if tree[2] != None:
            dfs(visited, tree[2], tree[2][0])

dfs_gen = dfs(visited_dfs, tree, 1)

# -- BFS --
visited_bfs = []
queue_bfs = []
def bfs(visited, tree, node):
  visited.append(node)
  queue_bfs.append(tree)

  while queue_bfs:
    tree = queue_bfs.pop(0) 
    if tree[1] != None:
        if tree[1][0] not in visited:
            visited.append(tree[1][0])
            yield tree[1][0]
            queue_bfs.append(tree[1])
    if tree[2] != None:
        if tree[2][0] not in visited:
            visited.append(tree[2][0])
            queue_bfs.append(tree[2])

bfs_gen = bfs(visited_bfs, tree, 1)
