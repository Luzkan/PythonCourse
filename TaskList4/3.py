
from random import randint

def indexGenerator():
    n = 2
    while True:
        yield n
        n += 1

class Node(object):
    # Root is not left or right of a node
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data

    def insert(self, data, depth):
        if self.data:
            where_insert = randint(0, 1)
            if where_insert == 1:
                if self.left is None:
                    self.left = Node(data)
                    return depth
                else:
                    depth = self.left.insert(data, depth+1)
            else:
                if self.right is None:
                    self.right = Node(data)
                    return depth
                else:
                    depth = self.right.insert(data, depth+1)
        else:
            self.data = data
        return depth

    def dfs(self, root, res):
        if root:
            res.append(root.data)
            self.dfs(root.left, res)
            self.dfs(root.right, res)
        return res

    def bfs(self, root, res):
        queue = [root]
        while queue:
            root = queue.pop()
            if root:
                res.append(root.data)
                queue.append(root.left)
                queue.append(root.right)
        return res

def treeGen(height):
    index = indexGenerator()
    depth = 1

    # Root
    tree = Node(1)
    
    while depth < height:
        depth = tree.insert(next(index), 1)
    return tree

tree = treeGen(4)
print(tree.dfs(tree, []))
print(tree.bfs(tree, []))