class Node:
    def __init__(self, data):
        self.data = data
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        return Node(self.data + other.data)
    def __mul__(self, other):
        return Node(self.data * other.data)
    def __pow__(self, other):
        return Node(self.data ** other.data)
    def __truediv__(self, other):
        return Node(self.data / other.data)