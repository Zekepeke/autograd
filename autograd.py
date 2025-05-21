import math

class Node:
    def __init__(self, data, children=(), operation='', label=''):
        self.data = data
        self.grad = 0.0
        self.backward = lambda: None
        self.prev = set(children)
        self.op = operation
        self.label = label
        
    def __add__(self, other):
        out = Node(self.data + other.data, (self, other), '+')
    
        def back_prop():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        # Storing the function in the backward
        out.backward = back_prop
    
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        out = Node(self.data * other.data, (self, other), '*')

        def back_prop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        # Storing the function in the backward
        out.backward = back_prop

        return out
    def __pow__(self, other):
        return Node(self.data ** other.data, (self, other), '**')
    def __truediv__(self, other):
        return Node(self.data / other.data, (self, other), '/')
    
    def tanh(self):
        x = self.data
        # Definition of tanh
        tan = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Node(tan, (self, ), 'tanh')
        
        def back():
            self.grad += (1 - tan**2) * out.grad
        # Storing the function in the backward
        out.backward = back
        
        return out

    def backward(self):
        # Topological sort         
        top = []
        seen = set()
        
        # Recursively build the topological sort
        def build_topo(v):
            if v not in seen:
                seen.add(v)
                for child in v.prev:
                    build_topo(child)
                top.append(v)
        build_topo(self)
        self.grad = 1.0
        
        # Starting from the root node and going backwards
        for node in reversed(top):
            node.backward()
        

    
    # For pritning the node value, the children, and the operation
    def __repr__(self):
        return f"Node(data={self.data})"
    