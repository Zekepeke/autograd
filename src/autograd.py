import math

class Node:
    def __init__(self, data, children=(), operation='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self.prev = set(children)
        self.op = operation
        self.label = label

     # Checking if the other is an instance of Node
    def wrap(self, other):

        if isinstance(other, Node):
            other = other
        else:
            other = Node(other)
        return other


    def __add__(self, other):
        other = self.wrap(other)

        out = Node(self.data + other.data, (self, other), '+')

        def back_prop():
            # accumulating the gradients for instances of the
            # same node and not overwriting
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        # Storing the function in the backward
        out._backward = back_prop
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = self.wrap(other)
        out = Node(self.data * other.data, (self, other), '*')

        def back_prop():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        # Storing the function in the backward
        out._backward = back_prop

        return out


    def __pow__(self, other):
        # Supporting integers and float values
        assert isinstance(other, (int, float))

        out = Node(self.data**other, (self,), f'pow of {other}')

        '''
        Find the chainrule for the value that is raised
        to the power of other, where other could be any number (ex: -1)
        where the power is to the power of some constant
        '''
        def back_prop():
            self.grad += other * (self.data**(other - 1)) * out.grad
        out._backward = back_prop
        return out
  
    def __truediv__(self, other):
        return self * other**-1

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def relu(self):
        x = self.data
        # Definition of ReLu
        relu = max(0, x)
        out = Node(relu, (self, ), 'reLu')

        def back():
            if out.data > 0:
                self.grad += out.grad
            else:
                self.grad += 0
        # Storing the function in the backward
        out._backward = back

        return out

    def tanh(self):
        x = self.data
        # Definition of tanh
        tan = (math.exp(2*x) - 1)/(math.exp(2*x) + 1)
        out = Node(tan, (self, ), 'tanh')

        def back():
            self.grad += (1 - tan**2) * out.grad
        # Storing the function in the backward
        out._backward = back

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
        # Starting with the root node being 1
        self.grad = 1.0

        # Starting from the root node and going backwards
        for node in reversed(top):
            node._backward()

    # Redirecting the operations
    # Example other + self, other * self, etc.
    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * other**-1

    def __rtruediv__(self, other):
        return other * self**-1



    # For pritning the node value, the children, and the operation
    def __repr__(self):
        return f"Node(data={self.data})"
