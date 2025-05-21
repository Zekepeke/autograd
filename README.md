#Autograd
A class of mathical expression
scalar valued auto grad engine 


children is a tuple for convenience but maintaining it into a set, used for keeping track of pointers for what values produce what other values previously



operation is used for keeping track of the mathematical operation that created the new value of the children as we know the value of every single child but not the operation used to create a new value

backward function is the thing that does the piece of chain rule at each node that takes inputs and produces an output 