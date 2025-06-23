import torch
from math import isclose

import sys
import os

## Getting the autograd engine by moving up a path then into the src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.autograd import Node
print(__file__)

def test_autograd_engine():
    # My implementation
    x = Node(2.0, label='x')
    y = Node(3.0, label='y')
    z = x * y + x ** 2 + y.relu()
    z.backward()

    # --- PyTorch ---
    xt = torch.tensor(2.0, requires_grad=True)
    yt = torch.tensor(3.0, requires_grad=True)
    zt = xt * yt + xt ** 2 + torch.relu(yt)
    zt.backward()

    # --- Compare ---
    assert isclose(z.data, zt.item(), rel_tol=1e-5), f"Forward failed: {z.data} vs {zt.item()}"
    assert isclose(x.grad, xt.grad.item(), rel_tol=1e-5), f"x.grad failed: {x.grad} vs {xt.grad.item()}"
    assert isclose(y.grad, yt.grad.item(), rel_tol=1e-5), f"y.grad failed: {y.grad} vs {yt.grad.item()}"

    print("Test passed")

test_autograd_engine()
