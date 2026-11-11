import torch
from cliffordnet.ga_ops import dot_gate, wedge
def make_streams(seed=0):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(2,8,4,4, generator = g) #(B, C, H, W)
    c = torch.randn(2,8,4,4, generator = g)
    return h, c
def test_wedge_is_antisymmetric():
    """u^v = - v^u -- the defining property of oriented area"""
    h, c = make_streams()
    assert torch.allclose(wedge(h,c,1), -wedge(c,h,1))

def test_wedge_of_parallel_streams_is_zero():
    """u^u= 0  - a vector spans no area itself
    """
    h, _ = make_streams()
    assert torch.allclose(wedge(h,h,3), torch.zeros_like(h))

def test_shapes_are_preserved():
    h, c = make_streams()
    assert dot_gate(h, c).shape == h.shape
    assert wedge(h,c,1).shape == h.shape
def test_dot_gate_suppresses_opposition():
    """Strongly opposing inputs should give a strongly
    negative pre-gate value - SiLU squashes it toward zero
    """
    h = torch.ones(1,4,1,1)
    c= -10 * torch.ones(1,4,1,1)
    assert (dot_gate(h, c).abs() < 1e-3).all()

