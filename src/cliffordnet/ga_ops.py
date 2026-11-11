"""Geometric interaction ops for CliffordNet - Cl decomposition in PyTorch.

Feature tensors are images: shape (B , C ,H ,W) = batch, channels , height, width.
The channel dimension (dim = 1) plays the rols of the vector components.
"""

import torch
import torch.nn.functional as F

def dot_gate(h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:

	"""
	Coherence gate : dot-product part of the geometric product. 
	Elementwise agreement between streams, passed through SiLU so the gate is smooth and trainable. Aligned features pass: opposing features are suppressed.
	"""
	return F.silu(h * c)


def wedge(h: torch.Tensor, c: torch.Tensor, shift: int) -> torch.Tensor:
	"""Wedge part: bivector coefficients h_i*c_{i+s} - c_i*h_{i+s}.
	torch.roll wraps channels around, so channel I meets channel i+s. The result is the signed 'oriented area' spanned by the two  streams in each channel plane -- large where the streams twist apart (edges!), zero where they're parallel.

	"""
	c_s = torch.roll(c, shifts = shift, dims = 1)
	h_s = torch.roll(h, shifts = shift, dims = 1)
	return h * c_s - c * h_s