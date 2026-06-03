import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DifferentiableUR3FK(nn.Module):
	"""Differentiable analytical FK layer for UR3 using standard DH parameters."""

	def __init__(self) -> None:
		super().__init__()
		self.register_buffer(
			"alpha",
			torch.tensor([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0], dtype=torch.float32),
		)
		self.register_buffer(
			"a",
			torch.tensor([0.0, -0.24365, -0.21325, 0.0, 0.0, 0.0], dtype=torch.float32),
		)
		self.register_buffer(
			"d",
			torch.tensor([0.1519, 0.0, 0.0, 0.11235, 0.08535, 0.0819], dtype=torch.float32),
		)

	def forward(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
		if q.ndim != 2 or q.size(1) != 6:
			raise ValueError("Expected q with shape [N, 6].")

		batch_size = q.size(0)
		device = q.device
		dtype = q.dtype

		# Base-frame alignment with the same convention used in notebook experiments.
		T = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(batch_size, 1, 1)
		T[:, 0, 0] = -1.0
		T[:, 1, 1] = -1.0

		for i in range(6):
			theta = q[:, i]
			alpha = self.alpha[i].to(dtype=dtype)
			a = self.a[i].to(dtype=dtype)
			d = self.d[i].to(dtype=dtype)

			ct = torch.cos(theta)
			st = torch.sin(theta)
			ca = torch.cos(alpha)
			sa = torch.sin(alpha)

			T_i = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)

			T_i[:, 0, 0] = ct
			T_i[:, 0, 1] = -st * ca
			T_i[:, 0, 2] = st * sa
			T_i[:, 0, 3] = a * ct

			T_i[:, 1, 0] = st
			T_i[:, 1, 1] = ct * ca
			T_i[:, 1, 2] = -ct * sa
			T_i[:, 1, 3] = a * st

			T_i[:, 2, 1] = sa
			T_i[:, 2, 2] = ca
			T_i[:, 2, 3] = d

			T_i[:, 3, 3] = 1.0

			T = torch.bmm(T, T_i)

		T_tool = torch.zeros(batch_size, 4, 4, device=device, dtype=dtype)
		T_tool[:, 0, 1] = -1.0
		T_tool[:, 1, 2] = -1.0
		T_tool[:, 2, 0] = 1.0
		T_tool[:, 3, 3] = 1.0

		T = torch.bmm(T, T_tool)
		pos = T[:, :3, 3]
		rot = T[:, :3, :3]
		return pos, rot


def rotmat_to_quat(rot: torch.Tensor) -> torch.Tensor:
	"""Convert rotation matrices [N,3,3] to normalized quaternions [N,4] (qw,qx,qy,qz)."""
	if rot.ndim != 3 or rot.shape[1:] != (3, 3):
		raise ValueError("Expected rot with shape [N, 3, 3].")

	m00 = rot[:, 0, 0]
	m01 = rot[:, 0, 1]
	m02 = rot[:, 0, 2]
	m10 = rot[:, 1, 0]
	m11 = rot[:, 1, 1]
	m12 = rot[:, 1, 2]
	m20 = rot[:, 2, 0]
	m21 = rot[:, 2, 1]
	m22 = rot[:, 2, 2]

	qw = torch.sqrt(torch.clamp(1.0 + m00 + m11 + m22, min=1e-12)) * 0.5
	qx = torch.sqrt(torch.clamp(1.0 + m00 - m11 - m22, min=1e-12)) * 0.5
	qy = torch.sqrt(torch.clamp(1.0 - m00 + m11 - m22, min=1e-12)) * 0.5
	qz = torch.sqrt(torch.clamp(1.0 - m00 - m11 + m22, min=1e-12)) * 0.5

	qx = torch.copysign(qx, m21 - m12)
	qy = torch.copysign(qy, m02 - m20)
	qz = torch.copysign(qz, m10 - m01)

	quat = torch.stack([qw, qx, qy, qz], dim=1)
	quat = F.normalize(quat, p=2, dim=1)

	# Canonicalize sign to avoid equivalent +/- quaternion duplicates in the dataset.
	flip_mask = quat[:, :1] < 0
	quat = torch.where(flip_mask, -quat, quat)
	return quat

def quat_to_rot_matrix(quat):
    """
    Differentiable conversion from quaternion [qw, qx, qy, qz] to a 3x3 Rotation Matrix.
    """
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    R = torch.zeros((quat.size(0), 3, 3), device=quat.device)

    R[:, 0, 0] = 1.0 - 2.0 * (y**2 + z**2)
    R[:, 0, 1] = 2.0 * (x*y - z*w)
    R[:, 0, 2] = 2.0 * (x*z + y*w)

    R[:, 1, 0] = 2.0 * (x*y + z*w)
    R[:, 1, 1] = 1.0 - 2.0 * (x**2 + z**2)
    R[:, 1, 2] = 2.0 * (y*z - x*w)

    R[:, 2, 0] = 2.0 * (x*z - y*w)
    R[:, 2, 1] = 2.0 * (y*z + x*w)
    R[:, 2, 2] = 1.0 - 2.0 * (x**2 + y**2)

    return R