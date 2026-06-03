import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class KinematicsResidualBlock(nn.Module):
    """
    A foundational residual building block with LayerNorm and Skip Connections
    to preserve high-frequency coordinate data across deep layers.
    """
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.ln1(self.fc1(x)))
        out = self.ln2(self.fc2(out))
        out += residual  # Skip connection
        return F.relu(out)

class UR3ForwardModel(nn.Module):
    """
    Deep Residual Encoder: Maps 12D joint waves -> Outputs 7D Pose + 8D Posture Mode Logits.
    Increased hidden capacity to 512 units with nested skip layers.
    """
    def __init__(self, joint_dim=12, pos_dim=7, num_classes=8, hidden_dim=512):
        super().__init__()

        # Project low-dimensional inputs up to the high-capacity feature space
        self.input_projection = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Deep residual core configuration
        self.res_layers = nn.Sequential(
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim)
        )

        # Output bottlenecks
        self.feature_bottleneck = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU()
        )

        self.pos_head = nn.Linear(256, pos_dim)
        self.latent_head = nn.Linear(256, num_classes)

    def forward(self, joints_continuous):
        x = self.input_projection(joints_continuous)
        x = self.res_layers(x)
        features = self.feature_bottleneck(x)

        pose_pred = self.pos_head(features)

        # Maintain valid structural unit quaternions (indices 3 to 7)
        pos, quat = pose_pred[:, :3], pose_pred[:, 3:]
        quat = F.normalize(quat, p=2, dim=-1)
        pose_pred = torch.cat([pos, quat], dim=-1)

        z_logits = self.latent_head(features)
        return pose_pred, z_logits


class UR3InverseModel(nn.Module):
    """
    Deep Residual Decoder: Takes 7D target pose + 8D discrete posture one-hot -> Outputs 12D joint waves.
    Expanded network depth to handle complex multi-solution coordinate crossings.
    """
    def __init__(self, pos_dim=7, num_classes=8, joint_dim=12, hidden_dim=512):
        super().__init__()

        # Input layer handling joint embedding + condition concatenation
        self.input_projection = nn.Sequential(
            nn.Linear(pos_dim + num_classes, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Core geometric mapping network
        self.res_layers = nn.Sequential(
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim),
            KinematicsResidualBlock(hidden_dim)
        )

        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, joint_dim)
        )

    def forward(self, pose, z_latent):
        inp = torch.cat([pose, z_latent], dim=1)
        x = self.input_projection(inp)
        x = self.res_layers(x)
        return self.output_head(x)