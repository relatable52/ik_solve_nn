import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm.auto import tqdm

from .models import UR3ForwardModel, UR3InverseModel
from .kinematics import DifferentiableUR3FK, rotmat_to_quat, quat_to_rot_matrix

def train_ur3_model(train_loader, test_loader, num_classes=8, epochs=15):
    """
    Trains the Forward and Inverse models using a fully Differentiable ETS
    Kinematics layer to enforce sub-millimeter task-space precision.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    forward_net = UR3ForwardModel(num_classes=num_classes).to(device)
    inverse_net = UR3InverseModel(num_classes=num_classes).to(device)
    dfk_layer = DifferentiableUR3FK().to(device)

    optimizer = optim.AdamW(
        list(forward_net.parameters()) + list(inverse_net.parameters()),
        lr=1e-3, weight_decay=1e-4
    )

    temperature = 1.0
    anneal_rate = 0.9

    # Added ori_rmse to history tracking
    history = {
        'train_total': [], 'train_fw': [], 'train_iv': [], 'train_kl': [], 'train_spatial_rmse': [], 'train_ori_rmse': [],
        'test_total': [],  'test_fw': [],  'test_iv': [],  'test_kl': [],  'test_spatial_rmse': [],  'test_ori_rmse': []
    }
    warmup_epochs = 25

    for epoch in (loop := tqdm(range(epochs))):
        # ---------------------------------------------------------
        # TRAINING PHASE
        # ---------------------------------------------------------
        forward_net.train()
        inverse_net.train()
        metrics_sums = {k: 0.0 for k in history.keys() if 'train' in k}

        for joints_true, poses_true in train_loader:
            joints_true = joints_true.to(device)
            poses_true = poses_true.to(device)
            optimizer.zero_grad()

            poses_pred, z_logits = forward_net(joints_true)
            z_soft = F.gumbel_softmax(z_logits, tau=temperature, hard=False)
            joints_pred = inverse_net(poses_true, z_soft)

            # --- Differentiable Physics Pass ---
            q_pred = torch.atan2(joints_pred[:, :6], joints_pred[:, 6:])

            # Now receiving BOTH position and rotation from DFK
            pos_pred_spatial, rot_pred_spatial = dfk_layer(q_pred)

            # Extract ground truth XYZ and Rotation Matrix
            pos_true_spatial = poses_true[:, :3]
            rot_true_spatial = quat_to_rot_matrix(poses_true[:, 3:])

            # --- Loss Calculations ---
            loss_fw = F.mse_loss(poses_pred, poses_true)
            loss_iv = F.mse_loss(joints_pred, joints_true)

            q_z = F.softmax(z_logits, dim=-1)
            log_q_z = F.log_softmax(z_logits, dim=-1)
            kl_div = (q_z * (log_q_z - np.log(1.0 / num_classes))).sum(dim=-1).mean()

            # Task-Space Spatial Position Loss
            loss_spatial_rmse = torch.sqrt(F.mse_loss(pos_pred_spatial, pos_true_spatial) + 1e-8)

            # Task-Space Orientation Loss
            loss_ori_rmse = torch.sqrt(F.mse_loss(rot_pred_spatial, rot_true_spatial) + 1e-8)

            # Curriculum Learning Weights
            if epoch < warmup_epochs:
                weight_pos = 0.0
                weight_ori = 0.0
            else:
                weight_pos = 5.0
                weight_ori = 0.05  # Gives the wrist joints a gentle but firm pull

            total_loss = (
                2.0 * loss_fw +
                10.0 * loss_iv +
                0.02 * kl_div +
                weight_pos * loss_spatial_rmse +
                weight_ori * loss_ori_rmse
            )

            total_loss.backward()
            optimizer.step()

            metrics_sums['train_total'] += total_loss.item()
            metrics_sums['train_fw'] += loss_fw.item()
            metrics_sums['train_iv'] += loss_iv.item()
            metrics_sums['train_kl'] += kl_div.item()
            metrics_sums['train_spatial_rmse'] += loss_spatial_rmse.item()
            metrics_sums['train_ori_rmse'] += loss_ori_rmse.item()

        # ---------------------------------------------------------
        # EVALUATION PHASE
        # ---------------------------------------------------------
        forward_net.eval()
        inverse_net.eval()
        test_sums = {k: 0.0 for k in history.keys() if 'test' in k}

        with torch.no_grad():
            for joints_true_test, poses_true_test in test_loader:
                joints_true_test = joints_true_test.to(device)
                poses_true_test = poses_true_test.to(device)

                poses_pred_test, z_logits_test = forward_net(joints_true_test)
                z_hard_test = F.gumbel_softmax(z_logits_test, tau=0.1, hard=True)
                joints_pred_test = inverse_net(poses_true_test, z_hard_test)

                # Physics Evaluation Pass
                q_pred_test = torch.atan2(joints_pred_test[:, :6], joints_pred_test[:, 6:])
                pos_pred_spatial_test, rot_pred_spatial_test = dfk_layer(q_pred_test)

                pos_true_spatial_test = poses_true_test[:, :3]
                rot_true_spatial_test = quat_to_rot_matrix(poses_true_test[:, 3:])

                loss_fw_test = F.mse_loss(poses_pred_test, poses_true_test)
                loss_iv_test = F.mse_loss(joints_pred_test, joints_true_test)

                loss_spatial_rmse_test = torch.sqrt(F.mse_loss(pos_pred_spatial_test, pos_true_spatial_test) + 1e-8)
                loss_ori_rmse_test = torch.sqrt(F.mse_loss(rot_pred_spatial_test, rot_true_spatial_test) + 1e-8)

                q_z_test = F.softmax(z_logits_test, dim=-1)
                log_q_z_test = F.log_softmax(z_logits_test, dim=-1)
                kl_div_test = (q_z_test * (log_q_z_test - np.log(1.0 / num_classes))).sum(dim=-1).mean()

                # Ensure test loss uses the same curriculum weights for accurate curves
                total_test_loss = (
                    3.0 * loss_fw_test +
                    10.0 * loss_iv_test +
                    0.02 * kl_div_test +
                    weight_pos * loss_spatial_rmse_test +
                    weight_ori * loss_ori_rmse_test
                )

                test_sums['test_total'] += total_test_loss.item()
                test_sums['test_fw'] += loss_fw_test.item()
                test_sums['test_iv'] += loss_iv_test.item()
                test_sums['test_kl'] += kl_div_test.item()
                test_sums['test_spatial_rmse'] += loss_spatial_rmse_test.item()
                test_sums['test_ori_rmse'] += loss_ori_rmse_test.item()

        # ---------------------------------------------------------
        # RECORDING METRICS
        # ---------------------------------------------------------
        n_train_batches = len(train_loader)
        n_test_batches = len(test_loader)

        for k in metrics_sums.keys():
            history[k].append(metrics_sums[k] / n_train_batches)
        for k in test_sums.keys():
            history[k].append(test_sums[k] / n_test_batches)

        temperature = max(0.05, temperature * anneal_rate)

        train_mm_error = history['train_spatial_rmse'][-1] * 1000.0
        test_mm_error = history['test_spatial_rmse'][-1] * 1000.0
        test_ori_error = history['test_ori_rmse'][-1]

        loop.set_description(
            f"Epoch {epoch+1:02d}/{epochs} | "
            f"Pos Err: {test_mm_error:.2f}mm | "
            f"Ori Err: {test_ori_error:.4f} | "
            f"Temp: {temperature:.2f}"
        )

    return forward_net, inverse_net, history

def test_ur3_model(inverse_net, test_loader, mode_idx=2, num_classes=8):
    device = next(inverse_net.parameters()).device
    inverse_net.eval()

    all_pos_errors = []
    all_ori_errors = []
    
    all_joints_true = []
    all_joints_pred = []
    all_ee_true = []
    all_ee_pred = []

    with torch.no_grad():
        for joints_true, poses_true in test_loader:
            joints_true = joints_true.to(device)
            poses_true = poses_true.to(device)

            z_onehot = torch.zeros(joints_true.size(0), num_classes, device=device)
            z_onehot[:, mode_idx] = 1.0

            joints_pred = inverse_net(poses_true, z_onehot)

            q_pred = torch.atan2(joints_pred[:, :6], joints_pred[:, 6:])
            dfk_layer = DifferentiableUR3FK().to(device)
            pos_pred, rot_pred = dfk_layer(q_pred)

            pos_true = poses_true[:, :3]
            rot_true = quat_to_rot_matrix(poses_true[:, 3:])

            pos_error = torch.norm(pos_pred - pos_true, dim=1)
            # Per-sample geodesic orientation error in degrees.
            rel_rot = torch.matmul(rot_pred, rot_true.transpose(1, 2))
            trace = rel_rot[:, 0, 0] + rel_rot[:, 1, 1] + rel_rot[:, 2, 2]
            cos_theta = torch.clamp((trace - 1.0) * 0.5, min=-1.0, max=1.0)
            ori_error = torch.acos(cos_theta) * (180.0 / np.pi)

            ee_true = np.concatenate([pos_true.cpu().numpy(), rot_true.cpu().numpy().reshape(-1, 9)], axis=1)
            ee_pred = np.concatenate([pos_pred.cpu().numpy(), rot_pred.cpu().numpy().reshape(-1, 9)], axis=1)

            all_pos_errors.append(pos_error.cpu())
            all_ori_errors.append(ori_error.cpu())

            all_joints_true.append(joints_true.cpu())
            all_joints_pred.append(joints_pred.cpu())
            all_ee_true.append(ee_true)
            all_ee_pred.append(ee_pred)

    all_pos_errors = torch.cat(all_pos_errors)
    all_ori_errors = torch.cat(all_ori_errors)
    all_joints_true = torch.cat(all_joints_true)
    all_joints_pred = torch.cat(all_joints_pred)
    all_ee_true = np.concatenate(all_ee_true, axis=0)
    all_ee_pred = np.concatenate(all_ee_pred, axis=0)
    return all_pos_errors, all_ori_errors, all_joints_true, all_joints_pred, all_ee_true, all_ee_pred

# Utility function to save models and training history
def save_model_and_history(forward_net, inverse_net, history, save_dir="model_checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(forward_net.state_dict(), os.path.join(save_dir, "forward_net.pth"))
    torch.save(inverse_net.state_dict(), os.path.join(save_dir, "inverse_net.pth"))

    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f)

# Utility function to save test results
def save_test_results(pos_errors, ori_errors, joints_true, joints_pred, ee_true, ee_pred, save_dir="test_results"):
    os.makedirs(save_dir, exist_ok=True)
    np.savez(os.path.join(save_dir, "test_results.npz"),
             pos_errors=pos_errors.numpy(),
             ori_errors=ori_errors.numpy(),
             joints_true=joints_true.numpy(),
             joints_pred=joints_pred.numpy(),
             ee_true=ee_true,
             ee_pred=ee_pred)

# Utility function to load models
def load_models(forward_path, inverse_path, num_classes=8):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    forward_net = UR3ForwardModel(num_classes=num_classes).to(device)
    inverse_net = UR3InverseModel(num_classes=num_classes).to(device)

    forward_net.load_state_dict(torch.load(forward_path, map_location=device))
    inverse_net.load_state_dict(torch.load(inverse_path, map_location=device))

    forward_net.eval()
    inverse_net.eval()
    return forward_net, inverse_net