import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from ik_solve_nn.data import UR3IKDataset
from ik_solve_nn.train import (
    train_ur3_model,
    test_ur3_model,
    save_model_and_history,
    save_test_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate UR3 IK neural network models."
    )
    parser.add_argument("--train-csv", type=Path, default=Path("datasets/ur3_fk_train.csv"),
                        help="Path to the training CSV dataset.")
    parser.add_argument("--test-csv", type=Path, default=Path("datasets/ur3_fk_test.csv"),
                        help="Path to the test CSV dataset.")
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--num-classes", type=int, default=8,
                        help="Number of discrete IK solution modes (latent classes).")
    parser.add_argument("--mode-idx", type=int, default=2,
                        help="Latent mode index to use during testing.")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("model_checkpoints"),
                        help="Directory to save model weights and training history.")
    parser.add_argument("--results-dir", type=Path, default=Path("test_results"),
                        help="Directory to save test result arrays.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader worker processes.")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load datasets
    # ------------------------------------------------------------------
    print(f"Loading training dataset from: {args.train_csv}")
    train_dataset = UR3IKDataset(args.train_csv)

    print(f"Loading test dataset from:     {args.test_csv}")
    test_dataset = UR3IKDataset(args.test_csv)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train samples: {len(train_dataset):,}  |  Test samples: {len(test_dataset):,}")

    # ------------------------------------------------------------------
    # 2. Train
    # ------------------------------------------------------------------
    print(f"\nStarting training for {args.epochs} epoch(s)...")
    forward_net, inverse_net, history = train_ur3_model(
        train_loader,
        test_loader,
        num_classes=args.num_classes,
        epochs=args.epochs,
    )

    # ------------------------------------------------------------------
    # 3. Save model weights + history
    # ------------------------------------------------------------------
    save_model_and_history(forward_net, inverse_net, history, save_dir=str(args.checkpoint_dir))
    print(f"\nModel and history saved to: {args.checkpoint_dir}/")

    # ------------------------------------------------------------------
    # 4. Test
    # ------------------------------------------------------------------
    print(f"\nRunning test evaluation (mode_idx={args.mode_idx})...")
    pos_errors, ori_errors, joints_true, joints_pred, ee_true, ee_pred = test_ur3_model(
        inverse_net,
        test_loader,
        mode_idx=args.mode_idx,
        num_classes=args.num_classes,
    )

    mean_pos_mm = pos_errors.mean().item() * 1000.0
    median_pos_mm = pos_errors.median().item() * 1000.0
    print(f"Position error — mean: {mean_pos_mm:.3f} mm  |  median: {median_pos_mm:.3f} mm")

    # ------------------------------------------------------------------
    # 5. Save test results
    # ------------------------------------------------------------------
    save_test_results(pos_errors, ori_errors, joints_true, joints_pred, ee_true, ee_pred,
                      save_dir=str(args.results_dir))
    print(f"Test results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
