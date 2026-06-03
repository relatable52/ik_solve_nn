import argparse
from pathlib import Path
import sys

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
    load_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and/or evaluate UR3 IK neural network models."
    )
    parser.add_argument(
        "--run-mode",
        choices=["train", "test", "all"],
        default="all",
        help="Choose 'train' to only train, 'test' to only evaluate saved models, or 'all' to train then test.",
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
    parser.add_argument("--forward-model-path", type=Path, default=None,
                        help="Path to forward model weights for --run-mode test (defaults to <checkpoint-dir>/forward_net.pth).")
    parser.add_argument("--inverse-model-path", type=Path, default=None,
                        help="Path to inverse model weights for --run-mode test (defaults to <checkpoint-dir>/inverse_net.pth).")
    parser.add_argument("--results-dir", type=Path, default=Path("test_results"),
                        help="Directory to save test result arrays.")
    parser.add_argument("--num-workers", type=int, default=0,
                        help="Number of DataLoader worker processes.")
    return parser.parse_args()


def main():
    args = parse_args()
    inverse_net = None

    if args.run_mode in ("train", "all"):
        # ------------------------------------------------------------------
        # 1. Load datasets for training
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

        if args.run_mode == "train":
            print("Run mode is 'train': skipping evaluation.")
            return

    # ------------------------------------------------------------------
    # 4. Test
    # ------------------------------------------------------------------
    print(f"\nLoading test dataset from: {args.test_csv}")
    test_dataset = UR3IKDataset(args.test_csv)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    if args.run_mode == "test":
        forward_path = args.forward_model_path or (args.checkpoint_dir / "forward_net.pth")
        inverse_path = args.inverse_model_path or (args.checkpoint_dir / "inverse_net.pth")
        print(f"Loading checkpoints:\n- Forward: {forward_path}\n- Inverse: {inverse_path}")
        _, inverse_net = load_models(str(forward_path), str(inverse_path), num_classes=args.num_classes)

    print(f"Running test evaluation (mode_idx={args.mode_idx})...")
    pos_errors, ori_errors, joints_true, joints_pred, ee_true, ee_pred = test_ur3_model(
        inverse_net,
        test_loader,
        mode_idx=args.mode_idx,
        num_classes=args.num_classes,
    )

    pos_mm = pos_errors * 1000.0
    print(
        "Position error (mm) - "
        f"mean: {pos_mm.mean().item():.3f}  |  "
        f"median: {pos_mm.median().item():.3f}  |  "
        f"min: {pos_mm.min().item():.3f}  |  "
        f"max: {pos_mm.max().item():.3f}"
    )
    print(
        "Orientation error (RMSE) - "
        f"mean: {ori_errors.mean().item():.6f}  |  "
        f"median: {ori_errors.median().item():.6f}  |  "
        f"min: {ori_errors.min().item():.6f}  |  "
        f"max: {ori_errors.max().item():.6f}"
    )

    # ------------------------------------------------------------------
    # 5. Save test results
    # ------------------------------------------------------------------
    save_test_results(pos_errors, ori_errors, joints_true, joints_pred, ee_true, ee_pred,
                      save_dir=str(args.results_dir))
    print(f"Test results saved to: {args.results_dir}/")


if __name__ == "__main__":
    main()
