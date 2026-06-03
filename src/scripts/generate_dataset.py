import argparse
from pathlib import Path
import sys

PROJECT_SRC = Path(__file__).resolve().parents[1]
if str(PROJECT_SRC) not in sys.path:
	sys.path.insert(0, str(PROJECT_SRC))

from ik_solve_nn.data import generate_fk_dataset_csv


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Generate UR3 FK dataset using DifferentiableUR3FK (no robotics-toolbox)."
	)
	parser.add_argument("--save-dir", type=Path, default=Path("datasets"))
	parser.add_argument("--samples", type=int, default=2_000_000)
	parser.add_argument("--train-ratio", type=float, default=0.8, help="Proportion of samples to use for training set.")
	parser.add_argument("--chunk-size", type=int, default=100_000)
	parser.add_argument("--seed", type=int, default=42)
	return parser.parse_args()


def main():
    args = parse_args()

    train_samples = int(args.samples * args.train_ratio)
    test_samples = args.samples - train_samples

    print(f"Generating training dataset with {train_samples} samples...")
    train_path = args.save_dir / "ur3_fk_train.csv"
    generate_fk_dataset_csv(train_path, num_samples=train_samples, chunk_size=args.chunk_size, seed=args.seed)

    print(f"Generating test dataset with {test_samples} samples...")
    test_path = args.save_dir / "ur3_fk_test.csv"
    generate_fk_dataset_csv(test_path, num_samples=test_samples, chunk_size=args.chunk_size, seed=args.seed + 1)

    print(f"Datasets generated:\n- Train: {train_path}\n- Test: {test_path}")

if __name__ == "__main__":
	main()