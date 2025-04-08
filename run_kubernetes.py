import os
import json
import argparse
import numpy as np
from phyto_nas_tsc import fit
from phyto_nas_tsc._data_handler import DataHandler

def parse_args():
    parser = argparse.ArgumentParser(description="Phyto-NAS-T Kubernetes runner")

    parser.add_argument("--population_size", type=int, default=20, help="Size of the model population")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--timeout", type=int, default=86400, help="Max optimization time in seconds")
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping")

    parser.add_argument("--data_dir", type=str, default="/app/phyto_nas_tsc/data", help="Path to data directory")
    parser.add_argument("--results_dir", type=str, default="/abyss/home/results", help="Where to save result files")
    parser.add_argument("--run_id", type=str, default="manual-run", help="Run identifier for naming output")

    return parser.parse_args()

def main():
    print("\n=== Phyto-NAS-T Kubernetes Runner ===")
    args = parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    print(f"Data directory: {args.data_dir}")
    print(f"Files in data directory: {os.listdir(args.data_dir)}")

    try:
        handler = DataHandler(args.data_dir)
        handler.load_and_preprocess()

        X = handler.X_analysis.reshape(handler.X_analysis.shape[0], 1, -1)
        y = handler.y_analysis

        print(f"\nData shape - X: {X.shape}, y: {y.shape}")

        result = fit(
            X=X,
            y=y,
            scoring='accuracy',
            others={
                'population_size': args.population_size,
                'generations': args.generations,
                'timeout': args.timeout,
                'early_stopping': args.early_stopping
            }
        )

        result_file = os.path.join(args.results_dir, f"results_{args.run_id}.json")
        with open(result_file, 'w') as f:
            json.dump({
                'accuracy': float(result['accuracy']),
                'architecture': str(result['architecture']),
                'parameters': result['parameters']
            }, f, indent=2)

        print(f"\nResults saved to: {result_file}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        raise

if __name__ == "__main__":
    main()
