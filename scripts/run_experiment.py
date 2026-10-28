import argparse
from telco_churn.experiment import run, GROUPS

def main():
    parser = argparse.ArgumentParser(
    description="Train and evaluate Telco customer churn classifiers"
    )   
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--models", nargs="+", choices=GROUPS, default=GROUPS)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()
    run(args.data_path, args.output_dir, args.seed, args.models, args.n_jobs)

if __name__ == "__main__":
    main()
