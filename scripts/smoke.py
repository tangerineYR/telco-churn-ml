from pathlib import Path
from tempfile import TemporaryDirectory
from scripts.synthetic import make_frame
from scripts.validate_artifacts import validate
from telco_churn.experiment import run

def main():
    with TemporaryDirectory() as tmp:
        p = Path(tmp); make_frame().to_csv(p / "synthetic.csv", index=False)
        run(p / "synthetic.csv", p / "results", smoke=True)
        print("SMOKE PASS", validate(p / "results"))

if __name__ == "__main__":
    main()
