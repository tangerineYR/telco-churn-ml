"""Download and validate the IBM Telco Customer Churn dataset."""
import argparse
from pathlib import Path
from urllib.request import urlopen
from datetime import datetime, timezone
from telco_churn.data import SOURCE_URL, load_data
from telco_churn.experiment import write_json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/Telco-Customer-Churn.csv")
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError("Refusing to overwrite an existing dataset")
    path.parent.mkdir(parents=True, exist_ok=True)
    with urlopen(SOURCE_URL, timeout=60) as response:
        payload = response.read()
    path.write_bytes(payload)
    _, _, metadata = load_data(path)
    write_json(path.with_suffix(".provenance.json"), {
        "source_url": SOURCE_URL,
        "accessed_at_utc": datetime.now(timezone.utc).isoformat(),
        "sha256": metadata["sha256"],
        "file_size_bytes": len(payload),
        "original_rows": metadata["raw_rows"],
        "columns": metadata["columns"]
    })
    print("Dataset downloaded and schema validated", metadata["sha256"])

if __name__ == "__main__":
    main()
