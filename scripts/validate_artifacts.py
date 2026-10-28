import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from telco_churn.data import digest

def validate(directory, full=False):
    p = Path(directory)
    metrics = json.loads((p / "metrics.json").read_text(encoding="utf8"))
    features = json.loads((p / "selected_features.json").read_text(encoding="utf8"))
    manifest = json.loads((p / "experiment_manifest.json").read_text(encoding="utf8"))
    cv = pd.read_csv(p / "cv_search_results.csv")
    
    assert manifest["status"] == "complete"
    
    for file, sha in manifest["artifacts_sha256"].items():
        assert digest(p / file) == sha, file
    
    # Verify that the current source matches the recorded experiment.
    project_root = Path(__file__).resolve().parents[1]
    recorded_sources = manifest.get("source_sha256", {})

    current_source_paths = {
        source.relative_to(project_root).as_posix()
        for folder in ("telco_churn", "scripts")
        for source in (project_root / folder).glob("*.py")
    }

    if not recorded_sources:
        raise ValueError("Manifest does not contain source checksums")

    if set(recorded_sources) != current_source_paths:
        raise ValueError(
            "Source file list does not match the experiment manifest"
        )

    for relative_path, expected_sha in recorded_sources.items():
        source_path = project_root / relative_path

        if not source_path.is_file():
            raise ValueError(f"Source file missing: {relative_path}")

        if digest(source_path) != expected_sha:
            raise ValueError(
                f"Source checksum mismatch: {relative_path}"
            )
    
    if full:
        assert manifest["full_primary"]
        assert cv.groupby("model").size().to_dict() == {"decision_tree": 72, "linear_svm": 9}
        assert len(metrics["models"]) == 7
    
    for name, entry in metrics["models"].items():
        record = features[name]
        assert len(record["selected_feature_names"]) == record["selected_feature_count"]
        if record["feature_setting"] == "RFE20": assert record["selected_feature_count"] == 20
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "average_precision"]:
            assert np.isfinite(entry["test"][key]) and 0 <= entry["test"][key] <= 1
        cm = np.asarray(entry["test"]["confusion_matrix"])
        assert cm.shape == (2, 2) and cm.sum() == metrics["design"]["split"]["test_rows"]
    
    if len(cv):
        assert cv.status.eq("ok").all() and np.isfinite(cv.mean_cv_score).all()
        assert not any("holdout" in c or c.startswith("test_") for c in cv.columns)
    
    for filename in ["decision_tree.png", "confusion_matrix_dt.png", "confusion_matrix_svm.png", "roc_comparison.png", "pr_comparison.png"]:
        if full or len(metrics["models"]) == 7:
            assert (p / "figures" / filename).stat().st_size > 1000
    
    return {"valid": True, "models": len(metrics["models"]), "cv_rows": len(cv)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("directory"); parser.add_argument("--full", action="store_true")
    args = parser.parse_args(); print(validate(args.directory, args.full))
