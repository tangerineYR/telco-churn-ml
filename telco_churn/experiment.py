"""All fitting and selection finish before the holdout is evaluated."""
import json
import platform
import random
import warnings
import hashlib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .data import load_data, split_data, distribution, digest
from .models import make_pipeline, fit_search
from .feature_selection import feature_record
from .evaluation import evaluate, score_model, plot_comparison

GROUPS = ["decision_tree", "linear_svm", "rbf_svm", "baselines"]

def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf8")

def run(data_path, output_dir, seed=42, groups=None, n_jobs=1, smoke=False):
    random.seed(seed); np.random.seed(seed)
    groups = GROUPS if groups is None else groups
    if not groups or set(groups) - set(GROUPS):
        raise ValueError("Unknown/empty model selection")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Use an empty output directory to avoid mixing experiment artifacts")
    X, y, data_meta = load_data(data_path)
    provenance_path = Path(data_path).with_suffix(".provenance.json")
    provenance = json.loads(provenance_path.read_text(encoding="utf8")) if provenance_path.exists() else {}
    if provenance.get("sha256") and provenance["sha256"] != data_meta["sha256"]:
        raise ValueError("Dataset provenance checksum mismatch")
    data_meta["provenance"] = provenance or {
    "source": "Local CSV",
    "source_verified": False
    }
    Xtr, Xte, ytr, yte = split_data(X, y, seed)
    config = {"experiment_type": "synthetic_smoke" if smoke else "holdout_evaluation",
              "seed": seed, "groups": groups, "test_fraction": 0.3,
              "split": {"stratified": True, "train_rows": len(ytr), "test_rows": len(yte),
                        "train_distribution": distribution(ytr), "test_distribution": distribution(yte),
                        "train_index_sha256": hashlib.sha256(np.asarray(Xtr.index, dtype="int64").tobytes()).hexdigest(),
                        "test_index_sha256": hashlib.sha256(np.asarray(Xte.index, dtype="int64").tobytes()).hexdigest()},
              "cv": {"type": "StratifiedKFold", "n_splits": 5, "shuffle": True, "random_state": seed},
              "selection": "maximum mean CV average precision; exact ties use first ParameterGrid index",
              "preprocessing": {"numeric": "StandardScaler", "categorical": "OneHotEncoder, dense, no drop, ignore unknown",
                                "SeniorCitizen": "categorical binary indicator", "customerID": "excluded"},
              "rfe": {"estimator": "LogisticRegression", "solver": "lbfgs", "max_iter": 5000,
                      "step": 1, "n_features_to_select": 20, "scope": "inside every CV training fold"},
              "baselines": "all transformed features; LogisticRegression class_weight=None",
              "linear_svc_solver": "dual=auto; C=[0.1,1,10]; max_iter=[1000,5000,10000]",
              "positive_class": "Yes=1", "metric_policy": "zero_division=0; AP is average precision, not trapezoidal PR area",
              "test_policy": "all candidates fixed before evaluation; default predict thresholds; no test-based selection"}
    write_json(out / "design.json", config)  # Written before model fitting/test metrics.
    warning_records = []
    original_showwarning = warnings.showwarning
    def record_warning(message, category, filename, lineno, file=None, line=None):
        warning_records.append({"phase": phase[0], "category": category.__name__, "message": str(message)})
        # Print without local machine paths; warnings are never silently suppressed.
        print(f"WARNING [{phase[0]}] {category.__name__}: {message}", flush=True)
    models, search_meta, cv_rows = {}, {}, []
    phase = ["training"]
    try:
        with warnings.catch_warnings(), threadpool_limits(limits=1):
            warnings.simplefilter("always")
            warnings.showwarning = record_warning
            for group in groups:
                kinds = ["dummy", "logistic"] if group == "baselines" else [group]
                for kind in kinds:
                    phase[0] = kind + "_fit"
                    print(f"Fitting {kind}", flush=True)
                    models[kind] = make_pipeline(kind, seed).fit(Xtr, ytr)
                    if kind in {"decision_tree", "linear_svm"}:
                        phase[0] = kind + "_search"
                        grid = None
                        if smoke:
                            grid = {"classifier__max_depth": [3]} if kind == "decision_tree" else {"classifier__C": [1]}
                        search = fit_search(Xtr, ytr, kind, seed, n_jobs, grid)
                        models[kind + "_tuned"] = search.best_estimator_
                        r = search.cv_results_
                        search_meta[kind + "_tuned"] = {
                            "best_parameters": search.best_params_, "best_cv_ap": float(r["mean_test_ap"][search.best_index_]),
                            "best_index": int(search.best_index_), "configuration_count": len(r["params"])}
                        for i, params in enumerate(r["params"]):
                            row = {"model": kind, "trial_index": i, "parameters": json.dumps(params, sort_keys=True),
                                   "mean_cv_score": float(r["mean_test_ap"][i]), "std_cv_score": float(r["std_test_ap"][i]),
                                   "rank": int(r["rank_test_ap"][i]), "status": "ok"}
                            for key in r:
                                if key.startswith(("mean_test_", "std_test_", "split")):
                                    row[key.replace("test_", "cv_")] = float(r[key][i])
                            cv_rows.append(row)
                        print(f"Completed {kind}: {len(r['params'])} configurations x 5 folds", flush=True)
            features = {name: feature_record(model) for name, model in models.items()}
            phase[0] = "final_holdout_evaluation"
            metrics, predictions, scores = {}, {}, {}
            for name, model in models.items():
                predictions[name] = model.predict(Xte)
                scores[name], score_type = score_model(model, Xte)
                metrics[name] = {"test": evaluate(yte, predictions[name], scores[name]),
                                 "score_type": score_type, "features": features[name], **search_meta.get(name, {})}
            plot_comparison(yte, predictions, scores, models, out / "figures")
    finally:
        warnings.showwarning = original_showwarning
        write_json(out / "warnings.json", warning_records)
    pd.DataFrame(cv_rows, columns=list(cv_rows[0]) if cv_rows else
                 ["model", "trial_index", "parameters", "mean_cv_score", "std_cv_score", "rank", "status"]).to_csv(out / "cv_search_results.csv", index=False)
    write_json(out / "metrics.json", {"dataset": data_meta, "design": config, "models": metrics})
    write_json(out / "selected_features.json", features)
    source_root = Path(__file__).resolve().parent.parent
    source_hashes = {f.relative_to(source_root).as_posix(): digest(f)
                     for folder in ["telco_churn", "scripts"] for f in (source_root / folder).glob("*.py")}
    manifest = {"completed_at_utc": datetime.now(timezone.utc).isoformat(), "status": "complete",
                "experiment_type": config["experiment_type"], "python": platform.python_version(),
                "platform": platform.system(), "packages": {p: version(p) for p in
                    ["numpy", "pandas", "scipy", "scikit-learn", "matplotlib", "joblib", "threadpoolctl"]},
                "seed": seed, "n_jobs": n_jobs, "blas_thread_limit": 1, "source_sha256": source_hashes,
                "warning_count_parent_process": len(warning_records),
                "warning_scope": "all warnings captured for n_jobs=1; parallel worker warnings also print to stderr",
                "full_primary": not smoke and set(groups) == set(GROUPS),
                "reproducibility": "Same-environment reruns intended; cross-version bitwise equality not guaranteed",
                "artifacts_sha256": {f.relative_to(out).as_posix(): digest(f) for f in out.rglob("*") if f.is_file()}}
    write_json(out / "experiment_manifest.json", manifest)
    return metrics
