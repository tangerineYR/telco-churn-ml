from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

def make_rfe(seed=42):
    return RFE(LogisticRegression(max_iter=5000, solver="lbfgs", random_state=seed),
               n_features_to_select=20, step=1)

def feature_record(pipeline):
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    if "rfe" in pipeline.named_steps:
        selected = names[pipeline.named_steps["rfe"].support_]
        if len(selected) != 20 or pipeline.named_steps["classifier"].n_features_in_ != 20:
            raise ValueError("RFE/model feature dimension mismatch")
    else:
        selected = names
    return {"original_transformed_feature_count": len(names), "selected_feature_count": len(selected),
            "selected_feature_names": selected.tolist(),
            "feature_setting": "RFE20" if "rfe" in pipeline.named_steps else "all_transformed"}
