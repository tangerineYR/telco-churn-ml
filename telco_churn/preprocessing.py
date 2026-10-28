from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .data import NUMERIC, CATEGORICAL

def make_preprocessor():
    # SeniorCitizen is a categorical binary indicator; no continuous scaling.
    return ColumnTransformer([
        ("numeric", StandardScaler(), NUMERIC),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=True)
