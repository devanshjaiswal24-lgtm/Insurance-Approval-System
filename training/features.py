"""Leakage-safe feature construction shared by training and explainability."""
import pandas as pd


ENGINEERED_NUMERIC = [
    "young_driver",
    "senior_driver",
    "claims_accidents_total",
    "experience_gap",
]

ENGINEERED_OWNERS = {
    "young_driver": "customer_age",
    "senior_driver": "customer_age",
    "claims_accidents_total": "previous_claims",
    "experience_gap": "driving_experience",
}


class FeatureBuilder:
    """Add deterministic features without reading the target or audit columns."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        out = X.copy()
        out["young_driver"] = (out["customer_age"] < 25).astype(float)
        out["senior_driver"] = (out["customer_age"] > 70).astype(float)
        out["claims_accidents_total"] = (
            out["previous_claims"] + out["previous_accidents"]
        ).astype(float)
        out["experience_gap"] = (
            out["customer_age"] - out["driving_experience"]
        ).astype(float)
        return out

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return list(ENGINEERED_NUMERIC)
        return list(input_features) + ENGINEERED_NUMERIC
