import joblib
from xgboost import XGBClassifier


def build_model() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
    )


def save_model(model, path) -> None:
    joblib.dump(model, path)


def load_model(path):
    return joblib.load(path)
