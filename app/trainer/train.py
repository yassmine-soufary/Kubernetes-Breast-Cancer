import io
import json
import os
import sys
import time
from datetime import datetime, timezone

import joblib
import numpy as np
from minio import Minio
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REG_TARGET = "mean area"


def log(msg: str) -> None:
    print(f"[trainer] {msg}", flush=True)


def build_models(algo: str):
    if algo == "forest":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        reg = RandomForestRegressor(n_estimators=200, random_state=42)
    elif algo == "linear":
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
        reg = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    else:
        raise ValueError(f"ALGO inconnu : {algo!r} (attendu : linear ou forest)")
    return clf, reg


def train(algo: str, version: str):
    data = load_breast_cancer(as_frame=True)
    X, y = data.data, data.target
    features = list(X.columns)

    clf, reg = build_models(algo)

    # ---------- 1. Classification ----------
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    clf.fit(Xtr, ytr)
    yp = clf.predict(Xte)
    clf_metrics = {
        "accuracy": round(float(accuracy_score(yte, yp)), 4),
        "f1": round(float(f1_score(yte, yp)), 4),
    }
    log(f"classifieur {algo} : {clf_metrics}")

    # ---------- 2. Régression ----------
    reg_features = [f for f in features if f != REG_TARGET]
    Xr, yr = X[reg_features], X[REG_TARGET]
    Xrtr, Xrte, yrtr, yrte = train_test_split(Xr, yr, test_size=0.2, random_state=42)
    reg.fit(Xrtr, yrtr)
    yrp = reg.predict(Xrte)
    reg_metrics = {
        "r2": round(float(r2_score(yrte, yrp)), 4),
        "mae": round(float(mean_absolute_error(yrte, yrp)), 2),
    }
    log(f"régresseur {algo} : {reg_metrics}")

    # Exemples prêts à l'emploi pour le frontend et les tests curl
    examples = {}
    for label, name in [(0, "malin"), (1, "benin")]:
        row = X[y == label].iloc[0]
        examples[name] = {k: round(float(v), 5) for k, v in row.items()}

    info = {
        "version": version,
        "algo": algo,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classification": {
            "features": features,
            "classes": {"0": "malin", "1": "benin"},
            "metrics": clf_metrics,
        },
        "regression": {
            "target": REG_TARGET,
            "features": reg_features,
            "metrics": reg_metrics,
        },
        "feature_means": {k: round(float(v), 5) for k, v in X.mean().items()},
        "examples": examples,
    }
    return clf, reg, info


def upload_bytes(client: Minio, bucket: str, key: str, payload: bytes, ctype: str) -> None:
    client.put_object(bucket, key, io.BytesIO(payload), length=len(payload), content_type=ctype)
    log(f"déposé s3://{bucket}/{key} ({len(payload)} octets)")


def dump_model(model) -> bytes:
    buf = io.BytesIO()
    joblib.dump(model, buf)
    return buf.getvalue()


def main() -> int:
    endpoint = os.getenv("MINIO_ENDPOINT", "minio:9000")
    bucket = os.getenv("MINIO_BUCKET", "models")
    version = os.getenv("MODEL_VERSION", "v1")
    algo = os.getenv("ALGO", "linear")
    access = os.getenv("MINIO_ACCESS_KEY")
    secret = os.getenv("MINIO_SECRET_KEY")
    secure = os.getenv("MINIO_SECURE", "false").lower() == "true"

    if not access or not secret:
        log("ERREUR : MINIO_ACCESS_KEY / MINIO_SECRET_KEY absents (Secret non injecté ?)")
        return 2

    log(f"version={version} algo={algo} endpoint={endpoint} bucket={bucket}")
    clf, reg, info = train(algo, version)

    client = Minio(endpoint, access_key=access, secret_key=secret, secure=secure)

    # MinIO peut démarrer après le Job : on réessaie quelques fois
    for attempt in range(1, 13):
        try:
            if not client.bucket_exists(bucket):
                client.make_bucket(bucket)
                log(f"bucket {bucket} créé")
            break
        except Exception as exc:  # noqa: BLE001
            log(f"MinIO injoignable (tentative {attempt}/12) : {exc}")
            time.sleep(5)
    else:
        log("ERREUR : MinIO toujours injoignable, abandon.")
        return 1  # le Job relancera le Pod (backoffLimit)

    upload_bytes(client, bucket, f"{version}/classifier.joblib", dump_model(clf), "application/octet-stream")
    upload_bytes(client, bucket, f"{version}/regressor.joblib", dump_model(reg), "application/octet-stream")
    upload_bytes(client, bucket, f"{version}/info.json",
                 json.dumps(info, ensure_ascii=False, indent=2).encode(), "application/json")
    log("entraînement terminé avec succès")
    return 0


if __name__ == "__main__":
    np.random.seed(42)
    sys.exit(main())
