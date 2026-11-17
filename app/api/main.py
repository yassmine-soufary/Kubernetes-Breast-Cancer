import hashlib
import io
import json
import logging
import os
import socket
import threading
import time
from typing import Dict, Optional

import joblib
import pandas as pd
import redis
import yaml
from fastapi import APIRouter, Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from minio import Minio
from pydantic import BaseModel

# --------------------------------------------------------------------------- config
APP_VERSION = os.getenv("APP_VERSION", "1.0")          # gravée dans l'image (Dockerfile)
MODEL_VERSION = os.getenv("MODEL_VERSION", "v1")        # ConfigMap
MINIO_ENDPOINT = os.getenv("MINIO_ENDPOINT", "minio:9000")
MINIO_BUCKET = os.getenv("MINIO_BUCKET", "models")
MINIO_ACCESS_KEY = os.getenv("MINIO_ACCESS_KEY", "")    # Secret
MINIO_SECRET_KEY = os.getenv("MINIO_SECRET_KEY", "")    # Secret
MINIO_SECURE = os.getenv("MINIO_SECURE", "false").lower() == "true"
REDIS_HOST = os.getenv("REDIS_HOST", "base")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "60"))
API_KEY = os.getenv("API_KEY", "")                      # Secret
MODEL_CONFIG_PATH = os.getenv("MODEL_CONFIG_PATH", "/etc/model-config/model-config.yaml")
POD_NAME = socket.gethostname()

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper(),
                    format="%(asctime)s [api] %(levelname)s %(message)s")
log = logging.getLogger("api")

# --------------------------------------------------------------------------- état
state = {"classifier": None, "regressor": None, "info": None, "error": "chargement en cours",
         "health_broken": False}
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, socket_timeout=1, socket_connect_timeout=1,
                  decode_responses=True)


def load_models_once() -> None:
    client = Minio(MINIO_ENDPOINT, access_key=MINIO_ACCESS_KEY, secret_key=MINIO_SECRET_KEY,
                   secure=MINIO_SECURE)

    def get(key: str) -> bytes:
        resp = client.get_object(MINIO_BUCKET, f"{MODEL_VERSION}/{key}")
        try:
            return resp.read()
        finally:
            resp.close()
            resp.release_conn()

    info = json.loads(get("info.json"))
    clf = joblib.load(io.BytesIO(get("classifier.joblib")))
    reg = joblib.load(io.BytesIO(get("regressor.joblib")))
    state.update(classifier=clf, regressor=reg, info=info, error=None)
    log.info("modèles %s chargés depuis s3://%s/%s/ (algo=%s)",
             MODEL_VERSION, MINIO_BUCKET, MODEL_VERSION, info.get("algo"))


def loader_loop() -> None:
    """Réessaie toutes les 5 s jusqu'à ce que les modèles soient disponibles dans MinIO."""
    while state["classifier"] is None:
        try:
            load_models_once()
        except Exception as exc:  # noqa: BLE001
            state["error"] = f"modèles {MODEL_VERSION} indisponibles : {type(exc).__name__}: {exc}"
            log.warning("%s - nouvel essai dans 5 s", state["error"])
            time.sleep(5)


_cfg_cache = {"mtime": None, "data": {}}


def model_config() -> dict:
    """Relit le fichier monté depuis le ConfigMap dès qu'il change (sans redémarrage)."""
    try:
        mtime = os.path.getmtime(MODEL_CONFIG_PATH)
        if mtime != _cfg_cache["mtime"]:
            with open(MODEL_CONFIG_PATH, encoding="utf-8") as fh:
                _cfg_cache["data"] = yaml.safe_load(fh) or {}
            _cfg_cache["mtime"] = mtime
            log.info("configuration modèle (re)chargée : %s", _cfg_cache["data"])
    except FileNotFoundError:
        _cfg_cache["data"] = {}
    return _cfg_cache["data"]


def redis_ok() -> bool:
    try:
        return bool(rds.ping())
    except redis.RedisError:
        return False


# --------------------------------------------------------------------------- app
app = FastAPI(title="Breast Cancer ML API", version=APP_VERSION)
api = APIRouter(prefix="/api")


@app.on_event("startup")
def _startup() -> None:
    log.info("démarrage API %s sur le pod %s (modèles attendus : %s)", APP_VERSION, POD_NAME, MODEL_VERSION)
    threading.Thread(target=loader_loop, daemon=True).start()


def require_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    if not API_KEY:
        raise HTTPException(500, "API_KEY non configurée (Secret absent ?)")
    if x_api_key != API_KEY:
        raise HTTPException(401, "clé d'API invalide ou absente (en-tête X-API-Key)")


def require_models() -> None:
    if state["classifier"] is None:
        raise HTTPException(503, state["error"] or "modèles non chargés")


@app.get("/health")
def health():
    if state["health_broken"]:
        return JSONResponse({"status": "KO", "reason": "panne simulée"}, status_code=500)
    return {"status": "ok", "pod": POD_NAME}


@app.get("/ready")
def ready():
    checks = {"models": state["classifier"] is not None, "redis": redis_ok()}
    body = {"ready": all(checks.values()), "checks": checks, "pod": POD_NAME}
    if not checks["models"]:
        body["detail"] = state["error"]
    return JSONResponse(body, status_code=200 if body["ready"] else 503)


class Features(BaseModel):
    features: Dict[str, float]


def build_frame(payload: Dict[str, float], expected: list) -> tuple:
    unknown = sorted(set(payload) - set(expected))
    if unknown:
        raise HTTPException(422, f"mesures inconnues : {unknown}")
    means = state["info"]["feature_means"]
    filled = [f for f in expected if f not in payload]
    row = {f: payload.get(f, means[f]) for f in expected}
    return pd.DataFrame([row], columns=expected), filled


def cached(kind: str, payload: dict, compute):
    key = "cache:" + hashlib.sha256(
        json.dumps([kind, MODEL_VERSION, payload, model_config()], sort_keys=True).encode()
    ).hexdigest()
    try:
        rds.incr(f"predictions:{kind}")
        if CACHE_TTL > 0:
            hit = rds.get(key)
            if hit:
                rds.incr("cache:hits")
                return {**json.loads(hit), "cache": True}
    except redis.RedisError as exc:
        log.warning("Redis indisponible : %s", exc)
    result = compute()
    try:
        if CACHE_TTL > 0:
            rds.setex(key, CACHE_TTL, json.dumps(result))
    except redis.RedisError:
        pass
    return {**result, "cache": False}


@api.get("/info")
def info():
    require_models()
    i = state["info"]
    return {
        "app_version": APP_VERSION,
        "model_version": i["version"],
        "algo": i["algo"],
        "trained_at": i["trained_at"],
        "pod": POD_NAME,
        "classification": i["classification"],
        "regression": i["regression"],
        "examples": i["examples"],
        "feature_means": i["feature_means"],
        "model_config": model_config(),
    }


@api.post("/predict/classification", dependencies=[Depends(require_key)])
def predict_classification(body: Features):
    require_models()
    feats = state["info"]["classification"]["features"]
    df, filled = build_frame(body.features, feats)

    def compute():
        threshold = float(model_config().get("threshold_malin", 0.5))
        proba = state["classifier"].predict_proba(df)[0]   # colonne 0 = malin, 1 = bénin
        p_malin = float(proba[0])
        label = "malin" if p_malin >= threshold else "benin"
        return {"prediction": label, "proba_malin": round(p_malin, 4), "threshold_malin": threshold,
                "filled_with_mean": filled}

    out = cached("classification", body.features, compute)
    return {**out, "model_version": MODEL_VERSION, "app_version": APP_VERSION, "pod": POD_NAME}


@api.post("/predict/regression", dependencies=[Depends(require_key)])
def predict_regression(body: Features):
    require_models()
    feats = state["info"]["regression"]["features"]
    df, filled = build_frame(body.features, feats)

    def compute():
        value = float(state["regressor"].predict(df)[0])
        return {"target": state["info"]["regression"]["target"], "prediction": round(value, 2),
                "filled_with_mean": filled}

    out = cached("regression", body.features, compute)
    return {**out, "model_version": MODEL_VERSION, "app_version": APP_VERSION, "pod": POD_NAME}


@api.get("/stats")
def stats():
    try:
        counts = {
            "classification": int(rds.get("predictions:classification") or 0),
            "regression": int(rds.get("predictions:regression") or 0),
            "cache_hits": int(rds.get("cache:hits") or 0),
        }
    except redis.RedisError as exc:
        raise HTTPException(503, f"Redis indisponible : {exc}")
    return {**counts, "served_by": POD_NAME, "app_version": APP_VERSION}


@api.post("/admin/break-health", dependencies=[Depends(require_key)])
def break_health():
    state["health_broken"] = True
    log.warning("panne simulée : /health renverra désormais 500")
    return {"health_broken": True, "pod": POD_NAME}


app.include_router(api)
