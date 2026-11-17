import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://bc-api:8000").rstrip("/")
API_KEY = os.getenv("API_KEY", "")
TITLE = os.getenv("APP_TITLE", "Aide au diagnostic - Breast Cancer")
HEADERS = {"X-API-Key": API_KEY}

st.set_page_config(page_title=TITLE, page_icon="🩺", layout="wide")
st.title(f"🩺 {TITLE}")
st.caption("TP Kubernetes - démonstration pédagogique, pas un outil médical.")


@st.cache_data(ttl=30)
def get_info():
    r = requests.get(f"{API_URL}/api/info", timeout=3)
    r.raise_for_status()
    return r.json()


try:
    info = get_info()
except Exception as exc:  # noqa: BLE001
    st.error(f"API injoignable ou modèles pas encore prêts ({API_URL}) : {exc}")
    st.info("Vérifiez : kubectl get pods,svc ; le Job d'entraînement a-t-il terminé ?")
    st.stop()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Version API", info["app_version"])
c2.metric("Modèles", f'{info["model_version"]} ({info["algo"]})')
c3.metric("Accuracy classif.", info["classification"]["metrics"]["accuracy"])
c4.metric("R² régression", info["regression"]["metrics"]["r2"])

tab_clf, tab_reg, tab_stats = st.tabs(["Classification (malin / bénin)",
                                       "Régression (mean area)", "Statistiques"])


def feature_form(prefix: str, features: list, defaults: dict) -> dict:
    values = {}
    cols = st.columns(3)
    for i, f in enumerate(features):
        values[f] = cols[i % 3].number_input(f, value=float(defaults[f]), format="%.5f",
                                             key=f"{prefix}-{f}")
    return values


def post(path: str, features: dict):
    r = requests.post(f"{API_URL}{path}", json={"features": features}, headers=HEADERS, timeout=5)
    if r.status_code != 200:
        st.error(f"Erreur {r.status_code} : {r.text}")
        return None
    return r.json()


with tab_clf:
    ex = st.radio("Pré-remplir avec :", ["exemple malin", "exemple bénin", "moyennes"],
                  horizontal=True, key="clf-ex")
    defaults = {"exemple malin": info["examples"]["malin"], "exemple bénin": info["examples"]["benin"],
                "moyennes": info["feature_means"]}[ex]
    with st.expander("Les 30 mesures de la tumeur", expanded=False):
        vals = feature_form(f"clf-{ex}", info["classification"]["features"], defaults)
    if st.button("Prédire le diagnostic", type="primary"):
        res = post("/api/predict/classification", vals)
        if res:
            color = "🔴" if res["prediction"] == "malin" else "🟢"
            st.subheader(f'{color} Prédiction : {res["prediction"].upper()}')
            st.progress(min(max(res["proba_malin"], 0.0), 1.0),
                        text=f'Probabilité malin = {res["proba_malin"]:.2%} '
                             f'(seuil = {res["threshold_malin"]})')
            st.caption(f'Servi par le pod {res["pod"]} - API {res["app_version"]} - '
                       f'modèles {res["model_version"]} - cache : {res["cache"]}')

with tab_reg:
    st.write(f'Prédire **{info["regression"]["target"]}** à partir des 29 autres mesures.')
    feats = info["regression"]["features"]
    base = info["examples"]["benin"]
    with st.expander("Les 29 mesures", expanded=False):
        vals_r = feature_form("reg", feats, base)
    if st.button("Prédire mean area", type="primary"):
        res = post("/api/predict/regression", vals_r)
        if res:
            st.subheader(f'Surface moyenne prédite : {res["prediction"]}')
            st.caption(f'Valeur réelle de l\'exemple : {base["mean area"]} - pod {res["pod"]} - '
                       f'API {res["app_version"]} - cache : {res["cache"]}')

with tab_stats:
    if st.button("Rafraîchir"):
        st.cache_data.clear()
    try:
        s = requests.get(f"{API_URL}/api/stats", timeout=3).json()
        st.dataframe(pd.DataFrame([s]), hide_index=True)
    except Exception as exc:  # noqa: BLE001
        st.warning(f"Statistiques indisponibles : {exc}")
    st.json(info.get("model_config", {}))
