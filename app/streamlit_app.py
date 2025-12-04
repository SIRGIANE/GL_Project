"""
Streamlit UI pour Bug Predictor AI
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import sys
from pathlib import Path
from radon.raw import analyze
from radon.metrics import h_visit
from radon.complexity import cc_visit

API_URL = "http://localhost:5000/api"

st.set_page_config(
    page_title="🐛 Bug Predictor AI",
    page_icon="🐛",
    layout="wide",
    initial_sidebar_state="expanded"
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
try:
    from src.utils.preprocessing import helper_for_complexity_evaluation
    HAS_HELPER = True
except Exception:
    HAS_HELPER = False


class APIClient:
    def __init__(self, base_url=API_URL):
        self.base = base_url

    def health(self):
        try:
            r = requests.get(f"{self.base}/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False

    def status(self):
        try:
            r = requests.get(f"{self.base}/status", timeout=5)
            return r.json() if r.ok else {}
        except Exception:
            return {}

    def analyze_code(self, code):
        try:
            r = requests.post(
                f"{self.base}/analyze/code",
                json={"code": code},
                timeout=30
            )
            return r.json() if r.ok else {"error": r.text}
        except Exception as e:
            return {"error": str(e)}


def extract_local(code_str):
    """Extraction locale pour debug"""
    try:
        raw = analyze(code_str)
        f1 = raw.loc
        f13 = raw.sloc
        f14 = raw.comments
        f15 = raw.blank
        f16 = raw.lloc

        ast = h_visit(code_str)
        hal = ast.total
        f17 = hal.h1
        f18 = hal.h2
        f19 = hal.N1
        f20 = hal.N2
        f5 = hal.length
        f6 = hal.volume
        f7 = (f6 / f5) if f5 > 0 else 0
        f8 = hal.difficulty
        f9 = f7 * f6
        f10 = hal.effort
        f12 = hal.time

        cc = cc_visit(code_str)
        try:
            f2 = cc[-1].complexity
        except Exception:
            f2 = 0

        if HAS_HELPER:
            try:
                f21 = helper_for_complexity_evaluation(f5, f6, f8, f10, f12)
            except Exception:
                f21 = int(((f5 >= 300) or (f6 >= 1000) or (f8 >= 50) or (f10 >= 500000) or (f12 >= 5000)))
        else:
            f21 = int(((f5 >= 300) or (f6 >= 1000) or (f8 >= 50) or (f10 >= 500000) or (f12 >= 5000)))

        features = [f1, f2, f5, f6, f7, f8, f9, f10, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21]
        return features, f21
    except Exception as e:
        st.error(f"❌ Erreur extraction: {e}")
        return None, None


# ===== UI =====
client = APIClient()

st.title("🐛 Bug Predictor AI")
st.markdown("Détecte les bugs potentiels via analyse statique")

# Sidebar
with st.sidebar:
    st.markdown("## 🔌 État API")
    if client.health():
        st.success("✅ API en ligne")
        status = client.status()
        if status:
            st.write(f"**Modèle:** {status.get('model_type', 'N/A')}")
            st.write(f"**Scaler:** {'✓' if status.get('scaler_loaded') else '✗'}")
            st.write(f"**PCA:** {'✓' if status.get('pca_loaded') else '✗'}")
    else:
        st.error("❌ API déconnectée")
        st.info("Lancez: `python app/api.py`")
    
    st.divider()
    page = st.radio("Page", ["🎯 Prédiction", "🔬 Debug", "ℹ️ À propos"])


# ===== PAGE PRÉDICTION =====
if page == "🎯 Prédiction":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        code = st.text_area("Collez le code Python", height=300, placeholder="def hello():\n    pass")
    
    with col2:
        st.markdown("### Exemples")
        if st.button("✅ Code Propre"):
            st.session_state.code = "def add(a, b):\n    return a + b"
            st.rerun()
        if st.button("🐛 Code Complexe"):
            st.session_state.code = (
                "a0 = a1 = a2 = a3 = a4 = a5 = a6 = a7 = a8 = a9 = 0\n"
                "a10 = a11 = a12 = a13 = a14 = a15 = a16 = a17 = a18 = a19 = 0\n"
                "result = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 + a8 + a9\n"
                "for i in range(200):\n"
                "    for j in range(100):\n"
                "        result *= (i+1)*(j+1)\n"
            )
            st.rerun()
    
    if "code" in st.session_state and not code:
        code = st.session_state.code
    
    if st.button("🚀 Analyser", key="btn_analyze", use_container_width=True, type="primary"):
        if not code.strip():
            st.warning("⚠️ Fournir du code")
        else:
            with st.spinner("Analyse..."):
                resp = client.analyze_code(code)
            
            # Debug JSON
            with st.expander("📋 Réponse brute API"):
                st.json(resp)
            
            st.divider()
            
            if "error" in resp:
                st.error(f"❌ {resp['error']}")
            else:
                # Résultats
                is_bug = resp.get("is_bug", False)
                prob = resp.get("probability", 0.0)
                risk = resp.get("risk_level", "N/A")
                metrics = resp.get("metrics", {})
                evaluation = metrics.get("evaluation", 0)
                
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Prédiction",
                        "🐛 BUGGY" if is_bug else "✅ PROPRE",
                        delta=f"{prob:.0%}"
                    )
                with c2:
                    st.metric("Probabilité", f"{prob:.1%}")
                with c3:
                    st.metric("Risque", risk)
                with c4:
                    st.metric(
                        "Évaluation",
                        "🚨 Critique" if evaluation == 1 else "✅ Normal"
                    )
                
                # Métriques détaillées
                st.markdown("### 📊 Métriques Radon")
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("LOC", f"{metrics.get('LOC', 0):.0f}")
                    st.metric("Complexité", f"{metrics.get('cyclomatic', 0):.0f}")
                with c2:
                    st.metric("Volume (v)", f"{metrics.get('v', 0):.0f}")
                    st.metric("Difficulté (d)", f"{metrics.get('d', 0):.2f}")
                with c3:
                    st.metric("Effort", f"{metrics.get('e', 0):.0f}")
                    st.metric("Temps (t)", f"{metrics.get('t', 0):.0f}s")
                
                # Features tableau
                st.markdown("### 🔬 18 Features")
                features = resp.get("features", [])
                if features:
                    names = [
                        'LOC', 'cyclomatic', 'n', 'v', 'l', 'd', 'i', 'e', 't',
                        'sloc', 'comments', 'blank', 'lloc',
                        'uniq_op', 'uniq_opnd', 'total_op', 'total_opnd', 'evaluation'
                    ]
                    df = pd.DataFrame({
                        'Feature': names[:len(features)],
                        'Valeur': features
                    })
                    st.dataframe(df, width='stretch')
                    
                    # Chart
                    fig = px.bar(
                        df,
                        x='Feature',
                        y='Valeur',
                        color='Valeur',
                        color_continuous_scale='Viridis',
                        title="Distribution des 18 Features"
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=400, showlegend=False)
                    st.plotly_chart(fig, width='stretch', config={"responsive": True})


# ===== PAGE DEBUG =====
elif page == "🔬 Debug":
    code = st.text_area("Code pour debug", height=300)
    
    if code.strip():
        features, evaluation = extract_local(code)
        if features:
            names = [
                'LOC', 'cyclomatic', 'n', 'v', 'l', 'd', 'i', 'e', 't',
                'sloc', 'comments', 'blank', 'lloc',
                'uniq_op', 'uniq_opnd', 'total_op', 'total_opnd', 'evaluation'
            ]
            df = pd.DataFrame({
                'Index': range(len(names)),
                'Feature': names,
                'Valeur': features
            })
            st.dataframe(df, width='stretch')
            
            st.markdown("### ⚡ Seuils")
            c1, c2 = st.columns(2)
            with c1:
                st.write(f"**n >= 300** ? {features[2]:.0f} → {features[2] >= 300}")
                st.write(f"**v >= 1000** ? {features[3]:.0f} → {features[3] >= 1000}")
                st.write(f"**d >= 50** ? {features[5]:.2f} → {features[5] >= 50}")
            with c2:
                st.write(f"**effort >= 500000** ? {features[7]:.0f} → {features[7] >= 500000}")
                st.write(f"**time >= 5000** ? {features[8]:.0f} → {features[8] >= 5000}")
                st.write(f"**Eval finale:** {int(evaluation)} → {'🐛 BUGGY' if evaluation == 1 else '✅ PROPRE'}")
    else:
        st.info("Entrez du code")


# ===== PAGE À PROPOS =====
elif page == "ℹ️ À propos":
    st.markdown("""
    ### 🐛 Bug Predictor AI
    
    Analyse statique de code Python pour détecter les bugs potentiels.
    
    **Méthode:**
    - Extraction 18 features via Radon (complexité, volume, effort)
    - StandardScaler + PCA normalisent
    - KNeighborsClassifier prédit BUGGY/PROPRE
    
    **Seuils d'évaluation** (au moins 1 = BUGGY):
    - n (halstead length) ≥ 300
    - v (volume) ≥ 1000
    - d (difficulty) ≥ 50
    - effort ≥ 500000
    - time ≥ 5000
    """)