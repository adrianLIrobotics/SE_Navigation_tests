# app.py
# Dark, compact, interactive SE + Trust simulator with:
# - JSON policies editor
# - 2 plots per row
# - detailed traces per policy
# - Documentation page with robust math rendering (st.latex everywhere for equations)
# - time-varying observed meta-parameters via optional m_obs_schedule in JSON
# - Examples page with three sections
#
# UPDATE 1: added p0 (prior trust) as a sidebar slider and wired it into trust init.
# UPDATE 2: added "Expected vs Observed (SE & ExM)" plot (2-column layout) with y=0 decision boundary.
#           Expected values are computed PRE-evidence at each timestep:
#           - pT_global_pred uses Markov prediction only (before correction by match/mismatch evidence).
#           - ExM_hat uses m_pred with current weights w(t).
#           - SE_hat uses ExM_hat and pT_global_pred.

import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# -----------------------------
# Dark plot style (global)
# -----------------------------
plt.style.use("dark_background")
plt.rcParams.update({
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.facecolor": "black",
    "axes.facecolor": "black",
})


# -----------------------------
# Math utilities
# -----------------------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(v))
    if s < eps:
        return np.ones_like(v) / len(v)
    return v / s


def markov_predict(b: np.ndarray, T: np.ndarray) -> np.ndarray:
    # b^- = T^T b
    return T.T @ b


def hmm_correct(b_pred: np.ndarray, O: np.ndarray) -> np.ndarray:
    # b ∝ O ⊙ b_pred
    return normalize(O * b_pred)


def exm(m_vec: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, m_vec))


def se_posterior(exm_val: float, pT: float, gamma: float) -> float:
    # SE = ExM * pT + gamma * (1 - pT), gamma < 0
    return float(exm_val * pT + gamma * (1.0 - pT))


def abs_contrib(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    # c = w ⊙ |e|
    return w * np.abs(e)


def rel_attrib(c: np.ndarray) -> np.ndarray:
    # rho = c / sum(c)
    s = float(np.sum(c))
    if s < 1e-12:
        return np.zeros_like(c)
    return c / s


# -----------------------------
# Schedule utilities for time-varying observed meta-params
# -----------------------------
def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def sanitize_schedule(raw_schedule: Any, meta_names: List[str]) -> List[Dict[str, Any]]:
    if raw_schedule is None:
        return []
    if not isinstance(raw_schedule, list):
        raise ValueError("m_obs_schedule must be a list of objects like {'t': 30, 'Efficiency': 0.5}.")

    cleaned: List[Dict[str, Any]] = []
    for item in raw_schedule:
        if not isinstance(item, dict) or "t" not in item:
            raise ValueError("Each schedule entry must be an object with at least key 't'.")
        entry: Dict[str, Any] = {"t": int(item["t"])}
        for m in meta_names:
            if m in item:
                entry[m] = clamp01(float(item[m]))
        cleaned.append(entry)

    cleaned.sort(key=lambda d: d["t"])
    return cleaned


def get_m_obs_at_time(m_obs_base: np.ndarray, schedule: List[Dict[str, Any]], meta_names: List[str], t: int) -> np.ndarray:
    m_obs_t = m_obs_base.copy()
    for entry in schedule:
        if entry["t"] <= t:
            for k, m in enumerate(meta_names):
                if m in entry:
                    m_obs_t[k] = float(entry[m])
        else:
            break
    return m_obs_t


def get_change_points(schedule: List[Dict[str, Any]]) -> List[int]:
    return [int(e["t"]) for e in schedule] if schedule else []


def draw_change_lines(ax, change_points: List[int]):
    for cp in change_points:
        ax.axvline(cp, linestyle=":", linewidth=1, alpha=0.6)


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Policy:
    name: str
    m_pred: np.ndarray
    m_obs_base: np.ndarray
    m_obs_schedule: Optional[List[Dict[str, Any]]] = None


@dataclass
class Model:
    meta_names: List[str]
    w0: np.ndarray
    taus: np.ndarray
    T: np.ndarray
    p_match_T: float
    p_match_D: float
    gamma: float
    phi: float
    rigidity: float
    adapt: bool
    p0: float  # prior trust P(T) at t=0


# -----------------------------
# Simulation
# -----------------------------
def simulate(policies: List[Policy], model: Model, steps: int) -> Tuple[Dict[str, pd.DataFrame], Dict[str, List[int]]]:
    K = len(model.meta_names)
    results: Dict[str, pd.DataFrame] = {}
    change_points_by_policy: Dict[str, List[int]] = {}

    p0 = float(np.clip(model.p0, 0.0, 1.0))

    for pol in policies:
        # per-meta-parameter belief b_k = [p(T), p(D)]
        b = np.tile(np.array([p0, 1.0 - p0], dtype=float), (K, 1))
        w = model.w0.copy()

        schedule = sanitize_schedule(pol.m_obs_schedule, model.meta_names) if pol.m_obs_schedule else []
        change_points_by_policy[pol.name] = get_change_points(schedule)

        rows = []
        for t in range(steps):
            m_obs_t = get_m_obs_at_time(pol.m_obs_base, schedule, model.meta_names, t)

            # --- PRE-evidence trust prediction (expected / prior) ---
            pT_local_pred = np.zeros(K, dtype=float)
            b_pred_cache = [None] * K  # store b^- for reuse
            for k in range(K):
                b_pred_k = markov_predict(b[k], model.T)
                b_pred_cache[k] = b_pred_k
                pT_local_pred[k] = b_pred_k[0]

            pT_global_pred = float(np.dot(w, pT_local_pred))

            # Expected value under analogy (uses predicted meta-params m_pred)
            exm_hat = exm(pol.m_pred, w)
            se_hat = se_posterior(exm_hat, pT_global_pred, model.gamma)

            # --- Evidence/correction (observed / posterior) ---
            pT_local = np.zeros(K, dtype=float)
            match_flags = np.zeros(K, dtype=bool)

            for k in range(K):
                err = abs(float(m_obs_t[k] - pol.m_pred[k]))
                match = err <= float(model.taus[k])

                O = np.array([
                    model.p_match_T if match else 1.0 - model.p_match_T,
                    model.p_match_D if match else 1.0 - model.p_match_D
                ], dtype=float)

                b[k] = hmm_correct(b_pred_cache[k], O)
                pT_local[k] = b[k, 0]
                match_flags[k] = match

            pT_global = float(np.dot(w, pT_local))

            exm_val = exm(m_obs_t, w)
            se_val = se_posterior(exm_val, pT_global, model.gamma)

            e = m_obs_t - pol.m_pred
            c = abs_contrib(e, w)
            rho = rel_attrib(c)

            # Weight calibration (attention adaptation)
            if model.adapt:
                delta = se_val - exm_val
                w_candidate = w + model.phi * rho * delta * m_obs_t
                w_candidate = normalize(np.clip(w_candidate, 0.0, None))

                r = float(np.clip(model.rigidity, 0.0, 1.0))
                w = r * w + (1.0 - r) * w_candidate
                w = normalize(np.clip(w, 0.0, None))

            row = {
                "t": t,

                # expected (pre-evidence)
                "ExM_hat": exm_hat,
                "SE_hat": se_hat,
                "pT_global_pred": pT_global_pred,

                # observed (post-evidence)
                "ExM": exm_val,
                "SE": se_val,
                "SE_minus_ExM": se_val - exm_val,
                "pT_global": pT_global,
            }

            for k, name in enumerate(model.meta_names):
                row[f"m_pred_{name}"] = float(pol.m_pred[k])
                row[f"m_obs_{name}"] = float(m_obs_t[k])
                row[f"e_{name}"] = float(e[k])
                row[f"abs_e_{name}"] = float(abs(e[k]))
                row[f"match_{name}"] = bool(match_flags[k])

                row[f"pT_pred_{name}"] = float(pT_local_pred[k])
                row[f"pT_{name}"] = float(pT_local[k])

                row[f"c_{name}"] = float(c[k])
                row[f"rho_{name}"] = float(rho[k])
                row[f"w_{name}"] = float(w[k])

            rows.append(row)

        results[pol.name] = pd.DataFrame(rows)

    return results, change_points_by_policy


# -----------------------------
# Helpers for UI parsing
# -----------------------------
def parse_csv_floats(text: str, K: int, fallback: float) -> np.ndarray:
    parts = [p.strip() for p in text.split(",") if p.strip()]
    vals = []
    for p in parts:
        try:
            vals.append(float(p))
        except Exception:
            pass
    if len(vals) != K:
        vals = [fallback] * K
    return np.array(vals, dtype=float)


def default_policies_json(meta_names: List[str]) -> str:
    e, c, tc = meta_names[0], meta_names[1], meta_names[2] if len(meta_names) >= 3 else "TaskCompletion"
    example = [
        {"name": "Fork",
         "m_pred": {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs":  {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs_schedule": [
             {"t": 0, e: 0.6, c: 0.8, tc: 1.0},
             {"t": 30, e: 0.3}
         ]},
        {"name": "Sticks",
         "m_pred": {e: 0.6, c: 0.8, tc: 1.0},
         "m_obs":  {e: 0.5, c: 0.2, tc: 0.7},
         "m_obs_schedule": [
             {"t": 0, e: 0.5, c: 0.2, tc: 0.7},
             {"t": 20, tc: 0.4}
         ]},
    ]
    return json.dumps(example, indent=2)


def parse_policies_from_json(raw: str, meta_names: List[str]) -> List[Policy]:
    data = json.loads(raw)
    if not isinstance(data, list) or len(data) == 0:
        raise ValueError("Policies JSON must be a non-empty list.")
    policies: List[Policy] = []
    for item in data:
        name = str(item.get("name", "Unnamed"))
        mp = item.get("m_pred", {})
        mo = item.get("m_obs", {})
        sched = item.get("m_obs_schedule", None)

        m_pred = np.array([clamp01(float(mp.get(m, 0.0))) for m in meta_names], dtype=float)
        m_obs_base = np.array([clamp01(float(mo.get(m, 0.0))) for m in meta_names], dtype=float)
        policies.append(Policy(name=name, m_pred=m_pred, m_obs_base=m_obs_base, m_obs_schedule=sched))
    return policies


def make_policy_colors(policy_names: List[str]) -> Dict[str, Any]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", ["#00ffcc", "#ff5555", "#aaaaaa"])
    return {n: cycle[i % len(cycle)] for i, n in enumerate(policy_names)}


# -----------------------------
# Pages
# -----------------------------
def render_docs():
    st.title("Documentation")
    st.header("Prior trust $p_0$")
    st.latex(r"b_0^k=\begin{bmatrix}p_0\\1-p_0\end{bmatrix}")
    st.markdown("This is now configurable in the Simulator sidebar under **Trust model**.")


def render_examples():
    st.title("Examples")
    st.caption("Load a JSON template into the Simulator editor.")
    if "policies_json" not in st.session_state:
        st.session_state.policies_json = None
    example = [
        {
            "name": "EScooter",
            "m_pred": {"Efficiency": 0.8, "Comfort": 0.75, "TaskCompletion": 1.0},
            "m_obs": {"Efficiency": 0.8, "Comfort": 0.75, "TaskCompletion": 1.0},
            "m_obs_schedule": [{"t": 0, "Efficiency": 0.8, "Comfort": 0.75, "TaskCompletion": 1.0},
                               {"t": 12, "Comfort": 0.35}],
        },
        {
            "name": "Walking",
            "m_pred": {"Efficiency": 0.55, "Comfort": 0.7, "TaskCompletion": 1.0},
            "m_obs": {"Efficiency": 0.55, "Comfort": 0.7, "TaskCompletion": 1.0},
        },
    ]
    if st.button("Load E-scooter vs Walking ✅"):
        st.session_state.policies_json = json.dumps(example, indent=2)
        st.success("Loaded! Go to Simulator and run.")

    st.code(json.dumps(example, indent=2), language="json")


def render_simulator():
    st.title("SE & Trust Simulator")

    if "run_clicked" not in st.session_state:
        st.session_state.run_clicked = True
    if "policies_json" not in st.session_state:
        st.session_state.policies_json = None

    with st.sidebar:
        st.header("Controls")

        with st.expander("Simulation", expanded=True):
            steps = st.number_input("Iterations", 1, 500, 30, 1)

        with st.expander("Meta-parameters", expanded=True):
            meta_names_str = st.text_input("Meta-parameter names (comma-separated)", value="Efficiency, Comfort, TaskCompletion")
            meta_names = [x.strip() for x in meta_names_str.split(",") if x.strip()]
            K = len(meta_names)

            w0_str = st.text_input("Initial weights (comma-separated)", value="0.4, 0.4, 0.2")
            w0 = normalize(np.clip(parse_csv_floats(w0_str, K, 1.0 / max(K, 1)), 0.0, None))

            taus_str = st.text_input("Match thresholds τ (comma-separated)", value="0.10, 0.10, 0.05")
            taus = np.clip(parse_csv_floats(taus_str, K, 0.10), 0.0, 1.0)

        with st.expander("Trust model (Markov + observations)", expanded=False):
            p0 = st.slider("Prior trust p0 = P(T) at t=0", 0.0, 1.0, 0.50, 0.01)

            c1, c2 = st.columns(2)
            with c1:
                T00 = st.slider("P(T→T)", 0.0, 1.0, 0.85, 0.01)
                T10 = st.slider("P(D→T)", 0.0, 1.0, 0.20, 0.01)
            with c2:
                T01 = st.slider("P(T→D)", 0.0, 1.0, 0.15, 0.01)
                T11 = st.slider("P(D→D)", 0.0, 1.0, 0.80, 0.01)

            row0 = normalize(np.array([T00, T01], dtype=float))
            row1 = normalize(np.array([T10, T11], dtype=float))
            T = np.vstack([row0, row1])

            p_match_T = st.slider("P(match | Trust)", 0.0, 1.0, 0.80, 0.01)
            p_match_D = st.slider("P(match | Distrust)", 0.0, 1.0, 0.30, 0.01)

        with st.expander("SE + weight adaptation", expanded=False):
            gamma = st.slider("γ (distrust penalty, <0)", -2.0, -0.01, -0.40, 0.01)
            adapt = st.checkbox("Enable weight calibration", value=True)
            phi = st.slider("φ (learning rate)", 0.0, 1.0, 0.10, 0.01)
            rigidity = st.slider("Rigidity r (1=stubborn, 0=flexible)", 0.0, 1.0, 0.50, 0.01)

        with st.expander("Policies (JSON editor)", expanded=True):
            if st.session_state.policies_json is None:
                st.session_state.policies_json = default_policies_json(meta_names)
            st.session_state.policies_json = st.text_area("Policies JSON", value=st.session_state.policies_json, height=280)

        if st.button("Run simulation ✅", type="primary"):
            st.session_state.run_clicked = True

    if not st.session_state.run_clicked:
        st.stop()

    model = Model(
        meta_names=meta_names, w0=w0, taus=taus, T=T,
        p_match_T=float(p_match_T), p_match_D=float(p_match_D),
        gamma=float(gamma), phi=float(phi), rigidity=float(rigidity), adapt=bool(adapt),
        p0=float(p0)
    )

    try:
        policies = parse_policies_from_json(st.session_state.policies_json, meta_names)
    except Exception as e:
        st.error(f"Policies JSON error: {e}")
        st.stop()

    results, change_points_by_policy = simulate(policies, model, int(steps))
    policy_names = list(results.keys())
    colors = make_policy_colors(policy_names)

    def fig_small():
        return plt.subplots(figsize=(4.8, 2.2))

    def fig_small_tall():
        return plt.subplots(figsize=(4.8, 2.6))

    # Row 1: expected vs observed and trust predicted vs posterior
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = fig_small_tall()
        for name, df in results.items():
            ax.plot(df["t"], df["SE"], label=f"{name} SE (obs)", color=colors[name], linewidth=2)
            ax.plot(df["t"], df["ExM"], linestyle="--", alpha=0.7, color=colors[name], label=f"{name} ExM (obs)")
            ax.plot(df["t"], df["SE_hat"], linestyle=":", alpha=0.95, color=colors[name], label=f"{name} SE (exp)")
            ax.plot(df["t"], df["ExM_hat"], linestyle="dashdot", alpha=0.8, color=colors[name], label=f"{name} ExM (exp)")
            draw_change_lines(ax, change_points_by_policy.get(name, []))
        ax.axhline(0.0, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Expected vs Observed (SE & ExM)  |  y=0 decision boundary")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    with col2:
        fig, ax = fig_small_tall()
        for name, df in results.items():
            ax.plot(df["t"], df["pT_global"], label=f"{name} pT (post)", color=colors[name])
            ax.plot(df["t"], df["pT_global_pred"], linestyle=":", alpha=0.8, label=f"{name} pT (pre)", color=colors[name])
            draw_change_lines(ax, change_points_by_policy.get(name, []))
        ax.set_ylim(0, 1)
        ax.set_title("Global trust: predicted vs posterior")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    # Row 2: distrust penalty and weight evolution
    col3, col4 = st.columns(2)
    with col3:
        fig, ax = fig_small()
        for name, df in results.items():
            ax.plot(df["t"], df["SE_minus_ExM"], label=name, color=colors[name])
            draw_change_lines(ax, change_points_by_policy.get(name, []))
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title("Distrust penalty: SE − ExM")
        ax.set_xlabel("t")
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    with col4:
        fig, ax = fig_small()
        pick = policy_names[0]
        dfp = results[pick]
        for m in meta_names:
            ax.plot(dfp["t"], dfp[f"w_{m}"], label=f"w_{m}")
        draw_change_lines(ax, change_points_by_policy.get(pick, []))
        ax.set_title(f"Weight evolution (policy: {pick})")
        ax.set_xlabel("t")
        ax.set_ylim(0, 1)
        ax.legend()
        st.pyplot(fig, use_container_width=True)

    st.subheader("Detailed traces")
    for name, df in results.items():
        with st.expander(f"{name}", expanded=False):
            st.dataframe(df, use_container_width=True)


# -----------------------------
# App entry
# -----------------------------
st.set_page_config(page_title="SE & Trust Simulator", layout="wide")

with st.sidebar:
    st.markdown("## Navigation")
    page = st.radio("Go to", ["Simulator", "Documentation", "Examples"], index=0)

if page == "Documentation":
    render_docs()
elif page == "Examples":
    render_examples()
else:
    render_simulator()
