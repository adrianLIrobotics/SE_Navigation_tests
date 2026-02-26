#!/usr/bin/env python3
"""se_prepost_eval_optionB.py

Option B: threshold-based trust update (+ optional post-execution weight calibration).

Input: a JSON file describing
- meta_names
- target: w_T, c_T
- a list of policies with: name, c_B, m_hat, kappa, (optional) m_obs
- trust_model: taus, T, p_match_T, p_match_D
- gamma (distrust penalty)
- p0 (initial trust prior)
- optional weight_update: enable, phi, rigidity

Outputs:
- a JSON report to stdout (or --out-json)
- an Excel (.xlsx) summary table (default: results.xlsx next to input) if --excel is provided

Usage:
  python se_prepost_eval_optionB.py input.json
  python se_prepost_eval_optionB.py input.json --excel results.xlsx --out-json report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd


# ----------------------------
# Math utilities
# ----------------------------

def clamp01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    s = float(np.sum(v))
    if s < eps:
        return np.ones_like(v) / max(len(v), 1)
    return v / s


def exm(m: np.ndarray, w: np.ndarray) -> float:
    return float(np.dot(w, m))


def r_static(m: np.ndarray) -> float:
    return float(np.mean(m))


def se_posterior(exm_val: float, pT: float, gamma: float) -> float:
    # SE = ExM * pT + gamma * (1 - pT)
    return float(exm_val * pT + gamma * (1.0 - pT))


def abs_contrib(e: np.ndarray, w: np.ndarray) -> np.ndarray:
    # c_k = w_k * |e_k|
    return w * np.abs(e)


def rel_attrib(c: np.ndarray) -> np.ndarray:
    s = float(np.sum(c))
    if s < 1e-12:
        return np.zeros_like(c)
    return c / s


# ----------------------------
# Trust model (2-state Markov + HMM correction)
# ----------------------------

def markov_predict(b: np.ndarray, T: np.ndarray) -> np.ndarray:
    # b^- = T^T b
    return T.T @ b


def hmm_correct(b_pred: np.ndarray, O: np.ndarray) -> np.ndarray:
    return normalize(O * b_pred)


# ----------------------------
# Distances
# ----------------------------

def d_context(c_B: np.ndarray, c_T: np.ndarray) -> float:
    # mean absolute distance (normalized if your c is already in [0,1])
    return float(np.mean(np.abs(c_T - c_B)))


def d_attention(w_B: np.ndarray, w_T: np.ndarray) -> float:
    return float(np.mean(np.abs(w_T - w_B)))


def d_meta(m_hat: np.ndarray, m_B: np.ndarray) -> float:
    # NOTE: we don't require m_B in your simplified inputs. Kept for completeness.
    return float(np.mean(np.abs(m_hat - m_B)))


# ----------------------------
# Parsing helpers
# ----------------------------

def vec_from_dict(d: Dict[str, Any], meta_names: List[str]) -> np.ndarray:
    return np.array([clamp01(float(d.get(k, 0.0))) for k in meta_names], dtype=float)


def dict_from_vec(v: np.ndarray, meta_names: List[str]) -> Dict[str, float]:
    return {k: float(v[i]) for i, k in enumerate(meta_names)}


def vec_from_list(lst: List[Any], n: int) -> np.ndarray:
    if lst is None:
        raise ValueError("Expected list, got null")
    if len(lst) != n:
        raise ValueError(f"Expected list of length {n}, got {len(lst)}")
    return np.array([clamp01(float(x)) for x in lst], dtype=float)


# ----------------------------
# Core evaluation
# ----------------------------

def evaluate(root: Dict[str, Any]) -> Dict[str, Any]:
    meta_names = root.get("meta_names")
    if not meta_names or not isinstance(meta_names, list):
        raise ValueError("meta_names must be a non-empty list")

    K = len(meta_names)

    gamma = float(root.get("gamma", -0.4))
    p0 = float(root.get("p0", 0.5))

    target = root.get("target", {})
    w_T = vec_from_list(target.get("w_T"), K)
    w_T = normalize(np.clip(w_T, 0.0, None))

    c_T = vec_from_list(target.get("c_T"), 4)  # you currently use 4 components

    # Trust model
    tm = root.get("trust_model", {})
    taus = tm.get("taus")
    if isinstance(taus, dict):
        taus_vec = vec_from_dict(taus, meta_names)
    else:
        taus_vec = vec_from_list(taus, K)
    taus_vec = np.clip(taus_vec, 0.0, 1.0)

    p_match_T = float(tm.get("p_match_T", 0.8))
    p_match_D = float(tm.get("p_match_D", 0.3))

    T_raw = tm.get("T")
    if T_raw is None:
        # default: fairly sticky
        T = np.array([[0.85, 0.15], [0.20, 0.80]], dtype=float)
    else:
        T = np.array(T_raw, dtype=float)
        if T.shape != (2, 2):
            raise ValueError("trust_model.T must be 2x2")
        # row-normalize
        T[0] = normalize(T[0])
        T[1] = normalize(T[1])

    # Optional: weight update
    wu = root.get("weight_update", {}) or {}
    weight_update_enabled = bool(wu.get("enable", False))
    phi = float(wu.get("phi", 0.0))
    rigidity = float(np.clip(float(wu.get("rigidity", 1.0)), 0.0, 1.0))

    # Distance weights (for D_pre)
    dist_w = root.get("distance_weights", {"alpha": 0.0, "beta": 1.0, "gamma": 0.0})
    # In your simplified setting: D_pre = beta*D_c by default (context-only)
    alpha = float(dist_w.get("alpha", 0.0))
    beta = float(dist_w.get("beta", 1.0))
    gamma_d = float(dist_w.get("gamma", 0.0))

    out: Dict[str, Any] = {
        "meta_names": meta_names,
        "gamma": gamma,
        "p0": p0,
        "target": {
            "w_T": dict_from_vec(w_T, meta_names),
            "c_T": c_T.tolist(),
        },
        "trust_model": {
            "taus": dict_from_vec(taus_vec, meta_names),
            "p_match_T": p_match_T,
            "p_match_D": p_match_D,
            "T": T.tolist(),
        },
        "distance_weights": {"alpha": alpha, "beta": beta, "gamma": gamma_d},
        "weight_update": {"enable": weight_update_enabled, "phi": phi, "rigidity": rigidity},
        "policies": [],
    }

    policies = root.get("policies")
    if not policies or not isinstance(policies, list):
        raise ValueError("policies must be a non-empty list")

    for pol in policies:
        name = str(pol.get("name", "Unnamed"))
        justification = str(pol.get("justification", ""))

        # Base context vector for this analogy (provided by LLM / operator)
        c_B = vec_from_list(pol.get("c_B"), 4)

        m_hat = vec_from_dict(pol.get("m_hat"), meta_names)

        # Optional: policy-proposed attention (we do NOT use it by default for ExM/SE)
        w_hat = pol.get("w_hat", None)
        w_hat_vec: Optional[np.ndarray] = None
        if isinstance(w_hat, dict):
            w_hat_vec = normalize(np.clip(vec_from_dict(w_hat, meta_names), 0.0, None))
        elif isinstance(w_hat, list):
            w_hat_vec = normalize(np.clip(vec_from_list(w_hat, K), 0.0, None))

        # Confidence per meta-parameter (kappa)
        kappa = vec_from_dict(pol.get("kappa", {}), meta_names)

        # --- PRE ---
        Dc = d_context(c_B, c_T)
        Dw = d_attention(w_hat_vec, w_T) if w_hat_vec is not None else 0.0
        # Dm is not used pre because we don't have m_B in this simplified input.
        D_pre = alpha * 0.0 + beta * Dc + gamma_d * Dw

        ExM_hat = exm(m_hat, w_T)
        R_hat = r_static(m_hat)

        # Option B trust prior: pT_local_init_k = p0 * kappa_k
        pT_local_init_vec = p0 * kappa
        pT_local_init = dict_from_vec(pT_local_init_vec, meta_names)
        pT_global_init = float(np.dot(w_T, pT_local_init_vec))
        SE_hat = se_posterior(ExM_hat, pT_global_init, gamma)

        pol_out: Dict[str, Any] = {
            "name": name,
            "justification": justification,
            "c_B": c_B.tolist(),
            "pre": {
                "D_pre": float(D_pre),
                "m_hat": dict_from_vec(m_hat, meta_names),
                "ExM_hat": float(ExM_hat),
                "R_static_hat": float(R_hat),
                "kappa": dict_from_vec(kappa, meta_names),
                "pT_local_init": pT_local_init,
                "pT_global_init": float(pT_global_init),
                "SE_hat": float(SE_hat),
            },
        }

        # --- POST --- (if observed meta-params are provided)
        if pol.get("m_obs", None) is not None:
            m_obs = vec_from_dict(pol["m_obs"], meta_names)

            # evidence via threshold match
            e = m_obs - m_hat
            match = (np.abs(e) <= taus_vec)

            # Local trust belief update (same structure as your simulator)
            # b_k = [p(T), p(D)]
            b = np.tile(np.array([pT_local_init_vec[0], 1.0 - pT_local_init_vec[0]], dtype=float), (K, 1))
            # NOTE: we want per-dimension priors, so set b[k] accordingly
            for k in range(K):
                b[k] = np.array([pT_local_init_vec[k], 1.0 - pT_local_init_vec[k]], dtype=float)

            pT_local_post_vec = np.zeros(K, dtype=float)
            for k in range(K):
                b_pred = markov_predict(b[k], T)

                O = np.array([
                    p_match_T if match[k] else (1.0 - p_match_T),
                    p_match_D if match[k] else (1.0 - p_match_D),
                ], dtype=float)

                b[k] = hmm_correct(b_pred, O)
                pT_local_post_vec[k] = float(b[k, 0])

            pT_global_post = float(np.dot(w_T, pT_local_post_vec))

            ExM_obs = exm(m_obs, w_T)
            R_obs = r_static(m_obs)
            SE_obs = se_posterior(ExM_obs, pT_global_post, gamma)

            # Attribution for responsibility: rho_k proportional to w_k * |e_k|
            c_abs = abs_contrib(e, w_T)
            rho = rel_attrib(c_abs)

            # Optional weight calibration (post)
            calibration = None
            if weight_update_enabled and phi > 0.0:
                delta = float(SE_obs - ExM_obs)
                g = m_obs.copy()  # g_k = m_obs_k

                w_candidate = w_T + phi * rho * delta * g
                w_candidate = normalize(np.clip(w_candidate, 0.0, None))

                w_cal = rigidity * w_T + (1.0 - rigidity) * w_candidate
                w_cal = normalize(np.clip(w_cal, 0.0, None))

                ExM_obs_cal = exm(m_obs, w_cal)
                pT_global_post_cal = float(np.dot(w_cal, pT_local_post_vec))
                SE_obs_cal = se_posterior(ExM_obs_cal, pT_global_post_cal, gamma)

                calibration = {
                    "delta_SE_minus_ExM": delta,
                    "w_candidate": dict_from_vec(w_candidate, meta_names),
                    "w_calibrated": dict_from_vec(w_cal, meta_names),
                    "ExM_obs_calibrated": float(ExM_obs_cal),
                    "pT_global_post_calibrated": float(pT_global_post_cal),
                    "SE_obs_calibrated": float(SE_obs_cal),
                }

            pol_out["post"] = {
                "m_obs": dict_from_vec(m_obs, meta_names),
                "ExM_obs": float(ExM_obs),
                "R_static_obs": float(R_obs),
                "pT_local_post": dict_from_vec(pT_local_post_vec, meta_names),
                "pT_global_post": float(pT_global_post),
                "SE_obs": float(SE_obs),
                "match": {meta_names[i]: bool(match[i]) for i in range(K)},
                "e": dict_from_vec(e, meta_names),
                "rho": dict_from_vec(rho, meta_names),
                "calibration": calibration,
            }

        out["policies"].append(pol_out)

    return out


def to_dataframe(report: Dict[str, Any]) -> pd.DataFrame:
    meta_names: List[str] = report["meta_names"]

    rows: List[Dict[str, Any]] = []
    for pol in report["policies"]:
        pre = pol.get("pre", {})
        post = pol.get("post", {})
        cal = post.get("calibration") or {}

        row: Dict[str, Any] = {
            "name": pol.get("name"),
            "justification": pol.get("justification"),
            "D_pre": pre.get("D_pre"),
            "ExM_hat": pre.get("ExM_hat"),
            "R_static_hat": pre.get("R_static_hat"),
            "pT_global_init": pre.get("pT_global_init"),
            "SE_hat": pre.get("SE_hat"),
            "ExM_obs": post.get("ExM_obs"),
            "R_static_obs": post.get("R_static_obs"),
            "pT_global_post": post.get("pT_global_post"),
            "SE_obs": post.get("SE_obs"),
            "ExM_obs_calibrated": cal.get("ExM_obs_calibrated"),
            "pT_global_post_calibrated": cal.get("pT_global_post_calibrated"),
            "SE_obs_calibrated": cal.get("SE_obs_calibrated"),
        }

        for m in meta_names:
            row[f"m_hat_{m}"] = pre.get("m_hat", {}).get(m)
            row[f"kappa_{m}"] = pre.get("kappa", {}).get(m)
            row[f"pT_local_init_{m}"] = pre.get("pT_local_init", {}).get(m)

            if post:
                row[f"m_obs_{m}"] = post.get("m_obs", {}).get(m)
                row[f"pT_local_post_{m}"] = post.get("pT_local_post", {}).get(m)
                row[f"match_{m}"] = post.get("match", {}).get(m)
                row[f"e_{m}"] = post.get("e", {}).get(m)
                row[f"rho_{m}"] = post.get("rho", {}).get(m)

            if cal:
                row[f"w_cal_{m}"] = cal.get("w_calibrated", {}).get(m)

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="Input JSON file")
    ap.add_argument("--out-json", type=str, default="", help="Write full JSON report to this path")
    ap.add_argument("--excel", type=str, default="", help="Write Excel summary (.xlsx) to this path")
    args = ap.parse_args()

    in_path = Path(args.input)
    root = json.loads(in_path.read_text(encoding="utf-8"))

    report = evaluate(root)

    # JSON output
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))

    # Excel output
    if args.excel:
        df = to_dataframe(report)
        out_xlsx = Path(args.excel)
        out_xlsx.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(out_xlsx, index=False)


if __name__ == "__main__":
    main()
