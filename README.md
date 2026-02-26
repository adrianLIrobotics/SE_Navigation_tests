# calculations_calibrates

Compact toolkit for simulating and evaluating subjective experience (SE) under trust dynamics.

This repo includes:
- A Streamlit simulator (`app.py`) for interactive SE/trust trajectories.
- A CLI evaluator (`se_prepost_eval_optionB.py`) for pre/post analogy analysis and exports.
- Example JSON inputs (`analogy*.json`, `analogy*_optionB.json`).
- Example outputs (`report.json`, `results*.xlsx`).

## Requirements

- Python 3.10+ (recommended)
- Dependencies in `requirements.txt`:
  - `streamlit`
  - `numpy`
  - `pandas`
  - `matplotlib`

Install:

```bash
pip install -r requirements.txt
```

## Run the Streamlit Simulator

```bash
streamlit run app.py
```

What you get:
- Interactive controls for trust model, priors, adaptation, and policy JSON.
- Multi-plot dashboard in 2-column layout.
- Expected vs observed traces for `SE` and `ExM`.
- Detailed per-policy trace tables.

## Run the Option B CLI Evaluator

The evaluator reads an Option B JSON and computes:
- Pre values (`SE_hat`, `ExM_hat`, initial trust).
- Post values (`SE_obs`, `ExM_obs`, posterior trust).
- Optional weight calibration (`w_candidate`, `w_calibrated`).

Usage:

```bash
python se_prepost_eval_optionB.py <input.json>
```

With export files:

```bash
python se_prepost_eval_optionB.py analogy1_optionB.json --out-json report.json --excel results.xlsx
```

## Input JSON Formats

### 1) Simulator format (`app.py`)

Top-level JSON is a list of policies:

```json
[
  {
    "name": "Policy name",
    "m_pred": {"Efficiency": 0.6, "Comfort": 0.8, "TaskCompletion": 1.0},
    "m_obs": {"Efficiency": 0.5, "Comfort": 0.7, "TaskCompletion": 0.9},
    "m_obs_schedule": [
      {"t": 0, "Efficiency": 0.5, "Comfort": 0.7, "TaskCompletion": 0.9},
      {"t": 20, "Efficiency": 0.3}
    ]
  }
]
```

`m_obs_schedule` is optional and lets observed meta-parameters change over time.

### 2) Option B format (`se_prepost_eval_optionB.py`)

Top-level JSON is an object with model settings and policy list:

```json
{
  "meta_names": ["Efficiency", "Safety", "TaskCompletion"],
  "gamma": -0.4,
  "p0": 0.5,
  "target": {
    "w_T": [0.25, 0.45, 0.3],
    "c_T": [0.4, 0.2, 1.0, 1.0]
  },
  "policies": [
    {
      "name": "Analogy 1",
      "m_hat": {"Efficiency": 0.85, "Safety": 0.8, "TaskCompletion": 0.9},
      "c_B": [1.0, 0.0, 1.0, 1.0],
      "kappa": {"Efficiency": 0.7, "Safety": 0.55, "TaskCompletion": 0.65},
      "m_obs": {"Efficiency": 0.624, "Safety": 1.0, "TaskCompletion": 1.0}
    }
  ],
  "trust_model": {
    "taus": {"Efficiency": 0.2, "Safety": 0.2, "TaskCompletion": 0.2},
    "p_match_T": 0.8,
    "p_match_D": 0.3,
    "T": [[0.85, 0.15], [0.2, 0.8]]
  },
  "weight_update": {"enable": true, "phi": 1.0, "rigidity": 0.0}
}
```

## Repository Structure

- `app.py`: Streamlit simulator (main interactive app).
- `app2.py`: Alternate/snapshot version of app logic.
- `se_prepost_eval_optionB.py`: CLI evaluator for Option B.
- `analogy1.json`, `analogy2.json`, `analogy3.json`: simple simulator examples.
- `analogy1_optionB.json`, `analogy2_optionB.json`, `analogy3_optionB.json`: Option B examples.
- `report.json`: sample evaluator output.
- `results.xlsx`, `results2.xlsx`, `results3.xlsx`: sample tabular outputs.

## Notes

- Trust values and probabilities are clamped/normalized where needed.
- `gamma` is expected to be negative (distrust penalty).
- If you change `meta_names`, ensure all policy keys match those names.
