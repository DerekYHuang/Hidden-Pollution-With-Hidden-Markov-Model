'''
Goal: Use Viterbi to find the most probable sequence of rainy/not rainy given the sequence of observables (UCI dataset)

Dataset: AirQualityUCI.csv
 - Relevant features ['CO', 'NO2', 'O3']
 - data points in 1 hour intervals, ordered chronologically (our sequence of observables)

HMM:
 - Observables: ['CO', 'NO2', 'O3']
 - Hidden States: [0, 1] # [not rainy, rainy]
 - Transition Probs: [0.71428, 0.285714,
                     0.485981, 0.514019]
 - P(Rain | AQI_Category) = {1: 0.40404040404040403, 2: 0.3492063492063492, 3: None, 4: None} # {key=aqi_category, value=probability}, Note: None can be assumed to be 0
 
Steps:
1. Use CO, NO2, O3 to calculate AQI category for dataset, add as new column called 'AQI_Category'
    - 1 is 'Good', AQI of 0-50
    - 2 is 'Moderate', AQI of 51-100
    - 3 is 'Unhealthy_Sensitive', AQI of 101-150
    - 4 is 'Unhealthy', AQI of 151-200
    - 5 is 'Very_Unhealthy' 201-300
    - 6 is 'Hazardous' 301+
2. Generate rainy (1), not rainy (0) labels for dataset using transition probability and P(Rain | AQI_Category), add as new column called 'label'
3. Run Viterbi on HMM to generate most probable sequence of rainy/not rainy given the observables
4. Evaluate Viterbi sequence against generated labels
'''


import math
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# EPA AQI breakpoints for sub-index calculation
BREAKPOINTS = {
    "PM2.5": [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ],
    "PM10": [
        (0, 54, 0, 50),
        (55, 154, 51, 100),
        (155, 254, 101, 150),
        (255, 354, 151, 200),
        (355, 424, 201, 300),
        (425, 504, 301, 400),
        (505, 604, 401, 500),
    ],
    "O3": [
        (0.000, 0.054, 0, 50),
        (0.055, 0.070, 51, 100),
        (0.071, 0.085, 101, 150),
        (0.086, 0.105, 151, 200),
        (0.106, 0.200, 201, 300),
    ],
    "NO2": [
        (0, 53, 0, 50),
        (54, 100, 51, 100),
        (101, 360, 101, 150),
        (361, 649, 151, 200),
        (650, 1249, 201, 300),
        (1250, 1649, 301, 400),
        (1650, 2049, 401, 500),
    ],
    "SO2": [
        (0, 35, 0, 50),
        (36, 75, 51, 100),
        (76, 185, 101, 150),
        (186, 304, 151, 200),
        (305, 604, 201, 300),
        (605, 804, 301, 400),
        (805, 1004, 401, 500),
    ],
    "CO": [
        (0.0, 4.4, 0, 50),
        (4.5, 9.4, 51, 100),
        (9.5, 12.4, 101, 150),
        (12.5, 15.4, 151, 200),
        (15.5, 30.4, 201, 300),
        (30.5, 40.4, 301, 400),
        (40.5, 50.4, 401, 500),
    ],
}


def compute_subindex(concentration: float, breakpoints: List[Tuple[float, float, int, int]]) -> Optional[float]:
    """
    Compute AQI sub-index via linear interpolation over EPA breakpoints.
    Returns None if concentration is NaN. Caps result to [0, 500].
    """
    if concentration is None or (isinstance(concentration, float) and math.isnan(concentration)):
        return None
    # Find the interval that contains the concentration
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= concentration <= c_hi:
            # Linear interpolation
            return ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
    # If above last breakpoint, extrapolate with last segment and cap at 500
    c_lo, c_hi, i_lo, i_hi = breakpoints[-1]
    if concentration > c_hi:
        val = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
        return min(500.0, max(0.0, val))
    # If below first breakpoint, extrapolate with first segment and floor at 0
    c_lo, c_hi, i_lo, i_hi = breakpoints[0]
    if concentration < c_lo:
        val = ((i_hi - i_lo) / (c_hi - c_lo)) * (concentration - c_lo) + i_lo
        return min(500.0, max(0.0, val))
    return None


def compute_aqi_row(row: pd.Series) -> Tuple[Optional[float], Optional[str]]:
    """
    Compute overall AQI (max sub-index) and dominant pollutant from a row with
    columns 'CO', 'NO2', 'O3'.

    Note: Units must match the breakpoints above. If units differ, convert prior.
    """
    subindices = {}
    if 'CO' in row:
        subindices['CO'] = compute_subindex(float(row['CO']), BREAKPOINTS['CO'])
    if 'NO2' in row:
        subindices['NO2'] = compute_subindex(float(row['NO2']), BREAKPOINTS['NO2'])
    if 'O3' in row:
        subindices['O3'] = compute_subindex(float(row['O3']), BREAKPOINTS['O3'])

    # Filter Nones
    valid = {k: v for k, v in subindices.items() if v is not None}
    if not valid:
        return None, None
    dominant = max(valid, key=valid.get)
    return float(valid[dominant]), str(dominant)


def aqi_to_category(aqi: Optional[float]) -> Optional[int]:
    if aqi is None or (isinstance(aqi, float) and math.isnan(aqi)):
        return None
    if 0 <= aqi <= 50:
        return 1
    if 51 <= aqi <= 100:
        return 2
    if 101 <= aqi <= 150:
        return 3
    if 151 <= aqi <= 200:
        return 4
    if 201 <= aqi <= 300:
        return 5
    return 6


def stationary_distribution(A: np.ndarray) -> np.ndarray:
    """Closed-form stationary distribution for a 2x2 transition matrix."""
    a00, a01 = A[0, 0], A[0, 1]
    a10, a11 = A[1, 0], A[1, 1]
    denom = a01 + a10
    if denom == 0:
        return np.array([0.5, 0.5])
    pi0 = a10 / denom
    pi1 = a01 / denom
    return np.array([pi0, pi1])


def ffbs_sample_labels(A: np.ndarray, log_emissions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Forward-Filtering Backward-Sampling to draw a hidden state sequence given
    transition matrix A (2x2) and per-time log emission weights log_emissions
    of shape (T, 2). Returns int states in {0,1} of length T.
    """
    T = log_emissions.shape[0]
    # Forward pass: compute alphas (filtered distributions)
    alphas = np.zeros((T, 2), dtype=float)
    pi = stationary_distribution(A)
    # t=0
    alpha0_unnorm = np.log(pi + 1e-15) + log_emissions[0]
    # normalize
    m = np.max(alpha0_unnorm)
    a0 = np.exp(alpha0_unnorm - m)
    a0 = a0 / a0.sum()
    alphas[0] = a0
    # transitions for forward: p(z_t | z_{t-1}) = A[z_{t-1}, z_t]
    logA_T = np.log(A.T + 1e-15)  # shape (2,2): columns correspond to next state
    for t in range(1, T):
        # predictive: sum_k alpha[t-1,k] * A[k, :]
        pred = np.log(alphas[t - 1] + 1e-15) + 0.0
        # log-sum-exp over previous states for each next state j
        log_pred_next = np.array([
            np.logaddexp.reduce(pred + logA_T[:, j]) for j in range(2)
        ])
        log_alpha_unnorm = log_pred_next + log_emissions[t]
        m = np.max(log_alpha_unnorm)
        a = np.exp(log_alpha_unnorm - m)
        alphas[t] = a / a.sum()

    # Backward sampling
    states = np.zeros(T, dtype=int)
    states[T - 1] = rng.choice(2, p=alphas[T - 1])
    logA = np.log(A + 1e-15)
    for t in range(T - 2, -1, -1):
        # p(z_t = i | z_{t+1}=j, y_{1:T}) ∝ alpha_t(i) * A[i, j]
        j = states[t + 1]
        log_p = np.log(alphas[t] + 1e-15) + logA[:, j]
        m = np.max(log_p)
        p = np.exp(log_p - m)
        p = p / p.sum()
        states[t] = rng.choice(2, p=p)
    return states


def viterbi_decode(A: np.ndarray, log_emissions: np.ndarray) -> np.ndarray:
    """Viterbi decoding for a 2-state HMM given transition matrix A and per-time log emissions."""
    T = log_emissions.shape[0]
    logA = np.log(A + 1e-15)
    pi = stationary_distribution(A)
    log_pi = np.log(pi + 1e-15)

    delta = np.zeros((T, 2), dtype=float)
    psi = np.zeros((T, 2), dtype=int)

    delta[0] = log_pi + log_emissions[0]
    for t in range(1, T):
        for j in range(2):
            vals = delta[t - 1] + logA[:, j]
            psi[t, j] = int(np.argmax(vals))
            delta[t, j] = float(np.max(vals)) + log_emissions[t, j]

    states = np.zeros(T, dtype=int)
    states[T - 1] = int(np.argmax(delta[T - 1]))
    for t in range(T - 2, -1, -1):
        states[t] = int(psi[t + 1, states[t + 1]])
    return states


def evaluate_sequence(true_labels: np.ndarray, pred_labels: np.ndarray) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and F1 for class 1."""
    assert true_labels.shape == pred_labels.shape
    tp = int(((true_labels == 1) & (pred_labels == 1)).sum())
    tn = int(((true_labels == 0) & (pred_labels == 0)).sum())
    fp = int(((true_labels == 0) & (pred_labels == 1)).sum())
    fn = int(((true_labels == 1) & (pred_labels == 0)).sum())
    acc = (tp + tn) / max(1, (tp + tn + fp + fn))
    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))
    f1 = 2 * prec * rec / max(1e-15, (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def build_emission_log_probs(aqi_categories: pd.Series, p_rain_by_cat: Dict[int, float]) -> np.ndarray:
    """
    Build per-time log emission weights for states {0,1} from AQI categories and given
    P(rain | category). We use e_t(1)=p, e_t(0)=1-p as pseudo-likelihoods.
    Returns array of shape (T, 2).
    """
    p = aqi_categories.map(lambda c: float(p_rain_by_cat.get(int(c), 0.0)) if pd.notna(c) else 0.0).astype(float).values
    p = np.clip(p, 1e-6, 1 - 1e-6)  # avoid log(0)
    e1 = np.log(p)
    e0 = np.log(1.0 - p)
    return np.stack([e0, e1], axis=1)


def run_hmm_pipeline(
    csv_path: str = "../data/processed/airquality_cleaned.csv",
    transition_matrix: Optional[np.ndarray] = None,
    p_rain_by_cat: Optional[Dict[int, Optional[float]]] = None,
    seed: int = 42,
) -> Dict[str, object]:
    """
    End-to-end pipeline:
    - Load data and compute AQI, AQI_Category from CO, NO2, O3
    - Generate synthetic labels via FFBS given transitions and emission weights
    - Decode via Viterbi on the same model
    - Evaluate and return results
    """
    if transition_matrix is None:
        transition_matrix = np.array([[0.71428, 0.285714], [0.485981, 0.514019]], dtype=float)
    if p_rain_by_cat is None:
        p_rain_by_cat = {1: 0.40404040404040403, 2: 0.3492063492063492, 3: 0.0, 4: 0.0}

    # Normalize any None -> 0.0 and ensure categories 1..6 exist (default 0)
    p_map: Dict[int, float] = {int(k): float(v) if v is not None else 0.0 for k, v in p_rain_by_cat.items()}
    for k in [1, 2, 3, 4, 5, 6]:
        if k not in p_map:
            p_map[k] = 0.0

    df = pd.read_csv(csv_path)
    # Compute AQI and category
    aqi_vals: List[Optional[float]] = []
    aqi_dom: List[Optional[str]] = []
    for _, row in df.iterrows():
        aqi, dom = compute_aqi_row(row)
        aqi_vals.append(aqi)
        aqi_dom.append(dom)
    df['AQI'] = aqi_vals
    df['AQI_Dominant'] = aqi_dom
    df['AQI_Category'] = df['AQI'].map(aqi_to_category)

    # Build emission log-probs
    log_emissions = build_emission_log_probs(df['AQI_Category'], p_map)

    # Generate synthetic labels by FFBS sampling
    rng = np.random.default_rng(seed)
    labels = ffbs_sample_labels(transition_matrix, log_emissions, rng)
    df['label'] = labels

    # Viterbi decoding
    viterbi_states = viterbi_decode(transition_matrix, log_emissions)
    df['viterbi'] = viterbi_states

    # Evaluation
    metrics = evaluate_sequence(df['label'].to_numpy(), df['viterbi'].to_numpy())

    return {
        "data": df,
        "metrics": metrics,
        "transition_matrix": transition_matrix,
        "p_rain_by_cat": p_map,
    }


if __name__ == "__main__":
    results = run_hmm_pipeline()
    print("Metrics:", results["metrics"]) 
    # Show a small preview
    print(results["data"][['CO', 'NO2', 'O3', 'AQI', 'AQI_Category', 'label', 'viterbi']].head())

