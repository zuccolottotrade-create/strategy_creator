from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CriterionResult:
    name: str
    label: str
    reason: str
    metrics: Dict[str, Any]
    thresholds: Dict[str, Any]



CriterionFn = Callable[[pd.DataFrame, Dict[str, Any]], CriterionResult]
_REGISTRY: List[CriterionFn] = []


def register(fn: CriterionFn) -> CriterionFn:
    _REGISTRY.append(fn)
    return fn


def get_registered_criteria() -> List[CriterionFn]:
    return list(_REGISTRY)


def run_all_criteria(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[CriterionResult]:
    return [fn(df, cfg) for fn in get_registered_criteria()]


DEFAULT_CFG: Dict[str, Any] = {
    "windows": {
        "vol_window": 50,
        "atr_window": 14,
        "trend_window": 120,
        "structure_window": 120,
        "news_window": 250,
    },
    "liquidity": {
        "vol_avg_high": 1_000_000,
        "vol_avg_low": 100_000,
        "cv_low": 0.8,
        "cv_high": 1.5,
    },
    "volatility": {
        "ret_std_high": 0.012,
        "ret_std_low": 0.005,
        "atr_pct_high": 0.020,
        "atr_pct_low": 0.008,
    },
    "directionality": {
        "r2_trend": 0.35,
        "r2_range": 0.15,
        "slope_norm_trend": 0.0010,
        "net_to_range_trend": 0.35,
        "net_to_range_range": 0.18,
    },
    "structure": {
        "body_ratio_impulsive": 0.55,
        "body_ratio_oscillating": 0.40,
        "alt_rate_oscillating": 0.55,
    },
    "news_reaction": {
        "spike_k": 3.0,
        "spike_freq_strong": 0.020,
        "spike_freq_weak": 0.008,
        "gap_pct": 0.006,
        "gap_freq_strong": 0.015,
        "gap_freq_weak": 0.006,
    },
    "horizon": {
        "runlen_position": 18,
        "runlen_swing": 8,
    },
}


def require_columns(df: pd.DataFrame, cols: List[str], crit_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"[{crit_name}] Colonne mancanti: {missing}. Disponibili: {list(df.columns)}")


def to_numeric_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    ss = s.astype(str).str.replace(",", ".", regex=False)
    return pd.to_numeric(ss, errors="coerce")


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    try:
        if b is None or b == 0 or (isinstance(b, float) and np.isnan(b)):
            return float(default)
        return float(a / b)
    except Exception:
        return float(default)


def get_close(df: pd.DataFrame) -> pd.Series:
    return to_numeric_series(df["close"])


def get_open(df: pd.DataFrame) -> pd.Series:
    return to_numeric_series(df["open"])


def get_high(df: pd.DataFrame) -> pd.Series:
    return to_numeric_series(df["high"])


def get_low(df: pd.DataFrame) -> pd.Series:
    return to_numeric_series(df["low"])


def get_volume(df: pd.DataFrame) -> pd.Series:
    return to_numeric_series(df["volume"])


def ensure_datetime(df: pd.DataFrame) -> Optional[pd.Series]:
    if "datetime" in df.columns:
        return pd.to_datetime(df["datetime"], errors="coerce")
    if "date" in df.columns and "time" in df.columns:
        return pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str), errors="coerce")
    return None


def infer_timeframe_minutes(df: pd.DataFrame) -> Optional[float]:
    dt = ensure_datetime(df)
    if dt is None:
        return None
    dt = dt.dropna()
    if len(dt) < 3:
        return None
    deltas = dt.sort_values().diff().dropna().dt.total_seconds() / 60.0
    if deltas.empty:
        return None
    return float(np.nanmedian(deltas.values))


def compute_returns(close: pd.Series) -> pd.Series:
    return close.pct_change()


def atr(df: pd.DataFrame, window: int) -> pd.Series:
    high = get_high(df)
    low = get_low(df)
    close = get_close(df)
    prev_close = close.shift(1)

    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    return tr.rolling(window, min_periods=max(3, window // 2)).mean()


def linear_trend_strength(close: pd.Series, window: int) -> Tuple[float, float]:
    y = close.astype(float).values
    msk = np.isfinite(y)
    y = y[msk]
    if len(y) < max(20, window // 2):
        return 0.0, 0.0
    y = y[-window:] if len(y) >= window else y
    x = np.arange(len(y), dtype=float)

    slope, intercept = np.polyfit(x, y, 1)
    y_hat = slope * x + intercept
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - safe_div(ss_res, ss_tot, default=1.0) if ss_tot > 0 else 0.0
    r2 = float(max(0.0, min(1.0, r2)))
    return float(slope), r2


def alternation_rate(returns: pd.Series) -> float:
    r = returns.dropna().values
    if len(r) < 5:
        return 0.0
    s = np.sign(r).astype(float)
    s[s == 0] = np.nan
    s = s[~np.isnan(s)]
    if len(s) < 5:
        return 0.0
    changes = np.sum(s[1:] != s[:-1])
    return float(changes / max(1, len(s) - 1))


def avg_body_ratio(df: pd.DataFrame, window: int) -> float:
    o = get_open(df)
    c = get_close(df)
    h = get_high(df)
    l = get_low(df)
    rng = (h - l).abs().replace(0, np.nan)
    body = (c - o).abs()
    ratio = (body / rng).replace([np.inf, -np.inf], np.nan).dropna()
    if ratio.empty:
        return 0.0
    tail = ratio.tail(window) if len(ratio) > window else ratio
    return float(np.nanmean(tail.values))


def net_to_range(df: pd.DataFrame, window: int) -> float:
    c = get_close(df).dropna()
    h = get_high(df).dropna()
    l = get_low(df).dropna()
    if len(c) < 10:
        return 0.0
    w = window if len(c) >= window else len(c)
    c_w = c.tail(w)
    h_w = h.tail(w)
    l_w = l.tail(w)
    net = float(abs(c_w.iloc[-1] - c_w.iloc[0]))
    rng = float(h_w.max() - l_w.min())
    return safe_div(net, rng, default=0.0)


@register
def c_liquidita(df: pd.DataFrame, cfg: Dict[str, Any]) -> CriterionResult:
    name = "Liquidità"
    require_columns(df, ["volume"], name)

    vol = get_volume(df).dropna()
    vol_avg = float(vol.mean()) if not vol.empty else 0.0
    vol_std = float(vol.std(ddof=0)) if len(vol) > 2 else 0.0
    cv = safe_div(vol_std, vol_avg, default=0.0)

    th = cfg.get("liquidity", {})
    high = float(th.get("vol_avg_high", 1_000_000))
    low = float(th.get("vol_avg_low", 100_000))
    cv_high = float(th.get("cv_high", 1.5))

    if vol_avg >= high and cv <= cv_high:
        label = "alta"
    elif vol_avg <= low or cv >= cv_high:
        label = "bassa"
    else:
        label = "media"

    reason = f"Volume medio={vol_avg:,.0f}, CV={cv:.2f} (std={vol_std:,.0f})."
    return CriterionResult(
        name=name,
        label=label,
        reason=reason,
        metrics={"vol_avg": vol_avg, "vol_std": vol_std, "cv": cv, "n": int(len(vol))},
        thresholds={},
    )


@register
def c_volatilita(df: pd.DataFrame, cfg: Dict[str, Any]) -> CriterionResult:
    name = "Volatilità"
    require_columns(df, ["high", "low", "close"], name)

    close = get_close(df)
    rets = compute_returns(close)
    w = int(cfg["windows"]["vol_window"])
    ret_std = float(rets.tail(w).std(ddof=0)) if rets.dropna().shape[0] >= 5 else 0.0

    atr_w = int(cfg["windows"]["atr_window"])
    atr_s = atr(df, atr_w)
    last_atr = float(atr_s.dropna().iloc[-1]) if atr_s.dropna().shape[0] else 0.0
    last_close = float(close.dropna().iloc[-1]) if close.dropna().shape[0] else 0.0
    atr_pct = safe_div(last_atr, last_close, default=0.0)

    th = cfg.get("volatility", {})
    ret_high = float(th.get("ret_std_high", 0.012))
    ret_low = float(th.get("ret_std_low", 0.005))
    atr_high = float(th.get("atr_pct_high", 0.020))
    atr_low = float(th.get("atr_pct_low", 0.008))

    if (ret_std >= ret_high) or (atr_pct >= atr_high):
        label = "alta"
    elif (ret_std <= ret_low) and (atr_pct <= atr_low):
        label = "bassa"
    else:
        label = "media"

    reason = f"Std returns({w})={ret_std:.4f}, ATR%({atr_w})={atr_pct:.3f}."
    return CriterionResult(
        name=name,
        label=label,
        reason=reason,
        metrics={"ret_std": ret_std, "atr": last_atr, "atr_pct": atr_pct, "close_last": last_close},
        thresholds={},
    )


@register
def c_direzionalita(df: pd.DataFrame, cfg: Dict[str, Any]) -> CriterionResult:
    name = "Direzionalità prevalente"
    require_columns(df, ["high", "low", "close"], name)

    close = get_close(df).dropna()
    tw = int(cfg["windows"]["trend_window"])
    slope, r2 = linear_trend_strength(close, tw)
    close_last = float(close.iloc[-1]) if len(close) else 0.0
    slope_norm = safe_div(slope, close_last, default=0.0)
    ntr = net_to_range(df, tw)

    th = cfg.get("directionality", {})
    r2_trend = float(th.get("r2_trend", 0.35))
    r2_range = float(th.get("r2_range", 0.15))
    slope_norm_trend = float(th.get("slope_norm_trend", 0.0010))
    ntr_trend = float(th.get("net_to_range_trend", 0.35))
    ntr_range = float(th.get("net_to_range_range", 0.18))

    is_trend = (r2 >= r2_trend) and (abs(slope_norm) >= slope_norm_trend) and (ntr >= ntr_trend)
    is_range = (r2 <= r2_range) and (ntr <= ntr_range)

    label = "trend" if is_trend else ("range" if is_range else "mista")
    reason = f"R²={r2:.2f}, slope%/bar={slope_norm:.4f}, net/range={ntr:.2f} (win={tw})."
    return CriterionResult(
        name=name,
        label=label,
        reason=reason,
        metrics={"r2": r2, "slope": slope, "slope_norm": slope_norm, "net_to_range": ntr, "trend_window": tw},
        thresholds={},
    )


@register
def c_struttura(df: pd.DataFrame, cfg: Dict[str, Any]) -> CriterionResult:
    name = "Struttura movimenti"
    require_columns(df, ["open", "high", "low", "close"], name)

    sw = int(cfg["windows"]["structure_window"])
    close = get_close(df)
    rets = compute_returns(close)
    body_ratio = avg_body_ratio(df, sw)
    alt_rate = alternation_rate(rets.tail(sw))

    th = cfg.get("structure", {})
    br_imp = float(th.get("body_ratio_impulsive", 0.55))
    alt_osc = float(th.get("alt_rate_oscillating", 0.55))

    if (alt_rate >= alt_osc) and (body_ratio <= br_imp):
        label = "oscillante"
    else:
        label = "impulsiva" if body_ratio >= br_imp else "oscillante"

    reason = f"Body/Range={body_ratio:.2f}, alternanza={alt_rate:.2f} (win={sw})."
    return CriterionResult(
        name=name,
        label=label,
        reason=reason,
        metrics={"body_ratio": body_ratio, "alt_rate": alt_rate, "window": sw},
        thresholds={},
    )


@register
def c_news(df: pd.DataFrame, cfg: Dict[str, Any]) -> CriterionResult:
    name = "Reazione a news/eventi"
    require_columns(df, ["open", "close"], name)

    close = get_close(df)
    open_ = get_open(df)

    nw = int(cfg["windows"]["news_window"])
    rets = compute_returns(close).dropna().tail(nw)
    std = float(rets.std(ddof=0)) if len(rets) >= 10 else 0.0

    th = cfg.get("news_reaction", {})
    k = float(th.get("spike_k", 3.0))
    gap_pct_th = float(th.get("gap_pct", 0.006))

    spike = (rets.abs() > (k * std)) if std > 0 else pd.Series(False, index=rets.index)
    spike_freq = float(spike.mean()) if len(spike) else 0.0

    prev_close = close.shift(1)
    gap_pct = ((open_ - prev_close).abs() / prev_close.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    gap = gap_pct > gap_pct_th
    gap_freq = float(gap.tail(nw).mean()) if gap.dropna().shape[0] else 0.0

    spike_strong = float(th.get("spike_freq_strong", 0.020))
    spike_weak = float(th.get("spike_freq_weak", 0.008))
    gap_strong = float(th.get("gap_freq_strong", 0.015))
    gap_weak = float(th.get("gap_freq_weak", 0.006))

    if (spike_freq >= spike_strong) or (gap_freq >= gap_strong):
        label = "forte"
    elif (spike_freq <= spike_weak) and (gap_freq <= gap_weak):
        label = "debole"
    else:
        label = "moderata"

    reason = f"Spike freq={spike_freq:.3f} (k={k}, std={std:.4f}), Gap freq={gap_freq:.3f} (>{gap_pct_th:.3f})."
    return CriterionResult(
        name=name,
        label=label,
        reason=reason,
        metrics={"spike_freq": spike_freq, "gap_freq": gap_freq, "ret_std": std, "k": k, "gap_pct_th": gap_pct_th,
                 "window": nw},
        thresholds={},
    )


@register
def c_orizzonte(df: pd.DataFrame, cfg: Dict[str, Any]) -> CriterionResult:
    name = "Orizzonte naturale"
    require_columns(df, ["close"], name)

    tf_min = infer_timeframe_minutes(df)
    close = get_close(df)
    rets = compute_returns(close).dropna()

    s = np.sign(rets.values).astype(float)
    s[s == 0] = np.nan
    s = s[~np.isnan(s)]
    run_lengths: List[int] = []
    if len(s) >= 5:
        cur = 1
        for i in range(1, len(s)):
            if s[i] == s[i - 1]:
                cur += 1
            else:
                run_lengths.append(cur)
                cur = 1
        run_lengths.append(cur)
    avg_run = float(np.mean(run_lengths)) if run_lengths else 0.0

    hcfg = cfg.get("horizon", {})
    run_pos = int(hcfg.get("runlen_position", 18))
    run_swing = int(hcfg.get("runlen_swing", 8))

    if tf_min is not None and tf_min <= 60:
        label = "swing" if avg_run >= run_pos else "intraday"
        tf_txt = f"{tf_min:.0f}m"
    else:
        if avg_run >= run_pos:
            label = "position"
        elif avg_run >= run_swing:
            label = "swing"
        else:
            label = "swing"
        tf_txt = "n/a" if tf_min is None else f"{tf_min:.0f}m"

    reason = f"Timeframe stimato={tf_txt}, run length medio={avg_run:.1f} (pos≥{run_pos}, swing≥{run_swing})."
    return CriterionResult(
        name=name,
        label=label,
        reason=reason,
        metrics={"timeframe_min": tf_min, "avg_run_length": avg_run, "runlen_position": run_pos,
                 "runlen_swing": run_swing},
        thresholds={},
    )
