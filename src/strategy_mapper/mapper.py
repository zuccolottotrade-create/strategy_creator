from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


# ----------------------------
# Base strategy families (side-neutral)
# ----------------------------
F_TREND = "Trend_Following"
F_PULLBACK = "Pullback_in_Trend"
F_BREAKOUT = "Breakout"
F_MEANREV = "Mean_Reversion"
F_RANGE = "Range_Trading"
F_VOLEXP = "Volatility_Expansion"
F_EVENT = "Event_Driven"
F_REGIME = "Regime_Switching"

FAMILIES: Tuple[str, ...] = (
    F_TREND, F_PULLBACK, F_BREAKOUT, F_MEANREV, F_RANGE, F_VOLEXP, F_EVENT, F_REGIME
)

SIDES: Tuple[str, ...] = ("LONG", "SHORT")


def fam_side(fam: str, side: str) -> str:
    return f"{fam}_{side}"


# Complexity preference (simplest -> most complex) for tie-break only
FAMILY_COMPLEXITY_ORDER: Dict[str, int] = {
    F_TREND: 1,
    F_MEANREV: 2,
    F_RANGE: 3,
    F_BREAKOUT: 4,
    F_PULLBACK: 5,
    F_VOLEXP: 6,
    F_EVENT: 7,
    F_REGIME: 8,
}


@dataclass(frozen=True)
class ClassificationVector:
    liquidità: str            # alta | media | bassa
    volatilità: str           # alta | media | bassa
    direzionalità: str        # trend | range | mista
    struttura: str            # impulsiva | oscillante
    news: str                 # forte | moderata | debole
    orizzonte: str            # intraday | swing | position
    timeframe: str            # 30m | 1h | 1d | 1w


@dataclass(frozen=True)
class SideResult:
    side: str
    top_strategies: List[str]
    scores: Dict[str, float]
    confidence_gap: float
    confidence_level: str
    flags: List[str]
    breakdown: Optional[Dict[str, Dict[str, float]]] = None


@dataclass(frozen=True)
class MappingResult:
    long: SideResult
    short: SideResult


WEIGHTS: Dict[str, float] = {
    "direzionalità": 3.0,
    "struttura": 2.0,
    "volatilità": 2.0,
    "orizzonte": 2.0,
    "timeframe": 1.5,
    "liquidità": 1.5,
    "news": 1.0,
}


SCORE_DIREZIONALITA: Dict[str, Dict[str, int]] = {
    "trend": {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: +1, F_MEANREV: -2, F_RANGE: -2, F_VOLEXP: +1, F_EVENT: 0,  F_REGIME: 0},
    "range": {F_TREND: -2, F_PULLBACK: -1, F_BREAKOUT: -1, F_MEANREV: +2, F_RANGE: +2, F_VOLEXP: -1, F_EVENT: 0,  F_REGIME: 0},
    "mista": {F_TREND: 0,  F_PULLBACK: 0,  F_BREAKOUT: 0,  F_MEANREV: 0,  F_RANGE: 0,  F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: +2},
}

SCORE_STRUTTURA: Dict[str, Dict[str, int]] = {
    "impulsiva":  {F_TREND: +1, F_PULLBACK: 0,  F_BREAKOUT: +2, F_MEANREV: -2, F_RANGE: -1, F_VOLEXP: +2, F_EVENT: +1, F_REGIME: 0},
    "oscillante": {F_TREND: -1, F_PULLBACK: +1, F_BREAKOUT: -2, F_MEANREV: +2, F_RANGE: +2, F_VOLEXP: -1, F_EVENT: -1, F_REGIME: 0},
}

SCORE_VOLATILITA: Dict[str, Dict[str, int]] = {
    "alta":  {F_TREND: +1, F_PULLBACK: 0,  F_BREAKOUT: +2, F_MEANREV: -1, F_RANGE: -1, F_VOLEXP: +2, F_EVENT: +2, F_REGIME: 0},
    "media": {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: +1, F_MEANREV: 0,  F_RANGE: +1, F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: 0},
    "bassa": {F_TREND: 0,  F_PULLBACK: +1, F_BREAKOUT: -2, F_MEANREV: +2, F_RANGE: +2, F_VOLEXP: -2, F_EVENT: -2, F_REGIME: 0},
}

SCORE_ORIZZONTE: Dict[str, Dict[str, int]] = {
    "intraday": {F_TREND: 0,  F_PULLBACK: 0,  F_BREAKOUT: +2, F_MEANREV: +2, F_RANGE: +2, F_VOLEXP: +1, F_EVENT: +1, F_REGIME: 0},
    "swing":    {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: +1, F_MEANREV: +1, F_RANGE: +1, F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: 0},
    "position": {F_TREND: +2, F_PULLBACK: +1, F_BREAKOUT: 0,  F_MEANREV: +1, F_RANGE: -1, F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: 0},
}

SCORE_LIQUIDITA: Dict[str, Dict[str, int]] = {
    "alta":  {F_TREND: +1, F_PULLBACK: +1, F_BREAKOUT: +1, F_MEANREV: +1, F_RANGE: +1, F_VOLEXP: +1, F_EVENT: +1, F_REGIME: +1},
    "media": {F_TREND: +1, F_PULLBACK: +1, F_BREAKOUT: 0,  F_MEANREV: +1, F_RANGE: +1, F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: +1},
    "bassa": {F_TREND: 0,  F_PULLBACK: 0,  F_BREAKOUT: -2, F_MEANREV: +1, F_RANGE: 0,  F_VOLEXP: -2, F_EVENT: -2, F_REGIME: 0},
}

SCORE_NEWS: Dict[str, Dict[str, int]] = {
    "forte":    {F_TREND: 0,  F_PULLBACK: 0,  F_BREAKOUT: +2, F_MEANREV: -2, F_RANGE: -2, F_VOLEXP: +2, F_EVENT: +2, F_REGIME: 0},
    "moderata": {F_TREND: +1, F_PULLBACK: +1, F_BREAKOUT: +1, F_MEANREV: 0,  F_RANGE: 0,  F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: 0},
    "debole":   {F_TREND: 0,  F_PULLBACK: 0,  F_BREAKOUT: -1, F_MEANREV: +2, F_RANGE: +2, F_VOLEXP: -1, F_EVENT: -2, F_REGIME: 0},
}

SCORE_TIMEFRAME: Dict[str, Dict[str, int]] = {
    "15m": {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: +2, F_MEANREV: +1, F_RANGE: +1, F_VOLEXP: +1, F_EVENT: +1, F_REGIME: 0},
    "30m": {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: +1, F_MEANREV: +1, F_RANGE: +1, F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: 0},
    "1h":  {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: +1, F_MEANREV: +1, F_RANGE: 0,  F_VOLEXP: 0,  F_EVENT: 0,  F_REGIME: 0},
    "1d":  {F_TREND: +2, F_PULLBACK: +2, F_BREAKOUT: 0,  F_MEANREV: +1, F_RANGE: -1, F_VOLEXP: -1, F_EVENT: 0,  F_REGIME: 0},
    "1w":  {F_TREND: +2, F_PULLBACK: +1, F_BREAKOUT: -1, F_MEANREV: +1, F_RANGE: -1, F_VOLEXP: -2, F_EVENT: 0,  F_REGIME: 0},
}

TABLES = {
    "direzionalità": SCORE_DIREZIONALITA,
    "struttura": SCORE_STRUTTURA,
    "volatilità": SCORE_VOLATILITA,
    "orizzonte": SCORE_ORIZZONTE,
    "liquidità": SCORE_LIQUIDITA,
    "news": SCORE_NEWS,
    "timeframe": SCORE_TIMEFRAME,
}

_ALLOWED = {
    "liquidità": {"alta", "media", "bassa"},
    "volatilità": {"alta", "media", "bassa"},
    "direzionalità": {"trend", "range", "mista"},
    "struttura": {"impulsiva", "oscillante"},
    "news": {"forte", "moderata", "debole"},
    "orizzonte": {"intraday", "swing", "position"},
    "timeframe": {"15m", "30m", "1h", "1d", "1w"},
}


def validate_vector(c: ClassificationVector) -> None:
    for field, allowed in _ALLOWED.items():
        val = getattr(c, field)
        if val not in allowed:
            raise ValueError(f"Invalid {field}={val!r}. Allowed: {sorted(allowed)}")


def _round(x: float) -> float:
    return round(float(x), 6)


def _uniq(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        if it not in seen:
            out.append(it)
            seen.add(it)
    return out


def _confidence(gap12: float, top1: float) -> Tuple[str, List[str]]:
    flags: List[str] = []
    if top1 < 3.0:
        flags.append("NO_TRADE_CANDIDATE")
    if gap12 < 1.0:
        flags.append("AMBIGUOUS")

    if gap12 >= 2.0 and top1 >= 6.0:
        return "HIGH", flags
    if (1.0 <= gap12 < 2.0) or (4.0 <= top1 < 6.0):
        return "MEDIUM", flags
    return "LOW", flags


def _add_fam(scores: Dict[str, float], breakdown: Dict[str, Dict[str, float]], fam: str, delta: float, label: str) -> None:
    scores[fam] += float(delta)
    breakdown[fam][label] = breakdown[fam].get(label, 0.0) + float(delta)


def _apply_synergies(c: ClassificationVector, fam_scores: Dict[str, float], fam_breakdown: Dict[str, Dict[str, float]]) -> None:
    if c.direzionalità == "trend" and c.struttura == "impulsiva":
        _add_fam(fam_scores, fam_breakdown, F_BREAKOUT, +2.0, "SYN_trend_impulsiva")
        _add_fam(fam_scores, fam_breakdown, F_VOLEXP, +1.0, "SYN_trend_impulsiva")

    if c.direzionalità == "range" and c.struttura == "oscillante":
        _add_fam(fam_scores, fam_breakdown, F_MEANREV, +2.0, "SYN_range_oscillante")
        _add_fam(fam_scores, fam_breakdown, F_RANGE, +2.0, "SYN_range_oscillante")

    if c.volatilità == "alta" and c.news == "forte":
        _add_fam(fam_scores, fam_breakdown, F_EVENT, +2.0, "SYN_highvol_newsforte")
        _add_fam(fam_scores, fam_breakdown, F_VOLEXP, +2.0, "SYN_highvol_newsforte")


def _apply_common_gates(c: ClassificationVector, fam_scores: Dict[str, float], fam_breakdown: Dict[str, Dict[str, float]]) -> List[str]:
    flags: List[str] = []

    if c.liquidità == "bassa" and c.timeframe in {"30m", "1h"}:
        _add_fam(fam_scores, fam_breakdown, F_BREAKOUT, -3.0, "G1_lowliq_short_tf")
        _add_fam(fam_scores, fam_breakdown, F_VOLEXP, -3.0, "G1_lowliq_short_tf")
        _add_fam(fam_scores, fam_breakdown, F_EVENT, -2.0, "G1_lowliq_short_tf")

    if c.timeframe == "1w":
        _add_fam(fam_scores, fam_breakdown, F_VOLEXP, -3.0, "G2_weekly_fast_convexity")
        _add_fam(fam_scores, fam_breakdown, F_BREAKOUT, -2.0, "G2_weekly_fast_convexity")

    if c.orizzonte == "intraday" and c.timeframe in {"1d", "1w"}:
        for fam in FAMILIES:
            _add_fam(fam_scores, fam_breakdown, fam, -4.0, "G3_inconsistent_horizon")
        flags.append("INCONSISTENT_HORIZON")

    if c.news == "forte" and c.liquidità != "alta":
        _add_fam(fam_scores, fam_breakdown, F_EVENT, -3.0, "G4_newsforte_needs_liq_alta")

    return flags


def _add_side(scores: Dict[str, float], breakdown: Dict[str, Dict[str, float]], key: str, delta: float, label: str) -> None:
    scores[key] += float(delta)
    breakdown[key][label] = breakdown[key].get(label, 0.0) + float(delta)


def _apply_side_adjustments(
    c: ClassificationVector,
    side: str,
    scores_side: Dict[str, float],
    breakdown_side: Dict[str, Dict[str, float]],
) -> None:
    if side == "SHORT" and c.news == "forte":
        _add_side(scores_side, breakdown_side, fam_side(F_EVENT, side), -1.0, "A1_short_news_squeeze_risk")
        _add_side(scores_side, breakdown_side, fam_side(F_VOLEXP, side), -0.5, "A1_short_news_squeeze_risk")

    if side == "LONG" and c.news == "forte":
        _add_side(scores_side, breakdown_side, fam_side(F_MEANREV, side), -1.0, "A2_long_gapdown_risk")
        _add_side(scores_side, breakdown_side, fam_side(F_RANGE, side), -1.0, "A2_long_gapdown_risk")


def _tie_key_family(fam: str, fam_breakdown: Dict[str, float]) -> Tuple[float, float]:
    dir_sum = sum(v for k, v in fam_breakdown.items() if k.startswith("direzionalità:"))
    tf_sum = sum(v for k, v in fam_breakdown.items() if k.startswith("timeframe:"))
    return (float(_round(dir_sum)), float(_round(tf_sum)))


def _rank_side(
    side: str,
    c: ClassificationVector,
    fam_scores: Dict[str, float],
    fam_breakdown: Dict[str, Dict[str, float]],
    flags_common: List[str],
    include_breakdown: bool,
) -> SideResult:
    scores_side: Dict[str, float] = {fam_side(fam, side): float(fam_scores[fam]) for fam in FAMILIES}
    breakdown_side: Dict[str, Dict[str, float]] = {fam_side(fam, side): dict(fam_breakdown[fam]) for fam in FAMILIES}

    _apply_side_adjustments(c, side, scores_side, breakdown_side)

    def sort_key(k: str):
        fam = k.rsplit("_", 1)[0]
        tb = _tie_key_family(fam, fam_breakdown[fam])
        return (_round(scores_side[k]), tb[0], tb[1], -FAMILY_COMPLEXITY_ORDER[fam])

    ranked = sorted(scores_side.keys(), key=sort_key, reverse=True)
    top3 = ranked[:3]

    gap12 = float(_round(scores_side[top3[0]] - scores_side[top3[1]]))
    conf_level, conf_flags = _confidence(gap12, scores_side[top3[0]])

    flags = _uniq(list(flags_common) + conf_flags)
    if "INCONSISTENT_HORIZON" in flags:
        conf_level = "LOW"

    return SideResult(
        side=side,
        top_strategies=top3,
        scores={k: float(_round(v)) for k, v in scores_side.items()},
        confidence_gap=float(_round(gap12)),
        confidence_level=conf_level,
        flags=flags,
        breakdown=breakdown_side if include_breakdown else None,
    )


def _norm(x) -> str:
    return str(x).strip().strip('"').strip("'").strip().lower()


def _safe_lookup(dim: str, label: str):
    """
    Safe lookup for TABLES[dim][label].
    Returns dict per_fam or None if missing.
    Normalizes both dim and label and supports optional aliases.
    """
    dim_n = _norm(dim)
    lab_n = _norm(label)

    # Optional aliases (add only if needed)
    # Example: if your TABLES uses 'intraday' but classifier outputs 'swing'
    LABEL_ALIASES = {
        # "orizzonte": {"swing": "intraday"},
        # "news": {"forte": "strong", "debole": "weak"},
        # "direzionalità": {"mista": "mixed"},
    }
    lab_key = LABEL_ALIASES.get(dim_n, {}).get(lab_n, lab_n)

    dim_tbl = TABLES.get(dim_n)
    if not dim_tbl:
        return None
    return dim_tbl.get(lab_key)


def rank_strategies(c: ClassificationVector, *, include_breakdown: bool = True) -> MappingResult:
    validate_vector(c)

    fam_scores: Dict[str, float] = {fam: 0.0 for fam in FAMILIES}
    fam_breakdown: Dict[str, Dict[str, float]] = {fam: {} for fam in FAMILIES}

    missing_flags = []

    for dim, label in (
        ("direzionalità", c.direzionalità),
        ("struttura", c.struttura),
        ("volatilità", c.volatilità),
        ("orizzonte", c.orizzonte),
        ("timeframe", c.timeframe),
        ("liquidità", c.liquidità),
        ("news", c.news),
    ):
        dim_n = _norm(dim)
        lab_n = _norm(label)

        # Weight must exist; if not, skip with flag
        if dim_n not in WEIGHTS:
            missing_flags.append(f"missing_weight:{dim_n}")
            continue

        w = WEIGHTS[dim_n]

        per_fam = _safe_lookup(dim_n, lab_n)
        if per_fam is None:
            missing_flags.append(f"missing_table:{dim_n}:{lab_n}")
            continue

        for fam in FAMILIES:
            if fam not in per_fam:
                missing_flags.append(f"missing_family_score:{dim_n}:{lab_n}:{fam}")
                continue
            contrib = float(w * per_fam[fam])
            fam_scores[fam] += contrib
            k = f"{dim_n}:{lab_n}"
            fam_breakdown[fam][k] = fam_breakdown[fam].get(k, 0.0) + contrib

    _apply_synergies(c, fam_scores, fam_breakdown)
    flags_common = _apply_common_gates(c, fam_scores, fam_breakdown)

    # add missing flags into common flags (dedup)
    if missing_flags:
        # flags_common may be list/tuple; normalize to list
        try:
            flags_common = list(flags_common)
        except Exception:
            flags_common = []
        for f in missing_flags:
            if f not in flags_common:
                flags_common.append(f)

    long_res = _rank_side("LONG", c, fam_scores, fam_breakdown, flags_common, include_breakdown)
    short_res = _rank_side("SHORT", c, fam_scores, fam_breakdown, flags_common, include_breakdown)

    return MappingResult(long=long_res, short=short_res)

