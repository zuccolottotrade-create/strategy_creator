from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import csv
import re


# -----------------------------
# Data model (CSV rules)
# -----------------------------
@dataclass(frozen=True)
class RuleRow:
    id: str
    enabled: str            # "TRUE"/"FALSE"
    scope: str              # "REGIME" | "ENTRY" | "EXIT"
    side: str               # "LONG" | "SHORT" | "BOTH"
    group: str              # "G0", "G1", ...
    lhs_col: str
    operator: str           # "==", ">", "<", "cross_above", "cross_below", "between"
    rhs_type: str           # "VALUE" | "COLUMN" | "LIST"
    rhs_value: str          # used if rhs_type in {"VALUE","LIST"}
    rhs_col: str            # used if rhs_type == "COLUMN"
    shift: str              # integer as string
    negate: str             # "TRUE"/"FALSE"


RULES_HEADER = [
    "id", "enabled", "scope", "side", "group",
    "lhs_col", "operator", "rhs_type", "rhs_value", "rhs_col",
    "shift", "negate",
]


# -----------------------------
# Timeframe extraction (NO DEFAULT)
# -----------------------------
_TF_RX = re.compile(r"(?P<tf>\d{1,3}[mhdw])", re.IGNORECASE)

def extract_timeframe_from_name(name: str) -> Optional[str]:
    """
    Extract timeframe token like 15m, 1h, 1d from filename.
    Returns normalized lowercase, or None.
    """
    m = _TF_RX.search(name)
    if not m:
        return None
    return m.group("tf").lower()


# -----------------------------
# Read selected strategy from mapper output CSV
# (robust: supports either a dedicated column or key/value rows)
# -----------------------------
def read_selected_strategy(input_csv: Path, sep: str = ";") -> Tuple[str, Optional[str], Optional[str]]:
    """
    Returns: (selected_strategy, symbol, timeframe_optional)

    Looks for:
      - a column named 'selected_strategy' / 'strategia_scelta'
      - or a row like: key;value where key in {'selected_strategy','strategia_scelta','strategia'}
    """
    if not input_csv.exists():
        raise FileNotFoundError(str(input_csv))

    # 1) try dict reader (header-based)
    with input_csv.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)

        # If header present and includes selected_strategy
        reader = csv.reader(f, delimiter=sep)
        rows = list(reader)

    if not rows:
        raise ValueError(f"File vuoto: {input_csv}")

    header = [h.strip() for h in rows[0]]
    header_l = [h.lower() for h in header]

    def _get_col(name_variants: List[str]) -> Optional[int]:
        for v in name_variants:
            if v in header_l:
                return header_l.index(v)
        return None

    sel_idx = _get_col(["selected_strategy", "strategia_scelta", "strategia"])
    sym_idx = _get_col(["symbol", "ticker", "strumento"])
    tf_idx  = _get_col(["timeframe", "tf"])

    # Header-based extraction
    if sel_idx is not None and len(rows) >= 2:
        # pick first non-empty value in column
        for r in rows[1:]:
            if sel_idx < len(r):
                val = (r[sel_idx] or "").strip()
                if val:
                    symbol = (r[sym_idx].strip() if sym_idx is not None and sym_idx < len(r) else None)
                    tf = (r[tf_idx].strip() if tf_idx is not None and tf_idx < len(r) else None)
                    return val, symbol, (tf or None)

    # 2) key/value style rows (no strict header)
    for r in rows:
        if len(r) >= 2:
            k = (r[0] or "").strip().lower()
            v = (r[1] or "").strip()
            if k in {"selected_strategy", "strategia_scelta", "strategia"} and v:
                # optional extras in same file (best-effort)
                tf = None
                symbol = None
                # try find other keys
                for rr in rows:
                    if len(rr) >= 2:
                        kk = (rr[0] or "").strip().lower()
                        vv = (rr[1] or "").strip()
                        if kk in {"timeframe", "tf"} and vv:
                            tf = vv
                        if kk in {"symbol", "ticker", "strumento"} and vv:
                            symbol = vv
                return v, symbol, tf

    raise ValueError(
        "Non trovo la strategia scelta nel CSV. "
        "Attesi: colonna 'selected_strategy' oppure riga 'selected_strategy;...'."
    )


# -----------------------------
# Strategy templates
# Assunzione: Run_strategia sa valutare gruppi come:
# - AND tra regole nello stesso (scope, side, group)
# - OR tra gruppi diversi nello stesso (scope, side)
# -----------------------------
def build_rules_for_strategy(selected_strategy: str) -> List[RuleRow]:
    """
    Returns list of RuleRow for ENTRY/EXIT and REGIME.
    selected_strategy: e.g. 'Pullback_in_Trend_LONG'
    """
    s = selected_strategy.strip()
    if not s:
        raise ValueError("selected_strategy vuota.")

    # Parse family + side
    if s.endswith("_LONG"):
        side = "LONG"
        family = s[:-5]
    elif s.endswith("_SHORT"):
        side = "SHORT"
        family = s[:-6]
    else:
        raise ValueError(f"Strategia non riconosciuta (manca _LONG/_SHORT): {s}")

    # Dispatch
    fn = _FAMILY_BUILDERS.get(family)
    if not fn:
        raise ValueError(f"Famiglia non supportata: {family}. Attese: {sorted(_FAMILY_BUILDERS.keys())}")
    return fn(side)


def _regime_trend(side: str) -> List[RuleRow]:
    """
    Trend regime using supertrend_dir + close vs ema40.
    Requires KPI columns: supertrend_dir, ema40, close
    """
    if side == "LONG":
        return [
            RuleRow("R1", "TRUE", "REGIME", "BOTH", "G0", "supertrend_dir", "==", "VALUE", "1", "", "0", "FALSE"),
            RuleRow("R2", "TRUE", "REGIME", "BOTH", "G0", "close", ">", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
    else:
        return [
            RuleRow("R1", "TRUE", "REGIME", "BOTH", "G0", "supertrend_dir", "==", "VALUE", "-1", "", "0", "FALSE"),
            RuleRow("R2", "TRUE", "REGIME", "BOTH", "G0", "close", "<", "COLUMN", "", "ema40", "0", "FALSE"),
        ]


def _rules_trend_following(side: str) -> List[RuleRow]:
    rows = []
    rows += _regime_trend(side)

    # ENTRY: breakout of trend confirmation (cross vs EMA40)
    if side == "LONG":
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "close", "cross_above", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
        rows += [
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "supertrend_dir", "==", "VALUE", "-1", "", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "close", "cross_below", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
        rows += [
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "supertrend_dir", "==", "VALUE", "1", "", "0", "FALSE"),
        ]
    return rows


def _rules_pullback_in_trend(side: str) -> List[RuleRow]:
    rows = []
    rows += _regime_trend(side)

    # Pullback condition + trigger
    # Requires: rsi21 (optional but recommended)
    if side == "LONG":
        # ENTRY: (RSI pulls back) AND (recovery trigger)
        rows += [
            RuleRow("E0", "TRUE", "ENTRY", "LONG", "G1", "rsi21", "<", "VALUE", "45", "", "0", "FALSE"),
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "rsi21", "cross_above", "VALUE", "50", "", "0", "FALSE"),
        ]
        rows += [
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "supertrend_dir", "==", "VALUE", "-1", "", "0", "FALSE"),
            RuleRow("X2", "FALSE", "EXIT", "LONG", "G2", "close", "cross_below", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("E0", "TRUE", "ENTRY", "SHORT", "G1", "rsi21", ">", "VALUE", "55", "", "0", "FALSE"),
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "rsi21", "cross_below", "VALUE", "50", "", "0", "FALSE"),
        ]
        rows += [
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "supertrend_dir", "==", "VALUE", "1", "", "0", "FALSE"),
            RuleRow("X2", "FALSE", "EXIT", "SHORT", "G2", "close", "cross_above", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
    return rows


def _rules_breakout(side: str) -> List[RuleRow]:
    rows = []
    # REGIME: optional, keep lighter to allow breakouts in many contexts
    # Requires: donchian_high_20, donchian_low_20 (or rolling highs/lows) computed in KPI step.
    if side == "LONG":
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "close", "cross_above", "COLUMN", "", "donchian_high_20", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "close", "cross_below", "COLUMN", "", "donchian_mid_20", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "close", "cross_below", "COLUMN", "", "donchian_low_20", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "close", "cross_above", "COLUMN", "", "donchian_mid_20", "0", "FALSE"),
        ]
    return rows


def _rules_mean_reversion(side: str) -> List[RuleRow]:
    rows = []
    # Requires: rsi21, bb_lower, bb_upper, bb_mid (or similar)
    if side == "LONG":
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "close", "<", "COLUMN", "", "bb_lower", "0", "FALSE"),
            RuleRow("E2", "TRUE", "ENTRY", "LONG", "G1", "rsi21", "<", "VALUE", "30", "", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "close", "cross_above", "COLUMN", "", "bb_mid", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "close", ">", "COLUMN", "", "bb_upper", "0", "FALSE"),
            RuleRow("E2", "TRUE", "ENTRY", "SHORT", "G1", "rsi21", ">", "VALUE", "70", "", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "close", "cross_below", "COLUMN", "", "bb_mid", "0", "FALSE"),
        ]
    return rows


def _rules_range_trading(side: str) -> List[RuleRow]:
    rows = []
    # Range trading assumes oscillation: use bbands/RSI mean-reversion but stricter take-profit
    # Requires: bb_lower, bb_upper, bb_mid, rsi21
    if side == "LONG":
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "rsi21", "<", "VALUE", "35", "", "0", "FALSE"),
            RuleRow("E2", "TRUE", "ENTRY", "LONG", "G1", "close", "<", "COLUMN", "", "bb_lower", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "close", "cross_above", "COLUMN", "", "bb_mid", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "rsi21", ">", "VALUE", "65", "", "0", "FALSE"),
            RuleRow("E2", "TRUE", "ENTRY", "SHORT", "G1", "close", ">", "COLUMN", "", "bb_upper", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "close", "cross_below", "COLUMN", "", "bb_mid", "0", "FALSE"),
        ]
    return rows


def _rules_volatility_expansion(side: str) -> List[RuleRow]:
    rows = []
    # Requires: atr_pct and optionally donchian_high_20/low_20
    # Idea: enter when volatility expands AND breakout triggers
    if side == "LONG":
        rows += [
            RuleRow("R1", "TRUE", "REGIME", "BOTH", "G0", "atr_pct", ">", "VALUE", "0.002", "", "0", "FALSE"),
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "close", "cross_above", "COLUMN", "", "donchian_high_20", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "close", "cross_below", "COLUMN", "", "donchian_mid_20", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("R1", "TRUE", "REGIME", "BOTH", "G0", "atr_pct", ">", "VALUE", "0.002", "", "0", "FALSE"),
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "close", "cross_below", "COLUMN", "", "donchian_low_20", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "close", "cross_above", "COLUMN", "", "donchian_mid_20", "0", "FALSE"),
        ]
    return rows


def _rules_event_driven(side: str) -> List[RuleRow]:
    rows = []
    # Requires: ret_abs (abs return) or gap_pct or ret_std_50.
    # Minimal proxy: ret_abs > threshold then follow-through with direction trigger
    if side == "LONG":
        rows += [
            RuleRow("R1", "TRUE", "REGIME", "BOTH", "G0", "ret_abs", ">", "VALUE", "0.01", "", "0", "FALSE"),
            RuleRow("E1", "TRUE", "ENTRY", "LONG", "G1", "close", ">", "COLUMN", "", "ema40", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "LONG", "G2", "close", "cross_below", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("R1", "TRUE", "REGIME", "BOTH", "G0", "ret_abs", ">", "VALUE", "0.01", "", "0", "FALSE"),
            RuleRow("E1", "TRUE", "ENTRY", "SHORT", "G1", "close", "<", "COLUMN", "", "ema40", "0", "FALSE"),
            RuleRow("X1", "TRUE", "EXIT", "SHORT", "G2", "close", "cross_above", "COLUMN", "", "ema40", "0", "FALSE"),
        ]
    return rows


def _rules_regime_switching(side: str) -> List[RuleRow]:
    rows = []
    # Requires: regime_label (e.g. "trend"/"range") OR a numeric regime flag.
    # We keep it simple: trade only in trend regime + trend-follow entry.
    if side == "LONG":
        rows += [
            RuleRow("R0", "TRUE", "REGIME", "BOTH", "G0", "regime_label", "==", "LIST", "(trend)", "", "0", "FALSE"),
        ]
    else:
        rows += [
            RuleRow("R0", "TRUE", "REGIME", "BOTH", "G0", "regime_label", "==", "LIST", "(trend)", "", "0", "FALSE"),
        ]
    # then reuse trend following triggers
    rows += _rules_trend_following(side)
    return rows


_FAMILY_BUILDERS = {
    "Trend_Following": _rules_trend_following,
    "Pullback_in_Trend": _rules_pullback_in_trend,
    "Breakout": _rules_breakout,
    "Mean_Reversion": _rules_mean_reversion,
    "Range_Trading": _rules_range_trading,
    "Volatility_Expansion": _rules_volatility_expansion,
    "Event_Driven": _rules_event_driven,
    "Regime_Switching": _rules_regime_switching,
}


# -----------------------------
# Write CSV rules
# -----------------------------
def write_rules_csv(rows: List[RuleRow], out_csv: Path, sep: str = ";") -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RULES_HEADER, delimiter=sep)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def build_rules_from_mapper_output(input_csv: Path, output_dir: Path, sep: str = ";") -> Path:
    """
    Main entry: reads selected_strategy from input_csv, extracts timeframe (NO DEFAULT),
    builds rules and writes RULES_<STRATEGY>_<TF>.csv in output_dir.
    Returns written path.
    """
    selected_strategy, symbol, tf_from_file = read_selected_strategy(input_csv, sep=sep)

    tf = (tf_from_file or "").strip().lower() if tf_from_file else None
    if not tf:
        tf = extract_timeframe_from_name(input_csv.name)

    if not tf:
        raise ValueError(
            "Timeframe non trovato. Deve essere presente nel CSV (colonna timeframe/tf o riga key/value) "
            "oppure nel nome file (es: ...15m...). Nessun default applicato."
        )

    rows = build_rules_for_strategy(selected_strategy)

    # output filename
    safe_sym = (symbol or "SYMBOL").strip().replace(" ", "_")
    safe_strat = selected_strategy.strip().replace(" ", "_")
    out_name = f"RULES_{safe_strat}_{safe_sym}_{tf}.csv"
    out_path = output_dir / out_name

    write_rules_csv(rows, out_path, sep=sep)
    return out_path
