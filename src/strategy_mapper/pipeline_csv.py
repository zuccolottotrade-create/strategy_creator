from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd

from .mapper import ClassificationVector, rank_strategies

import unicodedata


CRIT_MAP: Dict[str, str] = {
    "liquidita": "liquidità",
    "volatilita": "volatilità",
    "direzionalita prevalente": "direzionalità",
    "direzionalita": "direzionalità",
    "struttura movimenti": "struttura",
    "struttura dei movimenti": "struttura",
    "reazione a news/eventi": "news",
    "reazione a news eventi": "news",
    "reazione a news / eventi": "news",
    "orizzonte naturale": "orizzonte",
    "orizzonte": "orizzonte",
}


DEFAULT_TIMEFRAME = "No timeframe detected"


def _strip_accents(s: str) -> str:
    # NFKD separa base+accenti; poi eliminiamo i combining marks
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


import unicodedata

_MOJIBAKE_REMAP = str.maketrans({
    "ˆ": "a",   # Liquiditˆ -> liquidita
    "´": "",    # varianti
    "`": "",
    "¨": "",
    "˜": "",
})

def print_all_possible_strategies() -> None:
    families = [
        "Trend_Following",
        "Pullback_in_Trend",
        "Breakout",
        "Mean_Reversion",
        "Range_Trading",
        "Volatility_Expansion",
        "Event_Driven",
        "Regime_Switching",
    ]
    print("\n=== STRATEGY_MAPPER ===")
    print("Strategie possibili (famiglia x side):")
    for fam in families:
        print(f" - {fam}_LONG")
    for fam in families:
        print(f" - {fam}_SHORT")

def _strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _norm(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\ufeff", "").replace("\u00a0", " ").replace("\t", " ")
    s = s.strip().strip('"').strip("'")
    s = s.lower()

    # 1) fix mojibake-like chars first
    s = s.translate(_MOJIBAKE_REMAP)

    # 2) remove accents (for true unicode accents)
    s = _strip_accents(s)

    s = " ".join(s.split())
    return s


def detect_sep(path: Path) -> str:
    """
    Preferenza ';' (standard), ma se il file è TSV (tab) lo riconosciamo.
    """
    head = path.read_text(encoding="utf-8", errors="ignore").splitlines()[:3]
    sample = "\n".join(head)
    if "\t" in sample and sample.count("\t") >= sample.count(";"):
        return "\t"
    if ";" in sample:
        return ";"
    if "," in sample:
        return ","
    return ";"



def load_csv(path: Path) -> pd.DataFrame:
    """
    Legge CSV/TSV con encoding variabile e separatore auto-detect.
    """
    sep = detect_sep(path)
    last_err = None
    for enc in ("utf-8", "utf-8-sig", "cp1252", "latin1"):
        try:
            df = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False, encoding=enc)
            # Se il separatore era sbagliato, spesso risulta 1 sola colonna con dentro i tab.
            if len(df.columns) == 1 and ("\t" in df.columns[0]) and sep != "\t":
                df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding=enc)

            missing = {"criterion", "label"} - set(df.columns)
            if missing:
                raise ValueError(f"CSV missing columns: {sorted(missing)}. Found: {list(df.columns)} (sep={sep!r}, enc={enc!r})")
            return df
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Impossibile leggere file {path} con sep auto-detect e encoding utf-8/utf-8-sig/cp1252/latin1. Ultimo errore: {last_err}"
    )



def infer_timeframe_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "30m" in name:
        return "30m"
    if "1h" in name or "60m" in name:
        return "1h"
    if "1w" in name:
        return "1w"
    if "1d" in name:
        return "1d"
    return DEFAULT_TIMEFRAME


def extract_vector(df: pd.DataFrame, timeframe: str) -> ClassificationVector:
    values: Dict[str, str] = {}

    for _, row in df.iterrows():
        crit = _norm(str(row.get("criterion", "")))
        lab = _norm(str(row.get("label", "")))
        if not crit or not lab:
            continue

        if crit in CRIT_MAP:
            values[CRIT_MAP[crit]] = lab
            continue

        for k, field in CRIT_MAP.items():
            if k in crit:
                values[field] = lab
                break

    needed = ["liquidità", "volatilità", "direzionalità", "struttura", "news", "orizzonte"]
    missing = [k for k in needed if k not in values]
    if missing:
        raise ValueError(f"Mancano criteri nel CSV: {missing}")

    return ClassificationVector(
        liquidità=values["liquidità"],
        volatilità=values["volatilità"],
        direzionalità=values["direzionalità"],
        struttura=values["struttura"],
        news=values["news"],
        orizzonte=values["orizzonte"],
        timeframe=timeframe,
    )



def append_strategy_section(df: pd.DataFrame, *, result, vector: ClassificationVector, selected_strategy: str) -> pd.DataFrame:
    base_cols = ["criterion", "label", "reason", "metrics_json", "thresholds_json", "windows_json"]
    for c in base_cols:
        if c not in df.columns:
            df[c] = ""

    rows = []

    def add(criterion: str, label: str = "", reason: str = ""):
        rows.append({
            "criterion": criterion,
            "label": label,
            "reason": reason,
            "metrics_json": "",
            "thresholds_json": "",
            "windows_json": "",
        })

    add("=== STRATEGY_MAPPER ===")
    add("timeframe", vector.timeframe)

    add("top3_long", ",".join(result.long.top_strategies))
    add("confidence_long", result.long.confidence_level, f"gap={result.long.confidence_gap}")
    add("flags_long", ",".join(result.long.flags))

    add("top3_short", ",".join(result.short.top_strategies))
    add("confidence_short", result.short.confidence_level, f"gap={result.short.confidence_gap}")
    add("flags_short", ",".join(result.short.flags))
    add("selected_strategy", selected_strategy)

    add("vector", str(asdict(vector)))

    return pd.concat([df[base_cols], pd.DataFrame(rows)[base_cols]], ignore_index=True)

def choose_strategy_interactive(result) -> str:
    """
    Menu interattivo: propone le strategie LONG e SHORT separatamente.
    Ritorna stringa tipo 'Breakout_LONG' o 'Mean_Reversion_SHORT'.
    """
    long_opts = list(result.long.top_strategies)
    short_opts = list(result.short.top_strategies)

    options = []
    idx_map = {}

    print("\n==============================")
    print(" TOP 3 STRATEGIE LONG")
    print("==============================")
    for i, opt in enumerate(long_opts, start=1):
        print(f" {i}) {opt}")
        idx_map[i] = opt
        options.append(opt)

    offset = len(long_opts)

    print("\n==============================")
    print(" TOP 3 STRATEGIE SHORT")
    print("==============================")
    for j, opt in enumerate(short_opts, start=1):
        idx = offset + j
        print(f" {idx}) {opt}")
        idx_map[idx] = opt
        options.append(opt)

    print("\n 0) Nessuna (NO_TRADE)")

    while True:
        s = input("\nSeleziona numero: ").strip()
        if s == "0":
            return "NO_TRADE"
        if s.isdigit():
            k = int(s)
            if k in idx_map:
                return idx_map[k]
        print("Scelta non valida. Riprova.")


def run_map_strategies(
    input_path: Path,
    output_path: Path,
    *,
    timeframe: Optional[str] = None,
    interactive: bool = False,
    selection_json: Optional[Path] = None,
) -> Path:
    """
    Reads a CLASSIFICAZIONE_OPERATIVA CSV, ranks strategies, optionally asks user selection,
    and writes STRATEGIA_CLASSIFICAZIONE_OPERATIVA CSV.

    Timeframe precedence (NO default):
      1) CLI --timeframe
      2) CSV column 'timeframe' (robust to quoted headers)
      3) filename inference
    """
    df = load_csv(input_path)

    # ---- timeframe resolution (NO hidden default) ----
    tf: Optional[str] = None

    # (1) CLI
    if timeframe:
        tf = str(timeframe).strip().lower()

    # (2) CSV column (robust normalization, supports quoted headers like '"timeframe"')
    if not tf:
        norm_cols = {}
        for c in df.columns:
            c_norm = str(c).strip().strip('"').strip("'").strip().lower()
            norm_cols[c_norm] = c

        if "timeframe" in norm_cols:
            real_col = norm_cols["timeframe"]
            try:
                tf_val = df[real_col].dropna().iloc[0]
                tf = str(tf_val).strip().strip('"').strip("'").strip().lower()
            except Exception:
                tf = None

    # (3) filename
    if not tf:
        tf = infer_timeframe_from_filename(input_path)
        if tf:
            tf = str(tf).strip().lower()

    if not tf:
        raise ValueError(
            "Timeframe non trovato: specifica --timeframe oppure includilo nel CSV (colonna 'timeframe') "
            "oppure nel nome file (es: ...15m...). Nessun default applicato."
        )

    # ---- compute vector + ranking ----
    vec = extract_vector(df, tf)
    try:
        vec.timeframe = tf
    except Exception:
        pass

    print_all_possible_strategies()
    print("[Selezione delle top 3 migliori strategie in corso]")
    res = rank_strategies(vec, include_breakdown=False)

    selected = "NO_TRADE"
    if interactive:
        selected = choose_strategy_interactive(res)

    out_df = append_strategy_section(df, result=res, vector=vec, selected_strategy=selected)

    # propagate timeframe to output for downstream (build-rules)
    out_df["timeframe"] = tf

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path, sep=";", index=False)

    # Optional companion JSON for the next module
    if selection_json is not None:
        import json
        payload = {
            "selected_strategy": selected,
            "timeframe": tf,
            "vector": asdict(vec),
            "top3_long": res.long.top_strategies,
            "top3_short": res.short.top_strategies,
            "confidence_long": res.long.confidence_level,
            "confidence_short": res.short.confidence_level,
            "flags_long": res.long.flags,
            "flags_short": res.short.flags,
        }
        selection_json.parent.mkdir(parents=True, exist_ok=True)
        selection_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return output_path



