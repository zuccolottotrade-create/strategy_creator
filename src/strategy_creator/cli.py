from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd

from .criteria import DEFAULT_CFG, run_all_criteria

# ============================================================
# DEFAULTS / UI
# ============================================================
DEFAULT_INPUT_DIR = Path("/Users/claudio 1/Py_SUITE_TRADING/_data/Test Data").expanduser().resolve()
INCLUDE_PREFIX = "CLEAN_"


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Classificazione Operativa - CLI")
    p.add_argument(
        "-i",
        "--input",
        default=None,
        help="CSV input (se omesso: scelta interattiva)",
    )
    return p


def _print_banner() -> None:
    print("MODULO CLASSIFICAZIONE OPERATIVA")


def _list_candidate_files(folder: Path) -> List[Path]:
    """Lista i file candidati nella cartella: solo CLEAN_*.csv"""
    if not folder.exists() or not folder.is_dir():
        return []

    files: List[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue

        name = p.name
        if not name.startswith(INCLUDE_PREFIX):
            continue
        if p.suffix.lower() != ".csv":
            continue

        files.append(p)

    return files


def _ask_file_menu(folder: Path) -> Path:
    files = _list_candidate_files(folder)
    if not files:
        raise FileNotFoundError(
            f"Nessun file valido trovato in: {folder}\n"
            f"(mostro solo file {INCLUDE_PREFIX}*.csv)"
        )

    print("")
    print(f"Directory default: {folder}")
    print("File disponibili:")
    for i, p in enumerate(files, start=1):
        print(f"  {i}) {p.name}")

    print("")
    while True:
        try:
            s = input("VUOI LA CLASSIFICAZIONE DI QUALE FILE? (numero): ").strip()
        except KeyboardInterrupt:
            print("\nUscita richiesta (Ctrl+C).")
            raise

        if not s:
            print("Inserisci un numero.")
            continue
        try:
            n = int(s)
        except ValueError:
            print("Valore non valido. Inserisci un numero.")
            continue
        if 1 <= n <= len(files):
            return files[n - 1].resolve()
        print("Numero fuori range.")


def _print_report_table(rows: List[dict]) -> None:
    """
    Stampa a video un report tabellare a larghezze fisse, leggibile in terminale.
    rows: lista di dict con chiavi: criterion, label, reason
    """
    W_CRIT = 28
    W_LABEL = 10
    W_REASON = 80

    def _fit(s: str, w: int) -> str:
        s = "" if s is None else str(s)
        s = " ".join(s.split())
        if len(s) <= w:
            return s.ljust(w)
        return (s[: max(0, w - 1)] + "…") if w >= 2 else s[:w]

    header = f"{_fit('CRITERION', W_CRIT)} | {_fit('LABEL', W_LABEL)} | {_fit('REASON', W_REASON)}"
    sep = "-" * len(header)

    print("")
    print(sep)
    print(header)
    print(sep)

    for r in rows:
        line = (
            f"{_fit(r.get('criterion', ''), W_CRIT)} | "
            f"{_fit(r.get('label', ''), W_LABEL)} | "
            f"{_fit(r.get('reason', ''), W_REASON)}"
        )
        print(line)

    print(sep)
    print("")


# ============================================================
# TIMEFRAME inference
# ============================================================
def infer_timeframe_from_filename(path: Path) -> Optional[str]:
    """
    Estrae timeframe dal nome file.
    Esempi: FDAX15M -> 15m, ES1H -> 1h, ..._1D -> 1d
    """
    name = path.stem.upper()

    # Cerca pattern "15M", "1H", "1D", "2W" anche attaccati a simboli (FDAX15M)
    m = re.search(r"(\d+)\s*([MHDW])\b", name)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        unit_map = {"M": "m", "H": "h", "D": "d", "W": "w"}
        return f"{n}{unit_map[unit]}"

    # Pattern testuali (fallback)
    if "DAILY" in name:
        return "1d"
    if "WEEKLY" in name:
        return "1w"
    if "INTRADAY" in name:
        return "intraday"

    return None


def _normalize_seconds_to_timeframe(sec: int) -> Optional[str]:
    if sec <= 0:
        return None
    # preferenze "pulite"
    if sec % 86400 == 0:
        return f"{sec // 86400}d"
    if sec % 3600 == 0:
        return f"{sec // 3600}h"
    if sec % 60 == 0:
        return f"{sec // 60}m"
    # se non è multiplo pulito, meglio esplicitare in secondi
    return f"{sec}s"


def infer_timeframe_from_dataframe(df: pd.DataFrame, max_samples: int = 5000) -> Optional[str]:
    """
    Deduce timeframe dalla differenza temporale fra righe.
    - usa df['datetime'] se presente
    - altrimenti prova a costruirlo da date+time (se esistono)
    - calcola la moda delle differenze "piccole" per evitare gap (overnight/weekend)
    """
    if df is None or df.empty:
        return None

    # prendi una fetta (performance)
    dfx = df.iloc[:max_samples].copy()

    # recupera/crea datetime
    if "datetime" in dfx.columns:
        dt = pd.to_datetime(dfx["datetime"], errors="coerce")
    elif "date" in dfx.columns and "time" in dfx.columns:
        dt = pd.to_datetime(
            dfx["date"].astype(str) + " " + dfx["time"].astype(str),
            errors="coerce",
        )
    else:
        return None

    dt = dt.dropna()
    if len(dt) < 3:
        return None

    # sort e differenze
    dt = dt.sort_values()
    diffs = dt.diff().dropna().dt.total_seconds().astype(int)

    # filtra: solo positive e "ragionevoli" (escludo gap enormi)
    diffs = diffs[(diffs > 0) & (diffs <= 6 * 3600)]
    if diffs.empty:
        return None

    # moda
    counts = Counter(diffs.tolist())
    mode_sec, _ = counts.most_common(1)[0]

    return _normalize_seconds_to_timeframe(int(mode_sec))


def infer_timeframe(input_path: Path, df: pd.DataFrame) -> str:
    """
    Strategia:
    1) filename (più affidabile per pipeline)
    2) dataframe diffs (fallback)
    3) unknown
    """
    tf = infer_timeframe_from_filename(input_path)
    if tf:
        return tf

    tf = infer_timeframe_from_dataframe(df)
    if tf:
        return tf

    return "unknown"


def main(argv: Optional[List[str]] = None) -> int:
    """
    Entry point CLI.
    - Avvio in modalità interattiva se --input non è fornito.
    - Mostra solo file CLEAN_*.csv
    - Esegue i criteri modulari e scrive output CSV nella stessa directory
    - Uscita pulita con Ctrl+C
    """
    try:
        _print_banner()

        parser = build_arg_parser()
        args = parser.parse_args(argv)

        # 1) Determina input_path
        if args.input:
            input_path = Path(args.input).expanduser().resolve()
            if not input_path.exists():
                print(f"[ERRORE] File input non trovato: {input_path}")
                return 2
        else:
            try:
                input_path = _ask_file_menu(DEFAULT_INPUT_DIR)
            except FileNotFoundError as e:
                print(f"[ERRORE] {e}")
                return 2

        # 2) Output nello stesso folder dell’input
        out_dir = input_path.parent
        output_path = out_dir / f"CLASSIFICAZIONE_OPERATIVA_{input_path.stem}.csv"

        print(f"[OK] Input:  {input_path}")
        print(f"[OK] Output: {output_path.resolve()}")

        # 3) Carica CSV (sep=';') e normalizza colonne
        try:
            df = pd.read_csv(input_path, sep=";")
        except Exception as e:
            print(f"[ERRORE] Lettura CSV fallita: {e}")
            return 2

        df.columns = [str(c).strip().lower() for c in df.columns]

        # opzionale: costruisci datetime se serve (non forziamo: i criteri validano)
        if "datetime" not in df.columns and "date" in df.columns and "time" in df.columns:
            try:
                df["datetime"] = pd.to_datetime(
                    df["date"].astype(str) + " " + df["time"].astype(str),
                    errors="coerce",
                )
            except Exception:
                pass

        # >>> TIMEFRAME: calcolo una volta e lo porto in output
        timeframe = infer_timeframe(input_path, df)
        print(f"[OK] Timeframe: {timeframe}")

        # 4) Config soglie (per ora DEFAULT_CFG)
        cfg = DEFAULT_CFG

        # 5) Esegui criteri
        try:
            results = run_all_criteria(df, cfg)
        except Exception as e:
            print(f"[ERRORE] Esecuzione criteri fallita: {e}")
            return 2

        # 6) Scrivi output CSV
        try:
            CFG_KEY_BY_CRITERION = {
                "Liquidità": "liquidity",
                "Volatilità": "volatility",
                "Direzionalità prevalente": "directionality",
                "Struttura movimenti": "structure",
                "Reazione a news/eventi": "news_reaction",
                "Orizzonte naturale": "horizon",
            }

            WIN = cfg.get("windows", {}) if isinstance(cfg, dict) else {}

            out_rows = []
            for r in results:
                metrics = dict(r.metrics) if isinstance(r.metrics, dict) else {}

                cfg_key = CFG_KEY_BY_CRITERION.get(r.name)
                th = cfg.get(cfg_key, {}) if (cfg_key and isinstance(cfg, dict)) else {}

                out_rows.append(
                    {
                        "timeframe": timeframe,  # <<< NUOVO CAMPO
                        "criterion": r.name,
                        "label": r.label,
                        "reason": r.reason,
                        "metrics_json": json.dumps(metrics, ensure_ascii=False),
                        "thresholds_json": json.dumps(th, ensure_ascii=False) if th else "",
                        "windows_json": json.dumps(WIN, ensure_ascii=False) if WIN else "",
                    }
                )

            _print_report_table(out_rows)

            out_df = pd.DataFrame(out_rows)

            out_df.to_csv(
                output_path,
                sep=";",
                index=False,
                encoding="utf-8-sig",
                quoting=csv.QUOTE_ALL,
            )

        except Exception as e:
            print(f"[ERRORE] Scrittura output fallita: {e}")
            return 2

        print("[OK] Report scritto.")
        return 0

    except KeyboardInterrupt:
        print("\nUscita richiesta (Ctrl+C).")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

