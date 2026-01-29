from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline_csv import run_map_strategies


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="strategy_mapper")
    sub = p.add_subparsers(dest="cmd", required=True)

    # -------------------------
    # map-strategies (existing)
    # -------------------------
    m = sub.add_parser("map-strategies", help="Append strategy mapping to a CLASSIFICAZIONE_OPERATIVA CSV")
    m.add_argument("--input", required=False, help="CLASSIFICAZIONE_OPERATIVA_*.csv (separator ';' or TSV)")
    m.add_argument("--output", required=False, help="STRATEGIA_CLASSIFICAZIONE_OPERATIVA_*.csv")

    m.add_argument("--timeframe", required=False, help="Override timeframe (es: 15m, 30m, 1h, 1d, 1w)")

    m.add_argument("--interactive", action="store_true", help="Ask user to choose one strategy from the shortlist")
    m.add_argument("--selection-json", required=False, help="Optional JSON file to write selection payload")

    # -------------------------
    # build-rules (new)
    # -------------------------
    b = sub.add_parser("build-rules", help="Build ENTRY/EXIT operational rules from mapper output CSV")
    b.add_argument("--input", required=False, help="STRATEGIA_CLASSIFICAZIONE_OPERATIVA_*.csv (separator ';')")
    b.add_argument(
        "--output-dir",
        required=False,
        help="Output directory for RULES_*.csv (alternative to --output)",
    )
    b.add_argument(
        "--output",
        required=False,
        help="Explicit output RULES_*.csv path (alternative to --output-dir)",
    )
    b.add_argument("--interactive", action="store_true", help="Ask user to choose input file if --input is missing")
    b.add_argument("--sep", default=";", help="CSV separator (default ';')")

    return p


def main(argv=None) -> int:
    p = build_parser()
    args = p.parse_args(argv)

    if args.cmd == "map-strategies":
        # ------------------------------------------------------------
        # IO selection
        # - --interactive = selezione strategia (shortlist) e, se mancano, scelta file input/output
        # ------------------------------------------------------------
        default_dir = Path("/Users/claudio 1/Py_SUITE_TRADING/_data/Test Data").expanduser().resolve()

        def _choose_input_file(folder: Path) -> Path:
            cands = sorted(folder.glob("CLASSIFICAZIONE_OPERATIVA_*.csv"))
            if not cands:
                raise SystemExit(f"Nessun file trovato in {folder} con pattern CLASSIFICAZIONE_OPERATIVA_*.csv")
            print("")
            print("Seleziona file di input (CLASSIFICAZIONE_OPERATIVA_*.csv):")
            for i, p in enumerate(cands, 1):
                print(f"  {i}) {p.name}")
            while True:
                s = input("Scelta (numero): ").strip()
                try:
                    k = int(s)
                    if 1 <= k <= len(cands):
                        return cands[k-1]
                except Exception:
                    pass
                print("Scelta non valida.")

        # input
        if args.input:
            inp = Path(args.input).expanduser().resolve()
        else:
            if not args.interactive:
                raise SystemExit("Manca --input. Usa --input/--output oppure lancia con --interactive per scegliere i file.")
            inp = _choose_input_file(default_dir)

        # output
        if args.output:
            out = Path(args.output).expanduser().resolve()
        else:
            if not args.interactive:
                raise SystemExit("Manca --output. Usa --input/--output oppure lancia con --interactive per scegliere i file.")
            # output di default: stesso folder dell'input, prefisso STRATEGIA_
            out = inp.parent / f"STRATEGIA_{inp.name}"

        sel_json = Path(args.selection_json).expanduser().resolve() if args.selection_json else None

        from .rules_builder import extract_timeframe_from_name

        tf = args.timeframe
        if not tf:
            tf = extract_timeframe_from_name(inp.name)  # estrae 15m/1h/1d...

        if not tf:
            raise SystemExit(
                "Timeframe non trovato. Specifica --timeframe oppure includilo nel nome file (es: ...15m...). "
                "Nessun default applicato."
            )

        run_map_strategies(
            inp,
            out,
            timeframe=tf,  # ✅ usa quello estratto o passato
            interactive=args.interactive,
            selection_json=sel_json,
        )

        print(f"[OK] map-strategies: {inp} -> {out} (timeframe={tf})")
        return 0

    if args.cmd == "build-rules":
        from .rules_builder import build_rules_from_mapper_output, read_selected_strategy, extract_timeframe_from_name, build_rules_for_strategy, write_rules_csv

        # ------------------------------------------------------------
        # IO selection (build-rules)
        # - Se mancano --input / --output-dir / --output, in modalità --interactive scegliamo a runtime
        # ------------------------------------------------------------
        default_dir = Path("/Users/claudio 1/Py_SUITE_TRADING/_data/Test Data").expanduser().resolve()

        def _choose_mapper_output(folder: Path) -> Path:
            # preferenza: STRATEGIA_CLASSIFICAZIONE_OPERATIVA_*.csv, fallback: STRATEGIA_*.csv
            cands = sorted(folder.glob("STRATEGIA_CLASSIFICAZIONE_OPERATIVA_*.csv"))
            if not cands:
                cands = sorted(folder.glob("STRATEGIA_*.csv"))
            if not cands:
                raise SystemExit(f"Nessun file trovato in {folder} con pattern STRATEGIA_*.csv")
            print("")
            print("Seleziona file di input (STRATEGIA_*.csv):")
            for i, p in enumerate(cands, 1):
                print(f"  {i}) {p.name}")
            while True:
                s = input("Scelta (numero): ").strip()
                try:
                    k = int(s)
                    if 1 <= k <= len(cands):
                        return cands[k-1]
                except Exception:
                    pass
                print("Scelta non valida.")

        # input
        if args.input:
            inp = Path(args.input).expanduser().resolve()
        else:
            if not getattr(args, "interactive", False):
                raise SystemExit("Manca --input. Usa --input/--output-dir oppure lancia con --interactive per scegliere il file.")
            inp = _choose_mapper_output(default_dir)

        # output defaults (se mancanti)
        # - output-dir: directory dove scrivere i RULES_*.csv
        # - output: path esplicito (alternativa a output-dir)
        if not getattr(args, "output_dir", None) and not getattr(args, "output", None):
            # default: stessa cartella dell'input
            args.output_dir = str(inp.parent)
        if getattr(args, "output", None) is None and getattr(args, "output_dir", None):
            # se serve un output file esplicito, ne proponiamo uno coerente
            # (il rules_builder può comunque usare output_dir)
            args.output = str(Path(args.output_dir) / f"RULES_{inp.name}")

        sep = args.sep

        # output resolution:
        # - if --output provided: use it
        # - else require --output-dir and auto-name output
        if args.output:
            out_path = Path(args.output).expanduser().resolve()
            out_dir = out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)

            # Build rules using the same logic but writing to explicit output filename
            selected_strategy, symbol, tf_from_file = read_selected_strategy(inp, sep=sep)
            tf = (tf_from_file.strip().lower() if tf_from_file else None) or extract_timeframe_from_name(inp.name)

            if not tf:
                raise SystemExit(
                    "Timeframe non trovato. Deve essere nel CSV (colonna timeframe/tf o riga key/value) "
                    "oppure nel nome file (es: ...15m...). Nessun default applicato."
                )

            rows = build_rules_for_strategy(selected_strategy)
            write_rules_csv(rows, out_path, sep=sep)

            print(f"[OK] build-rules: {inp} -> {out_path}")
            return 0

        if not args.output_dir:
            raise SystemExit("Errore: specifica --output-dir oppure --output")

        out_dir = Path(args.output_dir).expanduser().resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = build_rules_from_mapper_output(inp, out_dir, sep=sep)
        print(f"[OK] build-rules: {inp} -> {out_path}")
        return 0

    raise SystemExit(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    raise SystemExit(main())

