import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True)
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] missing {input_path}", file=sys.stderr)
        return 2

    print(f"[OK] bootstrap {input_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
