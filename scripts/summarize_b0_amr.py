"""壁 AMR スイープ結果を集計し、HOME/Legacy 差と Hybrid(B0) の必要幅をまとめる"""

import sys
import csv
import re
from pathlib import Path


def read_csv(path: Path):
    """入力を読み取る"""
    lines = [ln.strip() for ln in path.open("r", encoding="utf-8", errors="ignore") if ln.strip()]
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("d0,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError(f"CSV header not found in {path}")
    reader = csv.DictReader(lines[header_idx:])
    return list(reader)


def to_float(x: str) -> float:
    """数値化できない値を NaN にする簡易パーサ"""
    try:
        return float(x)
    except Exception:
        return float("nan")


def parse_amr_from_name(name: str):
    """引数や入力を解析する"""
    # expected something like: b0_shape_amr2_thr0p5.csv
    m = re.search(r"amr(\d+)", name)
    amr = int(m.group(1)) if m else -1

    t = re.search(r"thr(\d+)p(\d+)", name)
    thr = None
    if t:
        thr = float(f"{t.group(1)}.{t.group(2)}")
    return amr, thr


def main():
    """スクリプトのエントリポイントを実行する"""
    if len(sys.argv) < 3:
        print("Usage: python scripts/summarize_b0_amr.py <results_dir> <shape_tag> [--target-frac 0.2] [--pick-d0 2]", file=sys.stderr)
        return 1

    results_dir = Path(sys.argv[1])
    shape_tag = sys.argv[2]

    target_frac = 0.2
    pick_d0 = 2

    args = sys.argv[3:]
    i = 0
    while i < len(args):
        if args[i] == "--target-frac" and i + 1 < len(args):
            target_frac = float(args[i + 1]); i += 2
        elif args[i] == "--pick-d0" and i + 1 < len(args):
            pick_d0 = int(args[i + 1]); i += 2
        else:
            i += 1

    files = sorted(results_dir.glob("*.csv"))
    files = [p for p in files if shape_tag in p.name]

    if not files:
        raise RuntimeError(f"No CSV files matching '{shape_tag}' in {results_dir}")

    # output header
    print(",".join([
        "file","amr_factor","amr_threshold",
        "home_diff",
        f"hybrid_diff_d{pick_d0}",
        f"best_d0_for_{target_frac}x",
        "legacy_ratio_at_best",
        "near_wall_cells"
    ]))

    for p in files:
        rows = read_csv(p)
        if not rows:
            continue

        # mean_absdiff_home is the same for every d0 row (HOME baseline). Use first row.
        home_diff = to_float(rows[0].get("mean_absdiff_home", "nan"))

        # pick d0 row
        hyb_pick = float("nan")
        for r in rows:
            if int(r["d0"]) == pick_d0:
                hyb_pick = to_float(r.get("mean_absdiff_hybrid", "nan"))
                break

        # find best d0 achieving target fraction
        best_d0 = None
        best_ratio = float("nan")
        for r in rows:
            d0 = int(r["d0"])
            hyb = to_float(r.get("mean_absdiff_hybrid", "nan"))
            if home_diff > 0 and hyb <= target_frac * home_diff:
                best_d0 = d0
                best_ratio = to_float(r.get("legacy_ratio", "nan"))
                break

        amr_from_name, thr_from_name = parse_amr_from_name(p.name)

        # Prefer explicit columns (added in v3), but fall back to file name.
        amr_col = int(rows[0].get("amr_factor", amr_from_name))
        thr_col = rows[0].get("amr_threshold", None)
        thr_col = float(thr_col) if thr_col is not None else (thr_from_name if thr_from_name is not None else float("nan"))

        near_wall = rows[0].get("near_wall_cells", "")
        print(",".join([
            p.name,
            str(amr_col),
            f"{thr_col:.3f}",
            f"{home_diff:.9e}",
            f"{hyb_pick:.9e}",
            ("" if best_d0 is None else str(best_d0)),
            ("" if best_d0 is None else f"{best_ratio:.6f}"),
            str(near_wall),
        ]))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
