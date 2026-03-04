from __future__ import annotations

import csv
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from logger import logger


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "00-raw"
CLEAN_DIR = DATA_DIR / "01-clean"

CSV_FILES = [
    "addresses.csv",
    "charges.csv",
    "checkouts.csv",
    "customers.csv",
    "payment_intents.csv",
    "stores.csv",
]


def try_pandas_read(filepath: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(filepath)
    except Exception:  # noqa: BLE001
        return None


def fix_row_with_embedded_commas(row: list[str], expected_cols: int) -> list[str] | None:
    extra = len(row) - expected_cols

    if extra <= 0:
        return row

    for merge_start in range(max(0, expected_cols - extra - 3), expected_cols - 1):
        candidate = row[:merge_start] + [",".join(row[merge_start:])]
        if len(candidate) == expected_cols:
            return candidate

    candidate = row[:1] + [",".join(row[1:-1])] + row[-1:]
    if len(candidate) == expected_cols:
        return candidate

    return row[:expected_cols]


def parse_with_csv_module(
    filepath: Path,
) -> tuple[list[str], list[list[str]], list[tuple[int, list[str]]]]:
    with open(filepath, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        expected_cols = len(header)

        good_rows = []
        bad_rows = []

        for line_num, row in enumerate(reader, start=2):
            if len(row) == expected_cols:
                good_rows.append(row)
            else:
                bad_rows.append((line_num, row))

    return header, good_rows, bad_rows


def fix_bad_rows(header: list[str], bad_rows: list[tuple[int, list[str]]]) -> list[list[str]]:
    expected_cols = len(header)
    fixed = []

    for line_num, row in bad_rows:
        fixed_row = fix_row_with_embedded_commas(row, expected_cols)
        if fixed_row:
            fixed.append(fixed_row)

    return fixed


def clean_csv_file(filepath: Path) -> dict:
    filename = filepath.name
    logger.info("Processing: %s", filename)

    result = {
        "file": filename,
        "good_rows": 0,
        "lost_rows": 0,
        "total": 0,
        "fixed_rows": 0,
    }

    df = try_pandas_read(filepath)
    if df is not None:
        result["total"] = len(df)
        result["good_rows"] = len(df)
        logger.info("  OK: %s rows (no issues)", len(df))
        output_path = CLEAN_DIR / filename
        df.to_csv(output_path, index=False)
        logger.info("  Saved to %s", output_path)
        return result

    logger.warning("  Standard parsing failed, using csv module fix...")

    header, good_rows, bad_rows = parse_with_csv_module(filepath)
    num_cols = len(header)

    logger.info("  Header has %s columns", num_cols)
    logger.info("  Initial good: %s, bad: %s", len(good_rows), len(bad_rows))

    fixed_rows = fix_bad_rows(header, bad_rows)
    result["fixed_rows"] = len(fixed_rows)
    result["total"] = len(good_rows) + len(bad_rows)
    result["good_rows"] = len(good_rows) + len(fixed_rows)
    result["lost_rows"] = result["total"] - result["good_rows"]

    if fixed_rows:
        logger.info("  Fixed %s rows", len(fixed_rows))

    all_rows = good_rows + fixed_rows
    df = pd.DataFrame(all_rows, columns=header)

    output_path = CLEAN_DIR / filename
    df.to_csv(output_path, index=False)
    logger.info("  Saved %s rows to %s", len(df), output_path)

    return result


def process_files(filenames: list[str]) -> tuple[list[dict], int]:
    results: list[dict] = []
    total_lost = 0

    for fname in filenames:
        fpath = RAW_DIR / fname
        if not fpath.exists():
            logger.warning("File not found: %s", fpath)
            continue

        try:
            result = clean_csv_file(fpath)
            results.append(result)
            total_lost += result["lost_rows"]
        except Exception as e:
            logger.error("  ERROR: %s", e)

    return results, total_lost


def print_summary(results: list[dict], total_lost: int) -> None:
    total_rows = 0

    logger.info("=" * 50)
    logger.info("SUMMARY")
    logger.info("=" * 50)

    for r in results:
        status = "OK" if r["lost_rows"] == 0 else "LOST %s" % r["lost_rows"]
        fixed_info = " (fixed %s)" % r["fixed_rows"] if r["fixed_rows"] > 0 else ""
        logger.info("  %s: %s rows%s -> %s", r["file"], r["good_rows"], fixed_info, status)
        total_rows += r["good_rows"]

    logger.info("=" * 50)
    logger.info("Total: %s rows saved, %s rows lost", total_rows, total_lost)

    if total_lost > 0:
        logger.warning("WARNING: %s rows were lost during cleaning!", total_lost)
    else:
        logger.info("SUCCESS: All rows preserved!")


def main():
    results, total_lost = process_files(CSV_FILES)
    print_summary(results, total_lost)


if __name__ == "__main__":
    main()
