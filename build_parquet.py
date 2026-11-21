"""
Regenerate `data/wowah_data_all.parquet` from the raw heartbeat files.

Writes the parquet with `parse_files` itself, so the two cannot drift. Run it
whenever `src/parse_to_single_table.py` changes:

    pixi run python build_parquet.py

Needs the raw corpus in `data/raw/WoWAH/`.
"""

import sys
import time

import polars as pl

import main

OUTPUT = "data/wowah_data_all.parquet"


def build():
    files = main.list_raw_files(main.RAW_DATA_ROOT)
    print(f"parsing {len(files)} files...")
    t0 = time.time()
    df = main.parse_files(files)
    print(f"parsed {len(df):,} rows in {time.time() - t0:.1f} s")

    # `parse_files` already sorts by (timestamp, char), so the parquet keeps
    # the parser's row order.
    df.write_parquet(OUTPUT)
    print(f"wrote {OUTPUT}")
    return df


if __name__ == "__main__":
    # `parse_files` uses a ProcessPoolExecutor, which needs the guard on macOS.
    sys.exit(build())
