#!/usr/bin/env python3
import argparse
import os

def parse_args():
    p = argparse.ArgumentParser(description="Split CSV rows into SeizureData and NonSeizure text files.")
    p.add_argument("csv_path", help="Path to input CSV file")
    p.add_argument("--seizure-dir", default="SeizureData", help="Output directory for label=1 rows (default: SeizureData)")
    p.add_argument("--nonseizure-dir", default="NonSeizure", help="Output directory for label!=1 rows (default: NonSeizure)")
    p.add_argument("--overwrite", action="store_true", help="If set, removes existing 1..N.txt numbering conflicts by continuing numbering after existing files")
    return p.parse_args()

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def next_index(dir_path):
    """
    Find the next numeric index for {n}.txt in dir_path.
    Scans existing files named like '123.txt' and returns max+1, else 1.
    """
    max_n = 0
    if not os.path.isdir(dir_path):
        return 1
    for name in os.listdir(dir_path):
        base, ext = os.path.splitext(name)
        if ext.lower() == ".txt" and base.isdigit():
            try:
                n = int(base)
                if n > max_n:
                    max_n = n
            except ValueError:
                pass
    return max_n + 1

def main():
    args = parse_args()

    ensure_dir(args.seizure_dir)
    ensure_dir(args.nonseizure_dir)

    # Start numbering after any existing files
    seizure_idx = next_index(args.seizure_dir)
    nonseizure_idx = next_index(args.nonseizure_dir)

    # Read input file line by line
    with open(args.csv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue  # skip empty lines

            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 3:
                # Need at least first, some data, and label
                continue

            # Determine label from the last column
            label_str = parts[-1]
            # Attempt to parse as integer if possible, else float
            try:
                label_val = int(label_str)
            except ValueError:
                try:
                    label_val = float(label_str)
                except ValueError:
                    # Skip lines with non-numeric labels
                    continue

            # Prepare the content by removing the first and last columns
            content_values = parts[1:-1]
            content_line = ",".join(content_values)

            if label_val == 1:
                out_dir = args.seizure_dir
                out_idx = seizure_idx
                seizure_idx += 1
            else:
                out_dir = args.nonseizure_dir
                out_idx = nonseizure_idx
                nonseizure_idx += 1

            out_path = os.path.join(out_dir, f"{out_idx}.txt")
            with open(out_path, "w", encoding="utf-8") as out_f:
                out_f.write(content_line + "\n")

if __name__ == "__main__":
    main()
