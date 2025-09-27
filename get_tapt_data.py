import argparse
from pathlib import Path

SEP = "|||"

def parse_label_text(line: str) -> str:
    # input: "label ||| some text..."
    if SEP in line:
        _, text = line.split(SEP, 1)
        return text.strip()
    return line.strip()

def read_labeled(path: str):
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(parse_label_text(line))
    return out

def main():
    ap = argparse.ArgumentParser(description="Concat CFIMDB + SST train texts into unlabeled TAPT train data")
    ap.add_argument("--cfimdb", default="data/cfimdb-train.txt")
    ap.add_argument("--sst", default="data/sst-train.txt")
    ap.add_argument("--output_file", default="data/tapt.txt")
    args = ap.parse_args()

    cf = read_labeled(args.cfimdb)
    sst = read_labeled(args.sst)
    texts = cf + sst

    with open(args.output_file, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t + "\n")
    print(f"Wrote {len(texts):,} lines -> {args.output_file}")


if __name__ == "__main__":
    main()
