"""Ghi SHA-256 của các file dữ liệu PTB-XL dùng để huấn luyện + ngày tải (mtime) → results/data_checksums.json (FAIR: phiên bản + checksum)."""
import argparse, datetime, hashlib, json, os

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

if __name__ == "__main__":
    a = argparse.ArgumentParser(); a.add_argument("--data", default="data/ptb-xl"); a.add_argument("--out", default="results"); a = a.parse_args()
    out = {"dataset": "PTB-XL v1.0.3 (PhysioNet)", "files": {}}
    for f in ("ptbxl_database.csv", "scp_statements.csv", "RECORDS", "SHA256SUMS.txt"):
        p = f"{a.data}/{f}"
        if os.path.exists(p):
            out["files"][f] = {"sha256": sha(p), "mtime": datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d")}
    os.makedirs(a.out, exist_ok=True); json.dump(out, open(f"{a.out}/data_checksums.json", "w"), indent=2); print(json.dumps(out, indent=2))
