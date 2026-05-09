import tarfile
from pathlib import Path
from tqdm import tqdm

archive = "mvtec_ad_2.tar.gz"
out_dir = Path("input")
out_dir.mkdir(exist_ok=True)

print(f"Extracting {archive} to {out_dir}...")

with tarfile.open(archive, "r:gz") as tar:
    print("Extracting files...")
    for m in tar.getmembers():
        print(f"Extracting {m.name}...")
        m.uid = m.gid = 0
        m.uname = m.gname = ""
        m.mode = 0o755 if m.isdir() else 0o644
        tar.extract(m, path=out_dir)

print("done")