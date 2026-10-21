import hashlib
import pathlib
import zipfile
from pathlib import Path
from huggingface_hub import hf_hub_download

REPO = "HAL-9001/spider-databases"
SHA256 = "00636695dabed6b5f4b8328a16b13e069a2f16591d5efcce57660669c85b121b"
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = pathlib.Path(hf_hub_download(
        REPO, "spider_data.zip", repo_type="dataset", local_dir=DATA_DIR
    ))
    assert sha256(zip_path) == SHA256, "Checksum mismatch — download corrupted"
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    zip_path.unlink()  # free disk space
    print("Done:", DATA_DIR / "spider_data")


if __name__ == "__main__":
    main()