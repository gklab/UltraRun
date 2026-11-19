"""Download the original checkpoint, verifying each shard against the pinned HF release."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
REPO = "Qwen/Qwen3.8-27B"
REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
MIRROR = f"https://modelscope.cn/models/{REPO}/resolve/master"
UPSTREAM = f"https://huggingface.co/{REPO}/resolve/{REVISION}"
METADATA = (
    "config.json", "tokenizer.json", "chat_template.jinja",
    "generation_config.json", "model.safetensors.index.json",
)


def good(path, metadata):
    if not path.is_file() or path.stat().st_size != metadata["size"]:
        return False
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest() == metadata["sha256"]


def download(url, path, metadata=None):
    if metadata and good(path, metadata):
        print(f"verified {path.name}", flush=True)
        return
    partial = path.with_name(path.name + ".partial")
    for _ in range(5):
        if metadata and partial.exists() and partial.stat().st_size >= metadata["size"]:
            if good(partial, metadata):
                partial.replace(path)
                print(f"verified {path.name}", flush=True)
                return
            partial.unlink()
        args = ["curl", "-fLsS", "--connect-timeout", "20", "--speed-limit", "16384",
                "--speed-time", "45", "--continue-at", "-", "-o", str(partial), url]
        status = subprocess.run(args).returncode
        if status == 0 and (metadata is None or good(partial, metadata)):
            partial.replace(path)
            print(f"verified {path.name}" if metadata else f"downloaded {path.name}", flush=True)
            return
        if status in (0, 33, 36):
            partial.unlink(missing_ok=True)
    raise RuntimeError(f"Could not retrieve and verify {path.name}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "weights" / "Qwen3.8-27B")
    parser.add_argument("--source", choices=("huggingface", "modelscope"), default="huggingface")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / "weights-manifest.json").read_text())
    remote = MIRROR if args.source == "modelscope" else UPSTREAM
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download, f"{UPSTREAM}/{name}", args.directory / name)
                   for name in METADATA if not (args.directory / name).is_file()]
        futures.extend(pool.submit(download, f"{remote}/{name}", args.directory / name, entry)
                       for name, entry in manifest.items())
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
