"""Run the single-request 4096/1024 harness across BF16 decode configurations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from ultrarun.profiles import OPTIMIZED_DRAFT, OPTIMIZED_MODEL
CANDIDATES = {
    "baseline": [],
    "merged": ["--merge"],
    "kv": ["--merge", "--reserve-kv"],
    "mtp2": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "2"],
    "fused2": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "2", "--fused-delta"],
    "fused4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4", "--fused-delta"],
    "wide4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4", "--fused-delta", "--wide-verify"],
    "packed4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                "--fused-delta", "--wide-verify", "--packed-ffn"],
    "mainpacked4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                    "--fused-delta", "--wide-verify", "--packed-main"],
    "shortlist4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                   "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768"],
    "packedhead4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                    "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768", "--packed-head"],
    "packedattn4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                    "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                    "--packed-head", "--packed-attention"],
    "packedattn5": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "5",
                    "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                    "--packed-head", "--packed-attention"],
    "lastdraft4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                   "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                   "--packed-head", "--packed-attention", "--fast-draft"],
    "fusedffn4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                  "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                  "--packed-head", "--packed-attention", "--fast-draft", "--fused-ffn"],
    "deltaio4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                 "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                 "--packed-head", "--packed-attention", "--fast-draft", "--fused-ffn", "--fused-delta-io"],
    "fusedffn5": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "5",
                  "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                  "--packed-head", "--packed-attention", "--fast-draft", "--fused-ffn"],
    "splitattn5": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "5",
                   "--fused-delta", "--wide-verify", "--packed-main", "--draft-vocab", "32768",
                   "--packed-head", "--packed-attention", "--fast-draft", "--fused-ffn", "--split-attention"],
    "mainpacked6": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "6",
                    "--fused-delta", "--wide-verify", "--packed-main"],
    "tree4x2": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                "--tree-depth", "2", "--fused-delta", "--wide-verify"],
    "tailtree4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
                  "--tree-tail-gap", "0.5", "--fused-delta", "--wide-verify"],
    "vector2": ["--merge", "--skinny", "--reserve-kv", "--vector-bf16", "--draft-depth", "2"],
    "mtp3": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "3", "--draft-vocab", "32768"],
    "vector3": ["--merge", "--skinny", "--reserve-kv", "--vector-bf16",
                "--draft-depth", "3", "--draft-vocab", "32768"],
    "mtp4": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4", "--draft-vocab", "16384"],
    "mtp6": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "6", "--draft-vocab", "16384"],
    "mtp8": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "8", "--draft-vocab", "16384"],
    "lookup16": ["--merge", "--skinny", "--reserve-kv", "--lookup-depth", "16"],
    "hybrid": ["--merge", "--skinny", "--reserve-kv", "--draft-depth", "4",
               "--draft-vocab", "16384", "--lookup-depth", "16"],
}
CANDIDATES["buffered5"] = CANDIDATES["splitattn5"]
CANDIDATES["adaptive5"] = [*CANDIDATES["splitattn5"], "--draft-confidence", "0.12"]
CANDIDATES["adaptive6"] = ["6" if value == "5" else value for value in CANDIDATES["adaptive5"]]
CANDIDATES["device5"] = [*CANDIDATES["adaptive5"], "--device-draft"]
CANDIDATES["residual5"] = [*CANDIDATES["device5"], "--fused-residual"]
CANDIDATES["compactdelta5"] = [*CANDIDATES["residual5"], "--compact-delta"]
CANDIDATES["wideblock10"] = [
    *("10" if value == "5" else value for value in CANDIDATES["compactdelta5"]),
    "--wide-blocks",
]
CANDIDATES["selective10"] = [*CANDIDATES["wideblock10"], "--draft-extension-confidence", "0.5"]
CANDIDATES["rootbranch10"] = [*CANDIDATES["selective10"], "--draft-root-branch-gap", "1.0"]
CANDIDATES["compacttree10"] = []
for key, value in {**OPTIMIZED_MODEL, **OPTIMIZED_DRAFT}.items():
    if key in {"lossless_vocabulary", "paired_down", "lossless_draft_rows", "shared_vocabulary",
               "compiled_delta", "layer_prefill", "fast_prefill", "async_verify", "direct_delta_sequence"}:
        continue  # Keep comparison-specific flags and server-only options out of the base CLI preset.
    CANDIDATES["compacttree10"].append("--" + key.replace("_", "-"))
    if value is not True:
        CANDIDATES["compacttree10"].append(str(value))
CANDIDATES["compacttree15"] = ["15" if value == "10" else value for value in CANDIDATES["compacttree10"]]
CANDIDATES["compacttree12"] = ["12" if value == "10" else value for value in CANDIDATES["compacttree10"]]
CANDIDATES["plainffndown10"] = [arg for arg in CANDIDATES["compacttree10"] if arg != "--lossless-down"]
CANDIDATES["plaingateup10"] = [arg for arg in CANDIDATES["compacttree10"] if arg != "--lossless-gate-up"]
CANDIDATES["plaininputs10"] = [arg for arg in CANDIDATES["compacttree10"] if arg != "--lossless-inputs"]
CANDIDATES["losslessinputs10"] = CANDIDATES["compacttree10"]
CANDIDATES["losslessoutputs10"] = [*CANDIDATES["compacttree10"], "--lossless-attention-outputs"]
CANDIDATES["losslessvocab10"] = [*CANDIDATES["compacttree10"], "--lossless-vocabulary"]
CANDIDATES["losslessdraft10"] = [*CANDIDATES["losslessvocab10"], "--lossless-draft-vocab"]
CANDIDATES["shareddraft10"] = [*CANDIDATES["losslessdraft10"], "--shared-vocabulary"]
CANDIDATES["compileddelta10"] = [*CANDIDATES["shareddraft10"], "--compiled-delta"]
CANDIDATES["confirmedvocab10"] = [*CANDIDATES["compileddelta10"], "--draft-confirmed-vocab"]
CANDIDATES["layerprefill10"] = [*CANDIDATES["compileddelta10"], "--layer-prefill"]
CANDIDATES["logical10"] = [*CANDIDATES["layerprefill10"], "--draft-schedule", "logical"]
CANDIDATES["frontier11"] = [*CANDIDATES["layerprefill10"], "--draft-schedule", "frontier"]
CANDIDATES["frontierprefetch11"] = [*CANDIDATES["layerprefill10"], "--draft-schedule", "frontier_prefetch"]
CANDIDATES["fastprefill11"] = [*CANDIDATES["frontierprefetch11"], "--fast-prefill"]
CANDIDATES["paireddown10"] = [*CANDIDATES["losslessvocab10"], "--paired-down"]
CANDIDATES["sharedpaireddown10"] = [*CANDIDATES["shareddraft10"], "--paired-down"]
CANDIDATES["dualhead10"] = [arg for arg in CANDIDATES["compacttree10"] if arg != "--packed-head-only"]
CANDIDATES["singlehead10"] = CANDIDATES["compacttree10"]
CANDIDATES["draftfused10"] = [*CANDIDATES["compacttree10"], "--fused-draft-ranking"]
CANDIDATES["panelcheck10"] = [*CANDIDATES["compacttree10"], "--draft-panel-checkpoint"]
CANDIDATES["longbranch10"] = [*CANDIDATES["compacttree10"], "--draft-branch-depth", "5"]
CANDIDATES["secondbranch10"] = [*CANDIDATES["compacttree10"], "--draft-second-branch"]
CANDIDATES["cachedembed10"] = [*CANDIDATES["compacttree10"], "--cache-draft-embedding"]
CANDIDATES["adaptivebranch10"] = [*CANDIDATES["compacttree10"], "--draft-branch-positions", "3"]
CANDIDATES["multifork10"] = [*CANDIDATES["adaptivebranch10"], "--draft-multi-fork"]
CANDIDATES["compact5"] = ["8192" if value == "32768" else value for value in CANDIDATES["splitattn5"]]
BENCH_ENV = {
    "buffered5": {"MLX_MAX_OPS_PER_BUFFER": "128", "MLX_MAX_MB_PER_BUFFER": "1024"},
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "weights" / "Qwen3.8-27B")
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "tuning.jsonl")
    parser.add_argument("--memory-gib", type=int)
    parser.add_argument("--cache-gib", type=int)
    parser.add_argument("--timeout", type=int, default=360, help="seconds per benchmark process")
    parser.add_argument("config", nargs="*", choices=tuple(CANDIDATES), metavar="CONFIG")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    selected = args.config or ["baseline", "mtp2"]
    input_args = ["--prompt-file", str(args.prompt_file)] if args.prompt_file else []
    memory_args = []
    for flag, value in (("--memory-gib", args.memory_gib), ("--cache-gib", args.cache_gib)):
        if value is not None:
            memory_args.extend((flag, str(value)))
    previous = [json.loads(line) for line in args.out.read_text().splitlines()] if args.out.is_file() else []
    baselines = {result["prompt_hash"]: result for result in previous if result["config"] == "baseline"}
    best = None
    for name in selected:
        print(f"Running {name}", flush=True)
        cmd = [sys.executable, "-m", "ultrarun.cli", "--model", str(args.model),
               *memory_args,
               *CANDIDATES[name], "bench", *input_args]
        try:
            env = {**os.environ, **BENCH_ENV.get(name, {})}
            run = subprocess.run(cmd, cwd=ROOT, text=True, stdout=subprocess.PIPE,
                                 check=True, timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            print(f"{name} exceeded {args.timeout}s and its process was stopped.", file=sys.stderr)
            continue
        except subprocess.CalledProcessError as error:
            raise SystemExit(f"{name} stopped with exit status {error.returncode}") from None
        result = json.loads(run.stdout.strip().splitlines()[-1])
        if name == "baseline":
            baselines[result["prompt_hash"]] = result
        baseline = baselines.get(result["prompt_hash"])
        result["config"] = name
        result["same_tokens_as_baseline"] = (
            result["token_hash"] == baseline["token_hash"]
            if baseline is not None and result["model"] == baseline["model"] else None
        )
        if best is None or result["decode_tok_s"] > best["decode_tok_s"]:
            best = result
        print(json.dumps(result), flush=True)
        with args.out.open("a") as file:
            file.write(json.dumps(result) + "\n")
    if best:
        targets = [target for target in (30, 50, 100) if best["decode_tok_s"] >= target]
        print(f"Fastest run {best['config']} at {best['decode_tok_s']:.2f} token/s. Reached stages {targets}.",
              flush=True)
    else:
        raise SystemExit("No benchmark completed.")


if __name__ == "__main__":
    main()
