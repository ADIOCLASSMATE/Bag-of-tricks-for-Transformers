"""Tokenize fineweb-edu parquet data with the standard GPT-2 tokenizer.

Reads raw text from parquet files, encodes with tiktoken's gpt2 encoding,
and writes binary shards compatible with trainlib/dataloader.py.

Val split: the FIRST --num-val-docs documents become fineweb_val_*.bin,
the rest become fineweb_train_*.bin.  This matches the convention used in
the original download_hf_docs_and_tokenize.py pipeline.

Parallelism: use --num-workers to control how many parquet files are
processed simultaneously.  Each worker uses its own thread pool for
encoding.  Default uses 3/4 of available CPUs.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


SHARD_SIZE = 10**8
DATAFILE_MAGIC = 20240520
DATAFILE_VERSION = 1
APPEND_EOS = False
DEFAULT_NUM_VAL_DOCS = 10_000

GPT2_EOT = 50256
GPT2_VOCAB_SIZE = 50257

# Per-parquet-file batch size used by worker threads when reading row groups
PARQUET_BATCH = max(1, int(os.environ.get("TOKENIZE_PARQUET_BATCH", "65536")))
# Number of texts to process in one ThreadPoolExecutor.map call inside a worker
WORKER_ENCODE_CHUNK = max(1, int(os.environ.get("TOKENIZE_ENCODE_CHUNK", "8192")))


def write_datafile(path: Path, toks: np.ndarray) -> None:
    if len(toks) >= 2**31:
        raise ValueError("token count too large")
    header = np.zeros(256, dtype="<i4")
    header[0] = DATAFILE_MAGIC
    header[1] = DATAFILE_VERSION
    header[2] = len(toks)
    toks = np.asarray(toks)
    if toks.dtype != np.uint16:
        if not ((0 <= toks).all() and (toks < 2**16).all()):
            raise ValueError("token dictionary too large for uint16")
        toks = toks.astype("<u2", copy=False)
    else:
        toks = toks.astype("<u2", copy=False)
    with path.open("wb") as f:
        f.write(header.tobytes())
        f.write(toks.tobytes())


def _count_docs_in_file(parquet_path: Path) -> int:
    import pyarrow.parquet as pq

    return pq.ParquetFile(parquet_path).metadata.num_rows


def _pre_scan(input_dir: Path) -> list[tuple[Path, int]]:
    """Return sorted list of (path, doc_count) for every parquet file."""
    files = sorted(input_dir.glob("*.parquet"))
    if not files:
        sys.exit(f"No .parquet files found in {input_dir}")
    return [(f, _count_docs_in_file(f)) for f in files]


# ---------------------------------------------------------------------------
# Worker (runs in a subprocess)
# ---------------------------------------------------------------------------


def _encode_texts_in_threads(texts: list[str], num_threads: int) -> list[list[int]]:
    """Encode a list of text strings in parallel using encode_ordinary."""
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    with ThreadPoolExecutor(max_workers=max(1, num_threads)) as ex:
        return list(ex.map(enc.encode_ordinary, texts, chunksize=max(1, len(texts) // num_threads // 4)))


def _process_file_fully(args: tuple[Path, Path, int, int]) -> tuple[int, int, int, list[Path]]:
    """Process one parquet file: encode every document, write to temp shards.

    Returns (file_index, doc_count, token_count, [temp_shard_paths]).
    """
    import pyarrow.parquet as pq

    parquet_path, temp_dir, num_threads, shard_size = args

    pf = pq.ParquetFile(parquet_path)
    buf = np.empty((shard_size,), dtype=np.uint16)
    fill = 0
    shard_paths: list[Path] = []
    doc_count = 0
    token_count = 0

    def _flush() -> None:
        nonlocal fill
        if fill == 0:
            return
        path = temp_dir / f"_{parquet_path.stem}_s{len(shard_paths):04d}.bin"
        write_datafile(path, buf[:fill])
        shard_paths.append(path)
        fill = 0

    for batch in pf.iter_batches(columns=["text"], batch_size=PARQUET_BATCH):
        texts = batch.column("text").to_pylist()
        batches = [texts[i : i + WORKER_ENCODE_CHUNK] for i in range(0, len(texts), WORKER_ENCODE_CHUNK)]

        for chunk in batches:
            token_lists = _encode_texts_in_threads(chunk, num_threads)
            for raw_tokens in token_lists:
                toks = np.empty((len(raw_tokens) + 1,), dtype=np.uint16)
                toks[0] = GPT2_EOT
                toks[1:] = raw_tokens

                doc_count += 1
                token_count += len(toks)

                pos = 0
                while pos < len(toks):
                    take = min(shard_size - fill, len(toks) - pos)
                    buf[fill : fill + take] = toks[pos : pos + take]
                    fill += take
                    pos += take
                    if fill == shard_size:
                        _flush()

    _flush()
    return (doc_count, token_count, shard_paths)


def _process_file_with_split(
    args: tuple[Path, Path, int, int, int, int | None],
) -> tuple[int, int, list[Path], int, int, list[Path]]:
    """Process one parquet file, splitting at *split_at* docs.

    Docs [0, split_at) → val.  Docs [split_at, max_total_docs) → train.
    If max_total_docs is None, processes the entire file.

    Returns (val_docs, val_tokens, val_shard_paths,
             train_docs, train_tokens, train_shard_paths).
    """
    import pyarrow.parquet as pq

    parquet_path, temp_dir, num_threads, shard_size, split_at, max_total_docs = args

    pf = pq.ParquetFile(parquet_path)

    def _make_flush(buf, fill, shard_paths, label):
        def _flush():
            nonlocal buf, fill, shard_paths
            if fill[0] == 0:
                return
            path = temp_dir / f"_{parquet_path.stem}_{label}_s{len(shard_paths):04d}.bin"
            write_datafile(path, buf[: fill[0]])
            shard_paths.append(path)
            fill[0] = 0
        return _flush

    val_buf = np.empty((shard_size,), dtype=np.uint16)
    train_buf = np.empty((shard_size,), dtype=np.uint16)
    val_fill = [0]
    train_fill = [0]
    val_paths: list[Path] = []
    train_paths: list[Path] = []
    val_flush = _make_flush(val_buf, val_fill, val_paths, "v")
    train_flush = _make_flush(train_buf, train_fill, train_paths, "t")

    val_docs = 0
    val_tokens = 0
    train_docs = 0
    train_tokens = 0
    total = 0
    done = False

    for batch in pf.iter_batches(columns=["text"], batch_size=PARQUET_BATCH):
        texts = batch.column("text").to_pylist()
        chunks = [texts[i : i + WORKER_ENCODE_CHUNK] for i in range(0, len(texts), WORKER_ENCODE_CHUNK)]

        for chunk in chunks:
            token_lists = _encode_texts_in_threads(chunk, num_threads)
            for raw_tokens in token_lists:
                toks = np.empty((len(raw_tokens) + 1,), dtype=np.uint16)
                toks[0] = GPT2_EOT
                toks[1:] = raw_tokens

                is_val = val_docs < split_at
                if is_val:
                    val_docs += 1
                    val_tokens += len(toks)
                    buf, fill = val_buf, val_fill
                    flush = val_flush
                else:
                    train_docs += 1
                    train_tokens += len(toks)
                    buf, fill = train_buf, train_fill
                    flush = train_flush

                pos = 0
                while pos < len(toks):
                    take = min(shard_size - fill[0], len(toks) - pos)
                    buf[fill[0] : fill[0] + take] = toks[pos : pos + take]
                    fill[0] += take
                    pos += take
                    if fill[0] == shard_size:
                        flush()

                total += 1
                if max_total_docs is not None and total >= max_total_docs:
                    done = True
                    break
            if done:
                break
        if done:
            break

    val_flush()
    train_flush()
    return (val_docs, val_tokens, val_paths, train_docs, train_tokens, train_paths)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _merge_shards(
    temp_paths: list[Path],
    output_dir: Path,
    prefix: str,
    *,
    start_idx: int = 0,
) -> int:
    """Rename temp shard paths into output_dir / prefix_XXXXXX.bin, return next index."""
    idx = start_idx
    for src in sorted(temp_paths):
        dst = output_dir / f"{prefix}_{idx:06d}.bin"
        src.rename(dst)
        idx += 1
    return idx


def tokenize_and_export(
    input_dir: Path,
    output_dir: Path,
    *,
    num_val_docs: int,
    shard_size: int,
    num_workers: int,
    max_docs: int | None,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("fineweb_train_*.bin", "fineweb_val_*.bin"):
        for stale in output_dir.glob(pattern):
            stale.unlink()

    file_docs = _pre_scan(input_dir)
    total_docs_available = sum(n for _, n in file_docs)
    if max_docs is not None:
        total_docs = min(max_docs, total_docs_available)
    else:
        total_docs = total_docs_available
    val_docs = min(num_val_docs, total_docs)

    print(f"Files:   {len(file_docs)}")
    print(f"Docs:    {total_docs_available:,} total, processing {total_docs:,}")
    print(f"Val:     first {val_docs:,} docs → fineweb_val_*.bin")
    print(f"Train:   remaining docs → fineweb_train_*.bin")
    print(f"Workers: {num_workers}")
    print()

    cpu_count = len(os.sched_getaffinity(0))
    threads_per_worker = max(2, cpu_count // num_workers)

    # Classify files by val boundary and optional max_docs boundary.
    # Val = first val_docs documents.  If max_docs is set, stop after total_docs.
    cum = 0
    val_files: list[int] = []
    boundary_file: int | None = None
    boundary_val_docs = 0
    train_files: list[int] = []
    stop_after_val = val_docs  # doc index where val→train transition happens

    for fi, (_, ndocs) in enumerate(file_docs):
        if cum >= total_docs:
            break  # past max_docs limit

        if cum + ndocs <= stop_after_val:
            # Entirely val
            val_files.append(fi)
        elif cum < stop_after_val:
            # Straddles val boundary → split needed
            boundary_file = fi
            boundary_val_docs = stop_after_val - cum
        elif max_docs is not None and cum + ndocs > total_docs:
            # Straddles max_docs boundary → split needed (rest is train, stop after total_docs - cum)
            boundary_file = fi
            boundary_val_docs = 0  # no val docs in this boundary file
        else:
            # Entirely train
            train_files.append(fi)

        cum += ndocs

    temp_dir = output_dir / "_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    train_idx = 0
    val_idx = 0

    # --- Process val-only files in parallel ---
    if val_files:
        print(f"Processing {len(val_files)} val-only file(s) in parallel...", flush=True)
        val_args = [
            (file_docs[fi][0], temp_dir, threads_per_worker, shard_size) for fi in val_files
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_file_fully, a): fi for fi, a in zip(val_files, val_args)}
            for fut in as_completed(futures):
                doc_count, tok_count, paths = fut.result()
                val_idx = _merge_shards(paths, output_dir, "fineweb_val", start_idx=val_idx)
                fi = futures[fut]
                print(f"  [{fi:3d}] {file_docs[fi][0].name}: {doc_count:,} docs, {tok_count:,} tokens → val shards", flush=True)

    # --- Process boundary file (single pass, split at boundary_val_docs) ---
    if boundary_file is not None:
        bpath, bdocs = file_docs[boundary_file]
        cum_before = sum(file_docs[fi][1] for fi in val_files)
        limit = None if max_docs is None else (total_docs - cum_before)
        print(f"Processing boundary file [{boundary_file}] {bpath.name} "
              f"(val={boundary_val_docs}, max={limit}) ...", flush=True)

        v_docs, v_toks, v_paths, t_docs, t_toks, t_paths = _process_file_with_split(
            (bpath, temp_dir, threads_per_worker, shard_size, boundary_val_docs, limit)
        )
        if v_paths:
            val_idx = _merge_shards(v_paths, output_dir, "fineweb_val", start_idx=val_idx)
            print(f"  [{boundary_file:3d}] val part:  {v_docs:,} docs, {v_toks:,} tokens", flush=True)
        if t_paths:
            train_idx = _merge_shards(t_paths, output_dir, "fineweb_train", start_idx=train_idx)
            print(f"  [{boundary_file:3d}] train part: {t_docs:,} docs, {t_toks:,} tokens", flush=True)

    # --- Process train-only files in parallel ---
    if train_files:
        print(f"Processing {len(train_files)} train-only file(s) in parallel...", flush=True)
        train_args = [
            (file_docs[fi][0], temp_dir, threads_per_worker, shard_size) for fi in train_files
        ]
        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            futures = {pool.submit(_process_file_fully, a): fi for fi, a in zip(train_files, train_args)}
            for fut in as_completed(futures):
                doc_count, tok_count, paths = fut.result()
                train_idx = _merge_shards(paths, output_dir, "fineweb_train", start_idx=train_idx)
                fi = futures[fut]
                print(f"  [{fi:3d}] {file_docs[fi][0].name}: {doc_count:,} docs, {tok_count:,} tokens → train shards", flush=True)

    # Cleanup temp dir
    for leftover in temp_dir.glob("_*.bin"):
        leftover.unlink()
    temp_dir.rmdir()

    elapsed = time.time() - t0
    print(f"\nElapsed: {elapsed/60:.1f} min")

    # Count final shards
    val_shards = len(list(output_dir.glob("fineweb_val_*.bin")))
    train_shards = len(list(output_dir.glob("fineweb_train_*.bin")))
    val_tokens = sum(
        int(np.fromfile(f, dtype="<i4", count=3)[2])
        for f in output_dir.glob("fineweb_val_*.bin")
    )
    train_tokens = sum(
        int(np.fromfile(f, dtype="<i4", count=3)[2])
        for f in output_dir.glob("fineweb_train_*.bin")
    )

    return {
        "docs_total": total_docs,
        "docs_val": val_docs,
        "docs_train": total_docs - val_docs,
        "tokens_val": val_tokens,
        "tokens_train": train_tokens,
        "tokens_total": val_tokens + train_tokens,
        "val_shards": val_shards,
        "train_shards": train_shards,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    cpu_count = len(os.sched_getaffinity(0))
    parser = argparse.ArgumentParser(
        description="Tokenize fineweb-edu parquet data with GPT-2 tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Val / train split:
  The FIRST --num-val-docs documents (in sorted parquet-file order) become
  fineweb_val_*.bin.  Everything else becomes fineweb_train_*.bin.
  This matches the convention from the original download_hf_docs_and_tokenize.py.

Examples:
  # Full tokenization with 8 workers
  uv run python data/tokenize_fineweb_edu_gpt2.py -o data/datasets/fineweb-edu_100BT_gpt2 -w 8

  # First 5% of the data only (quick test)
  uv run python data/tokenize_fineweb_edu_gpt2.py -o data/datasets/test --max-docs 4860000 -w 8
        """,
    )
    parser.add_argument(
        "--input-dir", "-i",
        default="/inspire/dataset/fineweb-edu/v1/sample/100BT",
        help="Directory containing fineweb-edu .parquet files",
    )
    parser.add_argument(
        "--output-dir", "-o",
        required=True,
        help="Output directory for tokenized .bin shards",
    )
    parser.add_argument(
        "--num-val-docs",
        type=int,
        default=DEFAULT_NUM_VAL_DOCS,
        help=f"Number of validation documents (default: {DEFAULT_NUM_VAL_DOCS:,})",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Only tokenize the first N documents (default: process all)",
    )
    parser.add_argument(
        "--num-workers", "-w",
        type=int,
        default=None,
        help=f"Number of parallel file-processing workers (default: min(48, cpu_count//4), currently {min(48, max(4, cpu_count // 4))})",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=SHARD_SIZE,
        help=f"Tokens per shard (default: {SHARD_SIZE:,})",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"Input directory not found: {input_dir}")

    cpu_count = len(os.sched_getaffinity(0))
    num_workers = args.num_workers
    if num_workers is None:
        num_workers = min(48, max(4, cpu_count // 4))

    output_dir = Path(args.output_dir).resolve()
    print(f"Input:   {input_dir}")
    print(f"Output:  {output_dir}")
    print(f"Vocab:   GPT-2 ({GPT2_VOCAB_SIZE})")
    print(f"Shard:   {args.shard_size:,} tokens")
    print()

    stats = tokenize_and_export(
        input_dir,
        output_dir,
        num_val_docs=args.num_val_docs,
        shard_size=args.shard_size,
        num_workers=num_workers,
        max_docs=args.max_docs,
    )

    print()
    print("=== Done ===")
    print(f"Docs:       {stats['docs_total']:,}")
    print(f"  val:      {stats['docs_val']:,}")
    print(f"  train:    {stats['docs_train']:,}")
    print(f"Tokens:     {stats['tokens_total']:,}")
    print(f"  val:      {stats['tokens_val']:,}")
    print(f"  train:    {stats['tokens_train']:,}")
    print(f"Shards:     {stats['val_shards'] + stats['train_shards']}")
    print(f"  val:      {stats['val_shards']}")
    print(f"  train:    {stats['train_shards']}")


if __name__ == "__main__":
    main()
