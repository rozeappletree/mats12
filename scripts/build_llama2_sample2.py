"""Build datasets_llama2_sample2: datasets_llama2 with exact-duplicate conversations
removed and files renumbered contiguously per attribute/level.

Training code (scripts/dataset.py TextDataset, src/probe_common.py TextDataset) loads
every *.txt in a directory via os.listdir/glob and reads the label off the filename
suffix (`..._<level>.txt`) — the numeric conversation_<i> index is never parsed as an
actual array index, so gaps from dropped duplicates would not break training. Files are
still renumbered here (0..n-1 per attribute/level) so the new dataset looks like a
normal freshly-generated batch rather than one with dedup holes.
"""
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "datasets_llama2"
DST = ROOT / "datasets_llama2_sample2"
ATTRS = ["gullibility", "rationality", "seriousness", "certainty_seeking"]


def parse_name(path: Path):
    # conversation_<idx>_<attribute>_<level>.txt
    stem = path.stem
    idx_str, rest = stem.split("_", 2)[1], stem.split("_", 2)[2]
    level = rest.rsplit("_", 1)[1]
    return int(idx_str), level


def main():
    if DST.exists():
        raise SystemExit(f"{DST} already exists — remove it first if you want to rebuild")

    total_src = 0
    total_dst = 0
    for attr in ATTRS:
        src_dir = SRC / attr
        if not src_dir.exists():
            continue
        dst_dir = DST / attr
        dst_dir.mkdir(parents=True, exist_ok=True)

        by_level = defaultdict(list)  # level -> [(idx, path, text)]
        for f in src_dir.glob("*.txt"):
            idx, level = parse_name(f)
            by_level[level].append((idx, f, f.read_text()))
        total_src += sum(len(v) for v in by_level.values())

        for level, items in by_level.items():
            items.sort(key=lambda t: t[0])  # keep lowest original idx as the survivor
            seen_sha = {}
            for idx, path, text in items:
                sha = hashlib.sha256(text.encode()).hexdigest()
                seen_sha.setdefault(sha, path)

            for new_idx, path in enumerate(seen_sha.values()):
                dst_name = f"conversation_{new_idx}_{attr}_{level}.txt"
                shutil.copyfile(path, dst_dir / dst_name)
                total_dst += 1

    print(f"{SRC.name}: {total_src} files -> {DST.name}: {total_dst} files "
          f"({total_src - total_dst} exact duplicates removed)")


if __name__ == "__main__":
    main()
