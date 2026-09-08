"""List exact-duplicate conversation files in datasets_llama2, grouped by content hash."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "datasets_llama2"
OUT_PATH = ROOT / "datasets_llama2" / "duplicate_files.json"

groups = {}
for f in sorted(DATASET.glob("*/*.txt")):
    text_sha = hashlib.sha256(f.read_text().encode()).hexdigest()
    groups.setdefault(text_sha, []).append(str(f.relative_to(ROOT)))

dup_items = [(sha, files) for sha, files in groups.items() if len(files) > 1]
dup_items.sort(key=lambda item: (-len(item[1]), item[1][0]))

report = {
    "dataset": "datasets_llama2",
    "total_files": sum(len(v) for v in groups.values()),
    "unique_texts": len(groups),
    "duplicate_groups": len(dup_items),
    "duplicate_files": sum(len(files) - 1 for _, files in dup_items),
    "groups": [{"text_sha": sha, "files": files} for sha, files in dup_items],
}

OUT_PATH.write_text(json.dumps(report, indent=2))
print(f"wrote {OUT_PATH} — {report['duplicate_groups']} duplicate groups, "
      f"{report['duplicate_files']} redundant files out of {report['total_files']}")
