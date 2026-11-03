#!/usr/bin/env python3
"""Summarize a run_throughput.sh output directory and compare it with the published numbers.

usage: summarize_throughput.py <output-dir> <published_mi350x.json>
"""
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
published = json.loads(Path(sys.argv[2]).read_text())
rows = []
for i, ref in enumerate(published["points"]):
    verified = json.loads((out / "correctness" / f"point-{i}.json").read_text())
    if not verified["verified_actual_LDS"]:
        raise SystemExit(f"point {i} ({ref['point']}) was not byte-exact verified")
    runs = [json.loads((out / "timing" / f"point-{i}-run-{r}.json").read_text()) for r in (1, 2, 3)]
    gbps = [r["effective_LDS_GBps"] for r in runs]
    if {r["block_size_bytes"] for r in runs} != {ref["block_size_bytes"]} or \
            {r["input"] for r in runs} != {ref["input"]}:
        raise SystemExit(f"point {i} ({ref['point']}): block size or input differs from the published setup")
    rows.append({
        "point": ref["point"],
        "replicas": runs[0]["replicas"],
        "archive_bytes": runs[0]["artifact_bytes"],
        "archive_bytes_published": ref["archive_bytes"],
        "process_GBps": [round(x, 1) for x in gbps],
        "median_GBps": round(statistics.median(gbps), 1),
        "published_median_GBps": ref["median_GBps"],
        "difference_percent": round((statistics.median(gbps) / ref["median_GBps"] - 1) * 100, 2),
        "ms_per_launch": round(statistics.median(r["decode_ms"]["median"] for r in runs), 4),
    })
sha = {}
for line in (out / "archives" / "sha256.txt").read_text().splitlines():
    digest, name = line.split()
    if name.endswith(".check.ans"):
        continue
    sha[name.removeprefix("archives/block").removesuffix(".ans")] = digest
same = {k: sha[k] == v for k, v in published["archive_sha256_by_block_size"].items()}
(out / "summary.json").write_text(json.dumps({"points": rows, "archive_identical_to_published": same}, indent=2) + "\n")

print(f"{'point':<20}{'R':>4}  {'GB/s per process':<24}{'median':>8}{'published':>11}{'diff':>8}")
for r in rows:
    print(f"{r['point']:<20}{r['replicas']:>4}  {str(r['process_GBps']):<24}{r['median_GBps']:>8}{r['published_median_GBps']:>11}{r['difference_percent']:>7}%")
print("archives byte-identical to the published ones:", same)
