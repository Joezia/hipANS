#!/usr/bin/env python3
"""Per-iteration cycle accounting for the six measured points (MI350X, Qwen3.8 bf16 expert matrix).

Input : output directory of scripts/reproduce/profile.sh
          <name>/analysis.txt                  rocprof-compute analyze output (counters)
          <name>/pc/<name>_results.json         rocprofv3 host_trap PC samples (code-object offsets)
          build/reproduce/bin/hipans_decode_bench_<variant>   the profiled binary (its gfx950 code object is disassembled)
Output: JSON with, per point,
  * counter-derived cycles per decode iteration (rocprof-compute section 7.2 divided by iterations),
  * PC-sampling share of every decode step, mapped by sample address (not by instruction text),
  * step cycles = share x counter cycles per iteration.

One decode iteration = one wave decoding 64 output bytes (one row of the 64 x 128 B tile).
Total iterations of a profiled dispatch = replicas x raw_bytes / 64.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path

LLVM = Path(os.environ.get("LLVM_BIN", "/opt/rocm/llvm/bin"))
REPLICAS = 16
RAW_BYTES = 67_108_864
ITERATIONS = REPLICAS * RAW_BYTES // 64  # 16,777,216 for every block size
POINTS = [  # (dir name, point label, binary variant, block bytes, stage kernel)
    ("r0", "R0", "barrier1_b8192", 8192, False),
    ("o1", "O1", "barrier0_b8192", 8192, False),
    ("o2", "O1+O2", "barrier0_b8192", 8192, True),
    ("o3_4k", "O1+O2+O3 @ 4 KiB", "barrier0_b4096", 4096, True),
    ("o3_2k", "O1+O2+O3 @ 2 KiB", "barrier0_b2048", 2048, True),
    ("o3_1k", "O1+O2+O3 @ 1 KiB", "barrier0_b1024", 1024, True),
]
STEP_ORDER = ["1_lookup", "2_barrier", "3_update", "45_vote_prefix", "6_refill", "7_write_loop", "0_outside_loop"]
METRIC_RE = re.compile(r"^\s*│\s*(\d+\.\d+\.\d+)\s*│\s*([^│]+?)\s*│\s*([0-9.eE+-]+)\s*│")


def parse_counters(path: Path) -> dict[str, tuple[str, float]]:
    m = {}
    for line in path.read_text(errors="replace").splitlines():
        g = METRIC_RE.match(line)
        if g:
            m[g.group(1)] = (g.group(2), float(g.group(3)))
    return m


def counter_cycles(m: dict[str, tuple[str, float]]) -> dict:
    need = {"7.2.0": "Kernel Time", "7.2.1": "Kernel Time (Cycles)", "7.2.2": "Instructions per wavefront",
            "7.2.3": "Wave Cycles", "7.2.4": "Dependency Wait Cycles", "7.2.5": "Issue Wait Cycles",
            "7.2.6": "Active Cycles", "7.2.7": "Wavefront Occupancy", "7.1.2": "Total Wavefronts",
            "7.1.5": "VGPRs", "7.1.8": "LDS Allocation", "11.2.0": "IPC", "11.2.3": "VALU Utilization"}
    for k, name in need.items():
        if k not in m or m[k][0] != name:
            raise RuntimeError(f"metric {k} {name!r} missing or renamed: {m.get(k)}")
    g = lambda k: m[k][1]
    wc = g("7.2.3")
    out = {
        "kernel_time_ns": g("7.2.0"),
        "kernel_time_cycles": g("7.2.1"),
        "core_clock_GHz": g("7.2.1") / g("7.2.0"),
        "total_wavefronts": g("7.1.2"),
        "vgprs": g("7.1.5"),
        "lds_allocation_bytes": g("7.1.8"),
        "avg_resident_waves": g("7.2.7"),
        "ipc": g("11.2.0"),
        "valu_utilization_percent": g("11.2.3"),
        "wave_cycles": wc,
        "iterations": ITERATIONS,
        "cycles_per_iteration": {
            "total": wc / ITERATIONS,
            "dependency_wait": g("7.2.4") / ITERATIONS,
            "issue_wait": g("7.2.5") / ITERATIONS,
            "active": g("7.2.6") / ITERATIONS,
        },
        "instructions_per_iteration": g("7.2.2") * g("7.1.2") / ITERATIONS,
        "components_sum_over_wave_cycles": (g("7.2.4") + g("7.2.5") + g("7.2.6")) / wc,
    }
    return out


def code_object_bytes(binary: Path, tmp: Path) -> Path:
    fat, co = tmp / "fat.bin", tmp / "gfx950.co"
    subprocess.run([LLVM / "llvm-objcopy", f"--dump-section=.hip_fatbin={fat}", str(binary), "/dev/null"], check=True)
    subprocess.run([LLVM / "clang-offload-bundler", "--unbundle", "--type=o",
                    "--targets=hipv4-amdgcn-amd-amdhsa--gfx950", f"--input={fat}", f"--output={co}"], check=True)
    return co


def disassemble_kernel(co: Path, stage: bool):
    """Return ({addr: text}, {label: addr}) of hipansLDS<128,false,false,stage>."""
    txt = subprocess.run([LLVM / "llvm-objdump", "-d", "--no-show-raw-insn", "--symbolize-operands",
                          "--mcpu=gfx950", str(co)], check=True, capture_output=True, text=True).stdout.splitlines()
    sym = f"<_Z9hipansLDSILi128ELb0ELb0ELb{1 if stage else 0}EEvPKhmPKjS1_Phjjmmj>:"
    start = next(i for i, l in enumerate(txt) if l.endswith(sym) and re.match(r"^[0-9a-f]{16} ", l))
    ins, labels = {}, {}
    for l in txt[start + 1:]:
        if re.match(r"^[0-9a-f]{16} <_Z", l):
            break
        m = re.match(r"^([0-9a-f]{16}) <(L\d+)>:", l)
        if m:
            labels[m.group(2)] = int(m.group(1), 16)
            continue
        m = re.match(r"^\s*(\S.*?)\s*//\s*([0-9A-Fa-f]{12}):", l)
        if m:
            ins[int(m.group(2), 16)] = re.sub(r"\s+", " ", m.group(1))
    return ins, labels


def find_decode_loop(ins: dict[int, str], labels: dict[str, int]):
    best = None
    for a, t in sorted(ins.items()):
        mt = re.match(r"s_(?:cbranch_\w+|branch) (L\d+)$", t)
        if not mt or labels[mt.group(1)] > a:
            continue
        lo = labels[mt.group(1)]
        body = [x for x in ins if lo <= x <= a]
        texts = [ins[x] for x in body]
        if any(s.startswith("v_mad_u32_u24") for s in texts) and any(s.startswith("ds_read_b32") for s in texts) \
                and any(s.startswith("ds_write_b8") for s in texts):
            if best is None or (a - lo) < (best[1] - best[0]):
                best = (lo, a)
    if best is None:
        raise RuntimeError("decode loop not found (unrolled?)")
    return best


def classify_loop(ins: dict[int, str], lo: int, hi: int) -> dict[int, str]:
    """Assign every instruction of the 64-lane decode loop to one of the seven steps by address."""
    addrs = [a for a in sorted(ins) if lo <= a <= hi]
    step: dict[int, str] = {}
    for a in addrs:
        t = ins[a]
        op = t.split(" ")[0]
        if op == "s_barrier":
            step[a] = "2_barrier"
        elif op == "ds_read_b32" or re.match(r"v_and_b32(_e32)? v\d+, 0x3ff,", t) \
                or re.match(r"v_lshl_add_u32 v\d+, v\d+, 2, 0$", t) or re.match(r"v_lshrrev_b32(_e32)? v\d+, 10, v\d+", t):
            step[a] = "1_lookup"
        elif op in ("v_bfe_u32", "v_mad_u32_u24") or re.match(r"v_lshrrev_b32(_e32)? v\d+, 20, v\d+", t):
            step[a] = "3_update"
        elif re.match(r"v_cmp_gt_u32(_e32)? vcc, s\d+, v\d+", t) or re.match(r"s_and_saveexec_b64 s\[\d+:\d+\], vcc$", t) \
                or "vcc_lo" in t or "vcc_hi" in t or op == "v_bcnt_u32_b32" or op.startswith("v_sub_u32") \
                or op == "v_lshl_add_u64" or re.match(r"v_lshl_add_u32 v\d+, v\d+, 1,", t):
            step[a] = "45_vote_prefix"
        elif op in ("global_load_ushort", "ds_read_u16", "v_perm_b32"):
            step[a] = "6_refill"
        elif op == "ds_write_b8" or re.match(r"v_and_b32(_e32)? v\d+, 0xff, v\d+", t):
            step[a] = "7_write_loop"
        elif op.startswith("s_waitcnt"):
            step[a] = "WAIT"
        else:
            step[a] = "7_write_loop"  # exec restore, loop counter, branches
    # attribute each s_waitcnt to the nearest preceding memory op in the loop body (cyclic)
    kinds = {a: ("1_lookup" if ins[a].startswith("ds_read_b32") else "6_refill")
             for a in addrs if ins[a].split(" ")[0] in ("ds_read_b32", "global_load_ushort", "ds_read_u16")}
    order = [a for a in addrs if a in kinds]
    for a in addrs:
        if step[a] != "WAIT":
            continue
        prev = [x for x in order if x < a]
        step[a] = kinds[prev[-1] if prev else order[-1]]
    return step


def pc_steps(results_json: Path, ins, labels, lo, hi) -> dict:
    d = json.loads(results_json.read_text())["rocprofiler-sdk-tool"][0]
    samples = d["buffer_records"]["pc_sample_host_trap"]
    kernel_lo, kernel_hi = min(ins), max(ins)
    loop_step = classify_loop(ins, lo, hi)
    cnt, in_kernel, unmatched = collections.Counter(), 0, 0
    for s in samples:
        off = s["record"]["pc"]["code_object_offset"]
        if off < kernel_lo or off > kernel_hi:
            continue
        in_kernel += 1
        if off not in ins:
            unmatched += 1
            continue
        cnt[loop_step[off] if lo <= off <= hi else "0_outside_loop"] += 1
    tot = sum(cnt.values())
    return {
        "samples_total": len(samples), "samples_in_decode_kernel": in_kernel, "samples_unmatched_address": unmatched,
        "share_percent": {k: 100.0 * cnt[k] / tot for k in STEP_ORDER},
        "loop_address_range": [hex(lo), hex(hi)],
        "loop_static_instructions": sum(1 for a in ins if lo <= a <= hi),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile-dir", type=Path, required=True, help="output directory of profile.sh")
    ap.add_argument("--bin-dir", type=Path, required=True, help="build/reproduce/bin")
    ap.add_argument("--throughput-json", type=Path, required=True,
                    help="summary.json of run_throughput.sh (or published_mi350x.json)")
    ap.add_argument("--output", type=Path, required=True)
    a = ap.parse_args()
    thr = {p["point"]: p for p in json.loads(a.throughput_json.read_text())["points"]}
    result = {"schema": "hipans-cycles-v1", "replicas": REPLICAS, "iterations": ITERATIONS, "points": {}}
    for name, label, variant, block, stage in POINTS:
        entry = {"binary": f"hipans_decode_bench_{variant}", "block_bytes": block, "iterations_per_block": block // 64}
        c = counter_cycles(parse_counters(a.profile_dir / name / "analysis.txt"))
        entry["counters"] = c
        with tempfile.TemporaryDirectory() as tmp:
            co = code_object_bytes(a.bin_dir / f"hipans_decode_bench_{variant}", Path(tmp))
            ins, labels = disassemble_kernel(co, stage)
        lo, hi = find_decode_loop(ins, labels)
        pc = pc_steps(a.profile_dir / name / "pc" / f"{name}_results.json", ins, labels, lo, hi)
        entry["pc_sampling"] = pc
        tot = c["cycles_per_iteration"]["total"]
        entry["step_cycles_per_iteration"] = {k: pc["share_percent"][k] / 100.0 * tot for k in STEP_ORDER}
        entry["measured_throughput_GBps"] = thr[label]["median_GBps"]
        result["points"][name] = entry
    a.output.write_text(json.dumps(result, indent=1) + "\n")
    for name, e in result["points"].items():
        print(name, json.dumps({"cycles/iter": round(e["counters"]["cycles_per_iteration"]["total"], 1),
                                "steps": {k: round(v, 1) for k, v in e["step_cycles_per_iteration"].items()}}))


if __name__ == "__main__":
    main()
