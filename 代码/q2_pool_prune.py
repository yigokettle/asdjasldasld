# -*- coding: utf-8 -*-
"""把 375 万扩充候选剪枝成 CP-SAT 可解规模，目标：打破 C 型机队瓶颈。

瓶颈诊断（lexN23）：
    A 型 9 架次/4 机 → 机队下界 4448 s（余量大）
    B 型 7 架次/2 机 → 6557 s
    C 型 7 架次/2 机 → 6942 s ≈ Cmax 6991 s      ← 真正的瓶颈
故 Cmax 受 C 型 2 架机总工作量支配，压 Cmax 必须把货箱从 C 型卸给 A 型。
原池 C 型候选以 6~7 箱大批次为主，求解器无法做这种再平衡。

剪枝规则（在保证「可覆盖全部 80 箱」的前提下尽量小）：
  1. 原池 572 个全部保留（保底，确保至少不劣于现有解）
  2. A/B 型：全部保留（本就只有 1.5 万，且是卸载 C 型负载的关键）
  3. C 型：对每个 (路线, 箱集大小) 只保留「时长最小」的前 K 个，
     并优先保留小批次（箱数 ≤ 5），使 C 型可飞短架次
  4. 支配剪枝：同一 (机型, 箱集) 只留时长最小的路线顺序
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Q2_架次优化"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-c", type=int, default=40,
                    help="C 型每个 (路线,箱数) 组保留的最短候选数")
    ap.add_argument("--c-max-boxes", type=int, default=6)
    args = ap.parse_args()

    pool = json.loads((OUT / "pool_expanded.json").read_text())
    orig = json.loads((ROOT / "Q2_开源对比与优化" / "optimization_pool.json").read_text())
    okey = {(c["model"], tuple(sorted(c["boxes"])), tuple(c["route"])) for c in orig}
    print(f"扩充池 {len(pool)}，原池 {len(orig)}")

    # ---- 规则4：同一 (机型, 箱集) 只留最短路线顺序 ----
    best = {}
    for c in pool:
        k = (c["model"], tuple(sorted(c["boxes"])))
        if k not in best or c["duration_s"] < best[k]["duration_s"]:
            best[k] = c
    pool = list(best.values())
    print(f"路线顺序去支配后 {len(pool)}")

    keep, drop_c = [], defaultdict(list)
    for c in pool:
        k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
        if k in okey:                       # 规则1
            keep.append(c)
        elif c["model"] in ("A", "B"):      # 规则2
            keep.append(c)
        else:                               # 规则3：C 型分组待筛
            nb = len(c["boxes"])
            if nb <= args.c_max_boxes:
                drop_c[(tuple(c["route"]), nb)].append(c)
    nC = 0
    for k, lst in drop_c.items():
        lst.sort(key=lambda z: z["duration_s"])
        keep.extend(lst[:args.keep_c])
        nC += min(len(lst), args.keep_c)
    print(f"C 型分组 {len(drop_c)} 组，抽取 {nC}")

    # 去重
    seen, out = set(), []
    for c in keep:
        k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
        if k in seen:
            continue
        seen.add(k)
        out.append(c)

    # 覆盖性检查
    allb = sorted({b for c in orig for b in c["boxes"]})
    cov = {b: 0 for b in allb}
    for c in out:
        for b in c["boxes"]:
            cov[b] += 1
    miss = [b for b, v in cov.items() if v == 0]
    ds = [c["duration_s"] for c in out]
    print(f"\n剪枝池 {len(out)} 个候选")
    print(f"  机型分布 {dict(Counter(c['model'] for c in out))}")
    print(f"  时长范围 {min(ds):.1f} ~ {max(ds):.1f} s")
    print(f"  箱覆盖 {len(allb)-len(miss)}/{len(allb)}" + (f" 缺 {miss}" if miss else " ✅"))
    for g in "ABC":
        s = [c for c in out if c["model"] == g]
        if s:
            print(f"  {g} 型 {len(s)} 个，时长 {min(c['duration_s'] for c in s):.0f}"
                  f"~{max(c['duration_s'] for c in s):.0f}s，"
                  f"箱数 {min(len(c['boxes']) for c in s)}~{max(len(c['boxes']) for c in s)}")
    if miss:
        print("⚠️ 覆盖不全，不保存")
        return
    (OUT / "pool_pruned.json").write_text(json.dumps(out, ensure_ascii=False))
    print("已保存", OUT / "pool_pruned.json")


if __name__ == "__main__":
    main()
