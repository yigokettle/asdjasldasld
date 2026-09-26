# -*- coding: utf-8 -*-
"""以原池 572 为基底，定向增补「能把 C 型负载卸给 A/B 型」的候选。

为什么要定向增补：
  瓶颈诊断显示 Cmax=6991s ≈ C 型机队下界 6942s（7 架次 / 2 机）。
  A 型 9 架次 / 4 机的下界只有 4448s，有大量空闲。
  故只需补充 A/B 型的短架次候选，让求解器把 C 型的箱子挪给 A/B。

为什么不能直接用大池：
  CP-SAT 在数千个可选区间 + AddCumulative 上连首个可行解都判不出（实测
  2329~21547 池全部 UNKNOWN）。必须把池控制在原池量级，并保证已知可行解
  100% 落在池内（暖启动才有效）。

增补规则：
  1. 原池 572 个全部保留 → 暖启动解必然可行
  2. 只补 A/B 型，且只补「单站、时长最短」的候选（最有利于并行填空）
  3. 每个服务区每个机型只补 --per 个，控制总规模
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="pool_c8.json", help="增补来源（已剪枝池）")
    ap.add_argument("--per", type=int, default=6,
                    help="每 (机型, 服务区, 箱数) 补充的最短候选数")
    ap.add_argument("--models", default="AB", help="只补这些机型")
    ap.add_argument("--max-boxes", type=int, default=3)
    ap.add_argument("--out", default="pool_aug.json")
    args = ap.parse_args()

    orig = json.loads((SRC / "optimization_pool.json").read_text())
    big = json.loads((OUT / args.src).read_text())
    okey = {(c["model"], tuple(sorted(c["boxes"])), tuple(c["route"])) for c in orig}
    print(f"原池 {len(orig)}，增补来源 {len(big)}")

    grp = defaultdict(list)
    for c in big:
        k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
        if k in okey:
            continue
        if c["model"] not in args.models:
            continue
        if len(c["route"]) != 1 or len(c["boxes"]) > args.max_boxes:
            continue
        grp[(c["model"], c["route"][0], len(c["boxes"]))].append(c)

    add = []
    for k, lst in grp.items():
        lst.sort(key=lambda z: z["duration_s"])
        add.extend(lst[:args.per])

    out = orig + add
    seen, ded = set(), []
    for c in out:
        k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
        if k not in seen:
            seen.add(k)
            ded.append(c)
    out = ded

    allb = sorted({b for c in orig for b in c["boxes"]})
    cov = Counter(b for c in out for b in c["boxes"])
    miss = [b for b in allb if cov[b] == 0]
    ds = [c["duration_s"] for c in out]
    print(f"\n增补池 {len(out)} 个（原 {len(orig)} + 新 {len(out)-len(orig)}）")
    print(f"  机型 {dict(Counter(c['model'] for c in out))}")
    print(f"  时长 {min(ds):.0f}~{max(ds):.0f}s  箱覆盖 {len(allb)-len(miss)}/{len(allb)}"
          + (f" 缺{miss}" if miss else " ✅"))
    for g in "ABC":
        s = [c for c in out if c["model"] == g]
        if s:
            print(f"  {g}: {len(s)} 个，最短 {min(c['duration_s'] for c in s):.0f}s")
    if miss:
        print("⚠️ 覆盖不全，不保存")
        return
    (OUT / args.out).write_text(json.dumps(out, ensure_ascii=False))
    print("已保存", OUT / args.out)


if __name__ == "__main__":
    main()
