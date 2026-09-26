# -*- coding: utf-8 -*-
"""从 pool_pruned.json 二次抽样出更瘦的池（避免重读 375 万扩充池）。

策略：保留全部原池候选 + 全部 A/B 型，C 型按 (路线, 箱数) 组只留最短 K 个。
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
    ap.add_argument("--keep-c", type=int, default=6)
    ap.add_argument("--keep-ab", type=int, default=0,
                    help="A/B 型每组保留数，0=全保留")
    ap.add_argument("--out", default="pool_slim.json")
    args = ap.parse_args()

    pool = json.loads((OUT / "pool_pruned.json").read_text())
    orig = json.loads((ROOT / "Q2_开源对比与优化" / "optimization_pool.json").read_text())
    okey = {(c["model"], tuple(sorted(c["boxes"])), tuple(c["route"])) for c in orig}

    keep, grp = [], defaultdict(list)
    for c in pool:
        k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
        if k in okey:
            keep.append(c)
        else:
            grp[(c["model"], tuple(c["route"]), len(c["boxes"]))].append(c)
    for (g, _r, _n), lst in grp.items():
        lst.sort(key=lambda z: z["duration_s"])
        k = args.keep_c if g == "C" else (args.keep_ab or len(lst))
        keep.extend(lst[:k])

    seen, out = set(), []
    for c in keep:
        k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
        if k not in seen:
            seen.add(k)
            out.append(c)

    allb = sorted({b for c in orig for b in c["boxes"]})
    cov = Counter(b for c in out for b in c["boxes"])
    miss = [b for b in allb if cov[b] == 0]
    ds = [c["duration_s"] for c in out]
    print(f"瘦池 {len(out)} 个  机型 {dict(Counter(c['model'] for c in out))}")
    print(f"  时长 {min(ds):.0f}~{max(ds):.0f}s  箱覆盖 {len(allb)-len(miss)}/{len(allb)}"
          + (f" 缺{miss}" if miss else " ✅"))
    if miss:
        print("⚠️ 覆盖不全，不保存")
        return
    (OUT / args.out).write_text(json.dumps(out, ensure_ascii=False))
    print("已保存", OUT / args.out)


if __name__ == "__main__":
    main()
