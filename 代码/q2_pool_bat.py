# -*- coding: utf-8 -*-
"""针对「C 型电池周转瓶颈」定向构池。

瓶颈诊断（bs23，Cmax=6875s）：
    机型  架次  飞行/机队      (飞行+充电)/电池
    A     10    20823/4=5206   32534/6=5422  ← 电池略紧
    B      6    12027/2=6014   22749/4=5687
    C      7    13328/2=6664   27195/4=6799  ← 真瓶颈：电池
    全局完美配平下界 46179/8 = 5772 s

C 型电池是瓶颈的原因：满充 3000 s，而 C 型大批次架次能耗 5.4~6.2 kWh，
按 SOC 线性折算需充 2300~2500 s。4 组电池支撑 7 个架次时电池占用总量
27195 s，除以 4 得 6799 s，直接顶住 Cmax。

对策：让 C 型飞「低能耗」架次（能耗低 → 充电快 → 电池周转快），把腾出的
货箱交给 A 型（A 型满充仅 1800 s，且有 4 机 6 池）。因此本脚本：
  * C 型：按 (路线,箱数) 组优先保留「能耗最低」的候选，而非时长最短
  * A/B 型：保留时长最短的候选（用于填补并行空档）
  * 始终全量包含原池 572 + 指定暖启动解的全部候选（保证 hint 100% 命中）
"""
from __future__ import annotations
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"


def key(c):
    return (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="pool_c8.json")
    ap.add_argument("--keep-c", type=int, default=10,
                    help="C 型每 (路线,箱数) 组保留的最低能耗候选数")
    ap.add_argument("--keep-ab", type=int, default=6,
                    help="A/B 型每 (路线,箱数) 组保留的最短时长候选数")
    ap.add_argument("--c-max-boxes", type=int, default=5,
                    help="C 型只保留不超过此箱数的候选（小批次→低能耗→快充）")
    ap.add_argument("--ab-max-boxes", type=int, default=3)
    ap.add_argument("--hints", default="solution_bs23.json,solution_lexN23.json")
    ap.add_argument("--out", default="pool_bat.json")
    args = ap.parse_args()

    orig = json.loads((SRC / "optimization_pool.json").read_text())
    big = json.loads((OUT / args.src).read_text())
    must = {key(c) for c in orig}
    # 必含候选的完整记录来源：原池 + 暖启动解所引用的池
    bank = {key(c): c for c in orig}
    for c in big:
        bank.setdefault(key(c), c)

    # 暖启动解里的候选必须在池内
    for nm in args.hints.split(","):
        f = OUT / nm.strip()
        if not f.exists():
            continue
        hs = json.loads(f.read_text())
        for t in hs["trips"]:
            if "boxes" in t and "route" in t:
                k = (t["model"], tuple(sorted(t["boxes"])), tuple(t["route"]))
                must.add(k)
                # 解自带完整字段时，直接作为候选记录入 bank
                if k not in bank and "duration_s" in t:
                    bank[k] = dict(
                        model=t["model"], boxes=t["boxes"], route=t["route"],
                        mass_kg=t["mass_kg"], volume_m3=t["volume_m3"],
                        duration_s=t["duration_s"], return_s=t["duration_s"],
                        energy_kwh=t["energy_kwh"], charge_s=t["charge_s"],
                        delivery_offsets=t["delivery_offsets"],
                        delivery_times=t["delivery_offsets"])
        print(f"纳入暖启动解 {f.name}（Cmax={hs.get('cmax_s')}）")

    keep, grp = [], defaultdict(list)
    # 必含候选统一从 bank 取，避免 --src 已剔除导致缺失
    for k in must:
        if k in bank:
            keep.append(bank[k])
    for c in big:
        k = key(c)
        if k in must:
            continue
        g = c["model"]
        nb = len(c["boxes"])
        if g == "C":
            if nb <= args.c_max_boxes:
                grp[k[0], tuple(c["route"]), nb].append(c)
        else:
            if nb <= args.ab_max_boxes:
                grp[k[0], tuple(c["route"]), nb].append(c)

    for (g, _r, _n), lst in grp.items():
        if g == "C":
            lst.sort(key=lambda z: (z["energy_kwh"], z["duration_s"]))
            keep.extend(lst[:args.keep_c])
        else:
            lst.sort(key=lambda z: (z["duration_s"], z["energy_kwh"]))
            keep.extend(lst[:args.keep_ab])

    seen, out = set(), []
    for c in keep:
        k = key(c)
        if k not in seen:
            seen.add(k)
            out.append(c)

    got = {key(c) for c in out}
    missing_must = must - got
    allb = sorted({b for c in orig for b in c["boxes"]})
    cov = Counter(b for c in out for b in c["boxes"])
    miss = [b for b in allb if cov[b] == 0]
    print(f"\n池 {len(out)} 个  机型 {dict(Counter(c['model'] for c in out))}")
    ds = [c["duration_s"] for c in out]
    print(f"  时长 {min(ds):.0f}~{max(ds):.0f}s  箱覆盖 "
          f"{len(allb)-len(miss)}/{len(allb)}" + (f" 缺{miss}" if miss else " ✅"))
    print(f"  必含候选缺失 {len(missing_must)}"
          + (" ✅" if not missing_must else " ⚠️"))
    for g in "ABC":
        s = [c for c in out if c["model"] == g]
        if s:
            print(f"  {g}: {len(s)} 个，时长 {min(c['duration_s'] for c in s):.0f}~"
                  f"{max(c['duration_s'] for c in s):.0f}s，能耗 "
                  f"{min(c['energy_kwh'] for c in s):.2f}~"
                  f"{max(c['energy_kwh'] for c in s):.2f}kWh，充电 "
                  f"{min(c['charge_s'] for c in s):.0f}~"
                  f"{max(c['charge_s'] for c in s):.0f}s")
    if miss or missing_must:
        print("⚠️ 不保存")
        return
    (OUT / args.out).write_text(json.dumps(out, ensure_ascii=False))
    print("已保存", OUT / args.out)


if __name__ == "__main__":
    main()
