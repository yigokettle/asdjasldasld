# -*- coding: utf-8 -*-
"""Cmax 的「聚合资源松弛」下界——用来回答「N=23 能否低于某个 Cmax」。

松弛掉什么：
    真实问题里每架次要占用一架具体无人机 [s, s+dur) 和一组具体电池
    [s, s+dur+chg)，是带排程的 Cumulative 约束。本脚本只保留「面积」约束：

        机队:  sum_{选中且机型=g} dur_i      <= K_g * Cmax
        电池:  sum_{选中且机型=g} (dur+chg)_i <= B_g * Cmax

    任何可行排程必然满足这两条（时间窗 [0,Cmax] 内 K_g 台机器最多提供
    K_g*Cmax 的机时），故本松弛的最优值是真实 Cmax 的**严格下界**。

松弛后没有 start 变量、没有 Cumulative，只剩 0/1 选择 + 线性面积约束，
CP-SAT 秒级出解，可以直接吃下万级候选池。

结论用法：
    LB > 6000  →  N=23 时 Cmax<6000 **不可能**（可写进论文当不可达性证明）
    LB < 6000  →  不能排除，需要继续找排程
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--pool", default="Q2_架次优化/pool_c8.json")
    ap.add_argument("--timeliness", choices=["strict", "hard"], default="strict")
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--free-n", action="store_true",
                    help="不固定架次数，只求 Cmax 下界（更松，但排除性更强）")
    args = ap.parse_args()

    p = Path(args.pool)
    if not p.is_absolute():
        p = ROOT / p
    pool = json.loads(p.read_text())
    b = pd.read_csv(SRC / "optimized_box_delivery.csv").set_index("货箱编号")
    hard = {k: (None if pd.isna(v) else float(v)) for k, v in b.硬截止时刻_s.items()}
    exp = b.期望送达时刻_s.to_dict()
    strict = args.timeliness == "strict"

    print(f"候选池 {len(pool)}（{dict(Counter(c['model'] for c in pool))}）")
    boxes = sorted({x for c in pool for x in c["boxes"]})
    bidx = {x: i for i, x in enumerate(boxes)}
    print(f"覆盖箱数 {len(boxes)}")

    m = cp_model.CpModel()
    # Cmax 上界取一个宽松值
    HI = 12000
    cmax = m.NewIntVar(0, HI, "cmax")

    x = [m.NewBoolVar(f"x{i}") for i in range(len(pool))]

    # 时效性：架次最早只能 0 出发，故 offset 超过期望/硬截止的候选直接禁用
    killed = 0
    for i, c in enumerate(pool):
        lim = []
        for bx in c["boxes"]:
            off = c["delivery_offsets"][bx]
            if hard[bx] is not None:
                lim.append(hard[bx] - off)
            if strict:
                lim.append(exp[bx] - off)
        if lim and min(lim) < 0:
            m.Add(x[i] == 0)
            killed += 1
    print(f"时效性直接淘汰 {killed} 个候选")

    cover = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for bx in c["boxes"]:
            cover[bidx[bx]].append(x[i])
    for k, lst in enumerate(cover):
        if not lst:
            raise SystemExit(f"箱 {boxes[k]} 无候选覆盖")
        m.AddExactlyOne(lst)

    if not args.free_n:
        m.Add(sum(x) == args.n)

    # ---- 聚合面积约束（松弛核心）----
    for g in "ABC":
        idx = [i for i, c in enumerate(pool) if c["model"] == g]
        if not idx:
            continue
        dur = [int(math.ceil(pool[i]["duration_s"])) for i in idx]
        occ = [int(math.ceil(pool[i]["duration_s"] + pool[i]["charge_s"]))
               for i in idx]
        # sum dur_i * x_i <= K_g * cmax
        m.Add(sum(d * x[i] for d, i in zip(dur, idx)) <= DRONES[g] * cmax)
        m.Add(sum(o * x[i] for o, i in zip(occ, idx)) <= BATTERIES[g] * cmax)

    # 另外：任一被选中架次自身的时长也是 Cmax 的下界
    for i, c in enumerate(pool):
        m.Add(cmax >= int(math.ceil(c["duration_s"]))).OnlyEnforceIf(x[i])

    m.Minimize(cmax)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = args.seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = 2026
    s.parameters.log_search_progress = False
    r = s.Solve(m)
    st = s.StatusName(r)
    print(f"\n求解状态 {st}")
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("未找到解")
        return

    lb = s.BestObjectiveBound()
    val = s.Value(cmax)
    sel = [i for i in range(len(pool)) if s.Value(x[i])]
    print(f"松弛最优 Cmax = {val}s   （证明下界 {lb:.0f}s）")
    print(f"选中架次 {len(sel)}  机型 {dict(Counter(pool[i]['model'] for i in sel))}")
    for g in "ABC":
        gi = [i for i in sel if pool[i]["model"] == g]
        if not gi:
            continue
        fd = sum(pool[i]["duration_s"] for i in gi)
        fo = sum(pool[i]["duration_s"] + pool[i]["charge_s"] for i in gi)
        print(f"  {g}: {len(gi)} 架次  飞行 {fd:.0f}s/{DRONES[g]}机="
              f"{fd/DRONES[g]:.0f}s  占池 {fo:.0f}s/{BATTERIES[g]}池="
              f"{fo/BATTERIES[g]:.0f}s")
    print(f"\n{'='*60}")
    print(f"【结论】N={args.n} 严格时效下，真实 Cmax >= {lb:.0f}s")
    if lb >= 6000:
        print(f"  → Cmax < 6000s 在 N={args.n} 下**不可能**（面积约束已证伪）")
    else:
        print(f"  → 不能排除 <6000s，需继续搜排程（松弛解 {val}s 仅为下界）")


if __name__ == "__main__":
    main()
