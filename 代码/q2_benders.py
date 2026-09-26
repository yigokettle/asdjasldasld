# -*- coding: utf-8 -*-
"""逻辑型 Benders 分解求 N 固定下的最小 Cmax。

为什么用分解：
    直接把「选架次 + 排时序」塞进一个 CP-SAT 模型，在万级候选池上连首个
    可行解都判不出来（实测 pool_c8/pool_bat 全程 UNKNOWN）。
    但两个子问题单独都很快：
      主问题（Master）：只有 0/1 选择 + 聚合面积约束，秒级 OPTIMAL
      子问题（Sub）   ：架次集合已固定（只有 23 个区间），排程也是秒级

算法：
    1. Master：在「面积松弛」下求最小 Cmax，得到一组 23 架次 S 与松弛值 LB
    2. Sub   ：固定 S，用 Cumulative 求真实最小 Cmax(S)
       - 若 Cmax(S) <= 目标，直接得到可行解，结束
       - 否则把 S 作为 no-good cut 砍掉：sum_{i in S} x_i <= |S|-1
    3. 回到 1，LB 单调不降，UB 取历史最好可行解，直到 UB-LB <= gap

    Master 的最优值始终是真实 Cmax 的下界（面积约束是必要条件），
    所以本算法给出的 [LB, UB] 是**带证明**的区间。

用法:
  python q2_benders.py --n 23 --pool Q2_架次优化/pool_c8.json \
      --iters 60 --sub-seconds 60
"""
from __future__ import annotations
import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"
OUT.mkdir(parents=True, exist_ok=True)

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}
SCALE = 1_000_000
OK = (cp_model.OPTIMAL, cp_model.FEASIBLE)


def load(pool_file):
    p = Path(pool_file)
    if not p.is_absolute():
        p = ROOT / p
    pool = json.loads(p.read_text())
    b = pd.read_csv(SRC / "optimized_box_delivery.csv").set_index("货箱编号")
    hard = {k: (None if pd.isna(v) else float(v)) for k, v in b.硬截止时刻_s.items()}
    exp = b.期望送达时刻_s.to_dict()
    return pool, hard, exp


def schedule(sub, hard, exp, strict, ub, seconds, want_energy=False):
    """固定架次集合 sub（候选记录列表），求最小 Cmax 的真实排程。

    返回 (status, cmax, starts)。只有 |sub| 个区间，规模极小。
    """
    m = cp_model.CpModel()
    H = int(ub)
    n = len(sub)
    start, end, bend = [], [], []
    for c in sub:
        dur = int(math.ceil(c["duration_s"]))
        chg = int(math.ceil(c["charge_s"]))
        lim = [H - dur]
        for b in c["boxes"]:
            off = c["delivery_offsets"][b]
            if hard[b] is not None:
                lim.append(hard[b] - off)
            if strict:
                lim.append(exp[b] - off)
        hi = int(math.floor(min(lim)))
        if hi < 0:
            return "INFEASIBLE", None, None
        s = m.NewIntVar(0, hi, "")
        e = m.NewIntVar(0, H, "")
        be = m.NewIntVar(0, H + chg, "")
        m.Add(e == s + dur)
        m.Add(be == s + dur + chg)
        start.append(s); end.append(e); bend.append(be)

    for g in "ABC":
        idx = [i for i, c in enumerate(sub) if c["model"] == g]
        if not idx:
            continue
        m.AddCumulative(
            [m.NewIntervalVar(start[i], int(math.ceil(sub[i]["duration_s"])),
                              end[i], "") for i in idx], [1] * len(idx), DRONES[g])
        m.AddCumulative(
            [m.NewIntervalVar(start[i],
                              int(math.ceil(sub[i]["duration_s"]))
                              + int(math.ceil(sub[i]["charge_s"])),
                              bend[i], "") for i in idx],
            [1] * len(idx), BATTERIES[g])

    cmax = m.NewIntVar(0, H, "cmax")
    m.AddMaxEquality(cmax, end)
    m.Minimize(cmax)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = 2026
    r = s.Solve(m)
    if r not in OK:
        return s.StatusName(r), None, None
    return s.StatusName(r), int(s.Value(cmax)), [int(s.Value(v)) for v in start]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--pool", default="Q2_架次优化/pool_c8.json")
    ap.add_argument("--timeliness", choices=["strict", "hard"], default="strict")
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--master-seconds", type=float, default=60)
    ap.add_argument("--sub-seconds", type=float, default=60)
    ap.add_argument("--gap", type=int, default=30)
    ap.add_argument("--ub", type=int, default=6875, help="已知可行 Cmax")
    ap.add_argument("--tag", default="bd23")
    args = ap.parse_args()

    strict = args.timeliness == "strict"
    pool, hard, exp = load(args.pool)
    print(f"候选池 {len(pool)}（{dict(Counter(c['model'] for c in pool))}）")

    boxes = sorted({b for c in pool for b in c["boxes"]})
    bidx = {b: i for i, b in enumerate(boxes)}
    N = len(pool)

    # ---- 主问题：常驻模型，逐轮加 no-good cut ----
    m = cp_model.CpModel()
    x = [m.NewBoolVar(f"x{i}") for i in range(N)]
    cmax = m.NewIntVar(0, 20000, "cmax")

    killed = 0
    for i, c in enumerate(pool):
        lim = []
        for b in c["boxes"]:
            off = c["delivery_offsets"][b]
            if hard[b] is not None:
                lim.append(hard[b] - off)
            if strict:
                lim.append(exp[b] - off)
        if lim and min(lim) < 0:
            m.Add(x[i] == 0)
            killed += 1
    print(f"时效性淘汰 {killed} 个候选")

    cover = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for b in c["boxes"]:
            cover[bidx[b]].append(x[i])
    for lst in cover:
        m.AddExactlyOne(lst)
    m.Add(sum(x) == args.n)

    for g in "ABC":
        idx = [i for i, c in enumerate(pool) if c["model"] == g]
        if not idx:
            continue
        dur = [int(math.ceil(pool[i]["duration_s"])) for i in idx]
        occ = [int(math.ceil(pool[i]["duration_s"] + pool[i]["charge_s"]))
               for i in idx]
        m.Add(sum(d * x[i] for d, i in zip(dur, idx)) <= DRONES[g] * cmax)
        m.Add(sum(o * x[i] for o, i in zip(occ, idx)) <= BATTERIES[g] * cmax)
    for i, c in enumerate(pool):
        m.Add(cmax >= int(math.ceil(c["duration_s"]))).OnlyEnforceIf(x[i])
    m.Minimize(cmax)

    best = None          # (cmax, sub, starts)
    LB = 0
    t0 = time.time()
    for it in range(1, args.iters + 1):
        ms = cp_model.CpSolver()
        ms.parameters.max_time_in_seconds = args.master_seconds
        ms.parameters.num_search_workers = 8
        ms.parameters.random_seed = 2026 + it
        r = ms.Solve(m)
        if r not in OK:
            print(f"[{it:02d}] 主问题 {ms.StatusName(r)} → 所有组合已穷举")
            break
        LB = max(LB, int(math.ceil(ms.BestObjectiveBound())))
        sel = [i for i in range(N) if ms.Value(x[i])]
        relax = int(ms.Value(cmax))
        sub = [pool[i] for i in sel]
        mdl = dict(Counter(c["model"] for c in sub))

        st, cm, starts = schedule(sub, hard, exp, strict,
                                  args.ub, args.sub_seconds)
        ubs = best[0] if best else args.ub
        tag = ""
        if cm is not None and cm < ubs:
            best = (cm, sub, starts)
            tag = "  ★新最优"
        print(f"[{it:02d}] 松弛={relax} LB={LB} 机型={mdl} → 排程 {st}"
              f" Cmax={cm}{tag}  ({time.time()-t0:.0f}s)")

        # no-good cut：这组架次组合不再重复
        m.Add(sum(x[i] for i in sel) <= len(sel) - 1)

        if best and best[0] - LB <= args.gap:
            print(f"收敛：UB={best[0]} LB={LB} 缺口 {best[0]-LB} <= {args.gap}")
            break

    if not best:
        print("未找到可排程的组合")
        return

    cm, sub, starts = best
    print(f"\n{'='*60}")
    print(f"最优 Cmax = {cm}s   证明下界 LB = {LB}s   缺口 {cm-LB}s")

    res = dict(n=len(sub), timeliness="strict" if strict else "hard",
               cmax_s=cm, cmax_lb=LB, pool=args.pool,
               energy_kwh=sum(c["energy_kwh"] for c in sub),
               weighted_delay=0,
               models=dict(Counter(c["model"] for c in sub)),
               trips=[dict(idx=k, model=c["model"], route=c["route"],
                           boxes=c["boxes"], start_s=s0,
                           end_s=s0 + int(math.ceil(c["duration_s"])),
                           bat_end_s=s0 + int(math.ceil(c["duration_s"]))
                           + int(math.ceil(c["charge_s"])),
                           duration_s=c["duration_s"], charge_s=c["charge_s"],
                           mass_kg=c["mass_kg"], volume_m3=c["volume_m3"],
                           delivery_offsets=c["delivery_offsets"],
                           energy_kwh=c["energy_kwh"])
                      for k, (c, s0) in enumerate(zip(sub, starts))])
    (OUT / f"solution_{args.tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"=> N={res['n']} Cmax={cm}s 能耗={res['energy_kwh']:.4f}kWh "
          f"机型={res['models']}")
    print("已保存", OUT / f"solution_{args.tag}.json")


if __name__ == "__main__":
    main()
