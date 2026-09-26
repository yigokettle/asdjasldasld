# -*- coding: utf-8 -*-
"""Q2 固定架次数下的及时性/完工时间优化（按题目原口径）。

题目口径修正（关键）：
  * 硬约束：医疗物资期望送达时间 + 首批保障货箱首批截止时间（共 31 箱）
  * 软指标：其他 49 箱的期望送达时间「用于衡量配送及时性」，不是硬约束
  原继承模型把全部 80 箱都设为零延误硬约束，过严，压缩了 Cmax 的可行空间。

优化顺序（固定 N）：
  阶段1  硬约束可行前提下最小化 Cmax
  阶段2  固定 Cmax，最小化加权延误（及时性）
  阶段3  固定 Cmax 与加权延误，最小化总能耗
用法: python q2_timely.py --n 23 --seconds 300
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
OUT.mkdir(parents=True, exist_ok=True)

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}
SCALE = 1_000_000


def load(pool_file=None):
    p = Path(pool_file) if pool_file else (SRC / "optimization_pool.json")
    if not p.is_absolute():
        p = ROOT / p
    pool = json.loads(p.read_text())
    b = pd.read_csv(SRC / "optimized_box_delivery.csv").set_index("货箱编号")
    hard = {k: (None if pd.isna(v) else float(v)) for k, v in b.硬截止时刻_s.items()}
    exp = b.期望送达时刻_s.to_dict()
    pri = b.应急优先系数.to_dict()
    return pool, hard, exp, pri


def build(pool, hard, exp, pri, horizon, n_fixed,
          cmax_ub=None, delay_ub=None, energy_ub=None):
    boxes = sorted({b for c in pool for b in c["boxes"]})
    bidx = {b: i for i, b in enumerate(boxes)}
    m = cp_model.CpModel()

    x, start, end, bat_end, durs, chgs, shi = [], [], [], [], [], [], []
    for i, c in enumerate(pool):
        xi = m.NewBoolVar(f"x{i}")
        dur = int(math.ceil(c["duration_s"]))
        chg = int(math.ceil(c["charge_s"]))
        # 仅硬截止箱限制最晚启动（题目口径）
        hs = [hard[b] - c["delivery_offsets"][b]
              for b in c["boxes"] if hard[b] is not None]
        hi = int(math.floor(min(hs))) if hs else horizon - dur
        hi = min(hi, horizon - dur)
        if hi < 0:
            m.Add(xi == 0)
            hi = 0
        s = m.NewIntVar(0, max(hi, 0), f"s{i}")
        e = m.NewIntVar(0, horizon, f"e{i}")
        be = m.NewIntVar(0, horizon + chg, f"be{i}")
        m.Add(e == s + dur)
        m.Add(be == s + dur + chg)
        x.append(xi); start.append(s); end.append(e); bat_end.append(be)
        durs.append(dur); chgs.append(chg); shi.append(max(hi, 0))

    cover = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for b in c["boxes"]:
            cover[bidx[b]].append(x[i])
    for lst in cover:
        m.AddExactlyOne(lst)

    m.Add(sum(x) == n_fixed)

    for g in "ABC":
        idx = [i for i, c in enumerate(pool) if c["model"] == g]
        if not idx:
            continue
        m.AddCumulative(
            [m.NewOptionalIntervalVar(start[i], durs[i], end[i], x[i], f"u{i}")
             for i in idx], [1] * len(idx), DRONES[g])
        m.AddCumulative(
            [m.NewOptionalIntervalVar(start[i], durs[i] + chgs[i], bat_end[i],
                                      x[i], f"b{i}") for i in idx],
            [1] * len(idx), BATTERIES[g])

    cmax = m.NewIntVar(0, horizon, "cmax")
    for i in range(len(pool)):
        m.Add(cmax >= end[i]).OnlyEnforceIf(x[i])
    if cmax_ub is not None:
        m.Add(cmax <= cmax_ub)

    # 加权延误（软指标）：对每个候选，若被选则其各箱延误计入
    terms = []
    for i, c in enumerate(pool):
        for b in c["boxes"]:
            off = c["delivery_offsets"][b]
            shift = int(math.ceil(off - exp[b]))     # 延误 = start + shift
            hi = shi[i] + shift                      # start 取上界时的延误
            if hi <= 0:
                continue                             # 该候选此箱恒不延误
            dv = m.NewIntVar(0, hi, f"d{i}_{b}")
            m.Add(dv >= start[i] + shift).OnlyEnforceIf(x[i])
            m.Add(dv == 0).OnlyEnforceIf(x[i].Not())
            terms.append(int(pri[b]) * dv)
    delay = m.NewIntVar(0, 10 ** 9, "delay")
    m.Add(delay == sum(terms) if terms else 0)
    if delay_ub is not None:
        m.Add(delay <= delay_ub)

    ecoef = [int(round(c["energy_kwh"] * SCALE)) for c in pool]
    energy = m.NewIntVar(0, sum(ecoef), "energy")
    m.Add(energy == sum(ecoef[i] * x[i] for i in range(len(pool))))
    if energy_ub is not None:
        m.Add(energy <= energy_ub)

    return m, dict(x=x, start=start, end=end, bat_end=bat_end,
                   cmax=cmax, delay=delay, energy=energy)


def run(m, obj, seconds, seed=2026):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = seed
    m.Minimize(obj)
    return s, s.Solve(m)


OK = (cp_model.OPTIMAL, cp_model.FEASIBLE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--horizon", type=int, default=20000)
    args = ap.parse_args()

    pool, hard, exp, pri = load()
    nh = sum(1 for v in hard.values() if v is not None)
    print(f"候选池 {len(pool)}；硬约束箱 {nh}，软指标箱 {len(hard)-nh}")
    print(f"固定 N={args.n}，优化顺序：Cmax → 加权延误 → 能耗\n")

    # 阶段1：最小 Cmax
    m1, v1 = build(pool, hard, exp, pri, args.horizon, args.n)
    s1, r1 = run(m1, v1["cmax"], args.seconds)
    print(f"[阶段1 Cmax ] 状态={s1.StatusName(r1)}", end=" ")
    if r1 not in OK:
        print("\n无可行解")
        return
    cmax1 = int(s1.Value(v1["cmax"]))
    print(f"Cmax={cmax1}s  下界={s1.BestObjectiveBound():.0f}  用时={s1.WallTime():.1f}s")

    # 阶段2：固定 Cmax，最小加权延误
    m2, v2 = build(pool, hard, exp, pri, args.horizon, args.n, cmax_ub=cmax1)
    s2, r2 = run(m2, v2["delay"], args.seconds)
    print(f"[阶段2 延误 ] 状态={s2.StatusName(r2)}", end=" ")
    if r2 in OK:
        d2 = int(s2.Value(v2["delay"]))
        print(f"加权延误={d2}  下界={s2.BestObjectiveBound():.0f}  用时={s2.WallTime():.1f}s")
    else:
        print("失败，跳过")
        d2 = None

    # 阶段3：固定 Cmax 与延误，最小能耗
    sv, vv = s2 if r2 in OK else s1, v2 if r2 in OK else v1
    if d2 is not None:
        m3, v3 = build(pool, hard, exp, pri, args.horizon, args.n,
                       cmax_ub=cmax1, delay_ub=d2)
        s3, r3 = run(m3, v3["energy"], args.seconds)
        print(f"[阶段3 能耗 ] 状态={s3.StatusName(r3)}", end=" ")
        if r3 in OK:
            print(f"E={s3.Value(v3['energy'])/SCALE:.4f}kWh  "
                  f"下界={s3.BestObjectiveBound()/SCALE:.4f}  用时={s3.WallTime():.1f}s")
            sv, vv = s3, v3
        else:
            print("失败，沿用阶段2解")

    sel = [i for i in range(len(pool)) if sv.Value(vv["x"][i])]
    res = dict(n=len(sel),
               cmax_s=max(sv.Value(vv["end"][i]) for i in sel),
               energy_kwh=sum(pool[i]["energy_kwh"] for i in sel),
               weighted_delay=int(sv.Value(vv["delay"])),
               models=dict(Counter(pool[i]["model"] for i in sel)),
               trips=[dict(idx=i, model=pool[i]["model"], route=pool[i]["route"],
                           boxes=pool[i]["boxes"],
                           start_s=sv.Value(vv["start"][i]),
                           end_s=sv.Value(vv["end"][i]),
                           bat_end_s=sv.Value(vv["bat_end"][i]),
                           energy_kwh=pool[i]["energy_kwh"]) for i in sel])
    tag = f"tN{args.n}"
    (OUT / f"solution_{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> N={res['n']}  Cmax={res['cmax_s']}s  "
          f"加权延误={res['weighted_delay']}  能耗={res['energy_kwh']:.4f}kWh  "
          f"机型={res['models']}")
    print("已保存", OUT / f"solution_{tag}.json")


if __name__ == "__main__":
    main()
