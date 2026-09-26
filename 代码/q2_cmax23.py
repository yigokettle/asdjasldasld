# -*- coding: utf-8 -*-
"""固定架次数 N 下最小化全部任务完成时间 Cmax，并保证配送及时性。

及时性口径（两档，--timeliness 选择）：
  strict  全部 80 箱均不晚于期望送达时间（与原基准 25 架次解同档，最严）
  hard    仅医疗 + 首批保障 31 箱为硬约束，其余 49 箱延误计入加权延误指标

优化顺序（字典序）：
  阶段1  最小化 Cmax
  阶段2  固定 Cmax，最小化加权延误（strict 档恒为 0，自动跳过）
  阶段3  固定前两项，最小化总能耗

资源模型：
  无人机  占用 [start, start+duration)，按机型 AddCumulative(容量=机数)
  电池    占用 [start, start+duration+charge)，按机型 AddCumulative(容量=电池数)

用法:
  python q2_cmax23.py --n 23 --pool Q2_架次优化/pool_pruned.json --seconds 600
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
OK = (cp_model.OPTIMAL, cp_model.FEASIBLE)


def load(pool_file):
    p = Path(pool_file)
    if not p.is_absolute():
        p = ROOT / p
    pool = json.loads(p.read_text())
    b = pd.read_csv(SRC / "optimized_box_delivery.csv").set_index("货箱编号")
    hard = {k: (None if pd.isna(v) else float(v)) for k, v in b.硬截止时刻_s.items()}
    exp = b.期望送达时刻_s.to_dict()
    pri = b.应急优先系数.to_dict()
    return pool, hard, exp, pri


def build(pool, hard, exp, pri, horizon, n_fixed, strict,
          cmax_ub=None, delay_ub=None, energy_ub=None):
    boxes = sorted({b for c in pool for b in c["boxes"]})
    bidx = {b: i for i, b in enumerate(boxes)}
    m = cp_model.CpModel()

    x, start, end, bat_end, durs, chgs, shi = [], [], [], [], [], [], []
    for i, c in enumerate(pool):
        xi = m.NewBoolVar(f"x{i}")
        dur = int(math.ceil(c["duration_s"]))
        chg = int(math.ceil(c["charge_s"]))
        # 最晚启动时刻：strict 档受全部箱期望时刻约束，hard 档仅受硬截止约束
        lim = []
        for b in c["boxes"]:
            off = c["delivery_offsets"][b]
            if hard[b] is not None:
                lim.append(hard[b] - off)
            if strict:
                lim.append(exp[b] - off)
        hi = int(math.floor(min(lim))) if lim else horizon - dur
        hi = min(hi, horizon - dur)
        if hi < 0:
            m.Add(xi == 0)
            hi = 0
        s = m.NewIntVar(0, hi, f"s{i}")
        e = m.NewIntVar(0, horizon, f"e{i}")
        be = m.NewIntVar(0, horizon + chg, f"be{i}")
        m.Add(e == s + dur)
        m.Add(be == s + dur + chg)
        x.append(xi); start.append(s); end.append(e); bat_end.append(be)
        durs.append(dur); chgs.append(chg); shi.append(hi)

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

    terms = []
    if not strict:
        for i, c in enumerate(pool):
            for b in c["boxes"]:
                shift = int(math.ceil(c["delivery_offsets"][b] - exp[b]))
                hi = shi[i] + shift
                if hi <= 0:
                    continue
                dv = m.NewIntVar(0, hi, f"d{i}_{b}")
                m.Add(dv >= start[i] + shift).OnlyEnforceIf(x[i])
                m.Add(dv == 0).OnlyEnforceIf(x[i].Not())
                terms.append(int(pri[b]) * dv)
    delay = m.NewIntVar(0, 10 ** 9, "delay")
    m.Add(delay == sum(terms) if terms else delay == 0)
    if delay_ub is not None:
        m.Add(delay <= delay_ub)

    ec = [int(round(c["energy_kwh"] * SCALE)) for c in pool]
    energy = m.NewIntVar(0, sum(ec), "energy")
    m.Add(energy == sum(ec[i] * x[i] for i in range(len(pool))))
    if energy_ub is not None:
        m.Add(energy <= energy_ub)

    return m, dict(x=x, start=start, end=end, bat_end=bat_end,
                   cmax=cmax, delay=delay, energy=energy)


def run(m, obj, seconds, hint=None, seed=2026):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = seed
    m.Minimize(obj)
    return s, s.Solve(m)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--pool", default="Q2_架次优化/pool_pruned.json")
    ap.add_argument("--timeliness", choices=["strict", "hard"], default="strict")
    ap.add_argument("--seconds", type=float, default=600)
    ap.add_argument("--horizon", type=int, default=20000)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    strict = args.timeliness == "strict"
    pool, hard, exp, pri = load(args.pool)
    nh = sum(1 for v in hard.values() if v is not None)
    print(f"候选池 {len(pool)}（{dict(Counter(c['model'] for c in pool))}）")
    print(f"硬约束箱 {nh}，及时性档位 = {args.timeliness}"
          f"（{'全部 80 箱零延误' if strict else f'仅 {nh} 箱硬约束'}）")
    print(f"固定 N={args.n}，优化顺序 Cmax → 加权延误 → 能耗\n")

    m1, v1 = build(pool, hard, exp, pri, args.horizon, args.n, strict)
    s1, r1 = run(m1, v1["cmax"], args.seconds)
    print(f"[阶段1 Cmax ] {s1.StatusName(r1)}", end=" ")
    if r1 not in OK:
        print("\n无可行解")
        return
    cmax1 = int(s1.Value(v1["cmax"]))
    print(f"Cmax={cmax1}s 下界={s1.BestObjectiveBound():.0f} 用时={s1.WallTime():.1f}s")

    sv, vv, d2 = s1, v1, 0
    if not strict:
        m2, v2 = build(pool, hard, exp, pri, args.horizon, args.n, strict,
                       cmax_ub=cmax1)
        s2, r2 = run(m2, v2["delay"], args.seconds)
        print(f"[阶段2 延误 ] {s2.StatusName(r2)}", end=" ")
        if r2 in OK:
            d2 = int(s2.Value(v2["delay"]))
            print(f"加权延误={d2} 下界={s2.BestObjectiveBound():.0f} "
                  f"用时={s2.WallTime():.1f}s")
            sv, vv = s2, v2
        else:
            print("失败，跳过"); d2 = None

    if d2 is not None:
        m3, v3 = build(pool, hard, exp, pri, args.horizon, args.n, strict,
                       cmax_ub=cmax1, delay_ub=d2)
        s3, r3 = run(m3, v3["energy"], args.seconds)
        print(f"[阶段3 能耗 ] {s3.StatusName(r3)}", end=" ")
        if r3 in OK:
            print(f"E={s3.Value(v3['energy'])/SCALE:.4f}kWh "
                  f"下界={s3.BestObjectiveBound()/SCALE:.4f} 用时={s3.WallTime():.1f}s")
            sv, vv = s3, v3
        else:
            print("失败，沿用上一阶段解")

    sel = [i for i in range(len(pool)) if sv.Value(vv["x"][i])]
    res = dict(n=len(sel), pool=args.pool, timeliness=args.timeliness,
               cmax_s=max(sv.Value(vv["end"][i]) for i in sel),
               energy_kwh=sum(pool[i]["energy_kwh"] for i in sel),
               weighted_delay=int(sv.Value(vv["delay"])),
               models=dict(Counter(pool[i]["model"] for i in sel)),
               trips=[dict(idx=i, model=pool[i]["model"], route=pool[i]["route"],
                           boxes=pool[i]["boxes"],
                           start_s=sv.Value(vv["start"][i]),
                           end_s=sv.Value(vv["end"][i]),
                           bat_end_s=sv.Value(vv["bat_end"][i]),
                           duration_s=pool[i]["duration_s"],
                           charge_s=pool[i]["charge_s"],
                           mass_kg=pool[i]["mass_kg"],
                           volume_m3=pool[i]["volume_m3"],
                           delivery_offsets=pool[i]["delivery_offsets"],
                           energy_kwh=pool[i]["energy_kwh"]) for i in sel])
    tag = args.tag or f"x{args.timeliness[0]}N{args.n}"
    (OUT / f"solution_{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> N={res['n']} Cmax={res['cmax_s']}s "
          f"延误={res['weighted_delay']} 能耗={res['energy_kwh']:.4f}kWh "
          f"机型={res['models']}")
    print("已保存", OUT / f"solution_{tag}.json")


if __name__ == "__main__":
    main()
