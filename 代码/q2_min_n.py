# -*- coding: utf-8 -*-
"""Q2 架次数真实下界探查：带完整时序 + 资源约束，直接最小化 N。

与 q2_opt_n.py 同口径（零延误、硬截止、机队/电池库存、能量上限），
但把 N 作为目标函数而非固定值，用于求出真实最小架次及其下界证书。
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter
from pathlib import Path

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"
OUT.mkdir(parents=True, exist_ok=True)

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}
SCALE = 1_000_000


def build(pool, horizon, n_fixed=None, cmax_ub=None, energy_ub=None):
    boxes = sorted({b for c in pool for b in c["boxes"]})
    bidx = {b: i for i, b in enumerate(boxes)}
    m = cp_model.CpModel()

    x, start, end, bat_end, durs, chgs = [], [], [], [], [], []
    for i, c in enumerate(pool):
        xi = m.NewBoolVar(f"x{i}")
        dur = int(math.ceil(c["duration_s"]))
        chg = int(math.ceil(c["charge_s"]))
        hi = min(int(c["latest_zero_delay_start"]), horizon - dur)
        if hi < 0:
            m.Add(xi == 0)
            hi = 0
        s = m.NewIntVar(0, max(hi, 0), f"s{i}")
        e = m.NewIntVar(0, horizon, f"e{i}")
        be = m.NewIntVar(0, horizon + chg, f"be{i}")
        m.Add(e == s + dur)
        m.Add(be == s + dur + chg)
        x.append(xi); start.append(s); end.append(e); bat_end.append(be)
        durs.append(dur); chgs.append(chg)

    cover = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for b in c["boxes"]:
            cover[bidx[b]].append(x[i])
    for lst in cover:
        m.AddExactlyOne(lst)

    for g in "ABC":
        idx = [i for i, c in enumerate(pool) if c["model"] == g]
        if not idx:
            continue
        m.AddCumulative(
            [m.NewOptionalIntervalVar(start[i], durs[i], end[i], x[i], f"u{i}") for i in idx],
            [1] * len(idx), DRONES[g])
        m.AddCumulative(
            [m.NewOptionalIntervalVar(start[i], durs[i] + chgs[i], bat_end[i], x[i], f"b{i}")
             for i in idx], [1] * len(idx), BATTERIES[g])

    n = m.NewIntVar(0, len(pool), "n")
    m.Add(n == sum(x))
    if n_fixed is not None:
        m.Add(n == n_fixed)

    cmax = m.NewIntVar(0, horizon, "cmax")
    for i in range(len(pool)):
        m.Add(cmax >= end[i]).OnlyEnforceIf(x[i])
    if cmax_ub is not None:
        m.Add(cmax <= cmax_ub)

    ecoef = [int(round(c["energy_kwh"] * SCALE)) for c in pool]
    energy = m.NewIntVar(0, sum(ecoef), "energy")
    m.Add(energy == sum(ecoef[i] * x[i] for i in range(len(pool))))
    if energy_ub is not None:
        m.Add(energy <= energy_ub)

    return m, dict(x=x, start=start, end=end, bat_end=bat_end,
                   n=n, cmax=cmax, energy=energy)


def run(m, obj, seconds, seed=2026):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = seed
    m.Minimize(obj)
    return s, s.Solve(m)


def dump(pool, sv, v, tag):
    sel = [i for i in range(len(pool)) if sv.Value(v["x"][i])]
    res = dict(
        n=len(sel),
        cmax_s=max(sv.Value(v["end"][i]) for i in sel),
        energy_kwh=sum(pool[i]["energy_kwh"] for i in sel),
        models=dict(Counter(pool[i]["model"] for i in sel)),
        trips=[dict(idx=i, model=pool[i]["model"], route=pool[i]["route"],
                    boxes=pool[i]["boxes"],
                    start_s=sv.Value(v["start"][i]), end_s=sv.Value(v["end"][i]),
                    bat_end_s=sv.Value(v["bat_end"][i]),
                    energy_kwh=pool[i]["energy_kwh"]) for i in sel])
    (OUT / f"solution_{tag}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--horizon", type=int, default=20000)
    args = ap.parse_args()

    pool = json.loads((SRC / "optimization_pool.json").read_text())
    print(f"候选池 {len(pool)} 个 —— 带时序+资源约束，直接最小化架次数 N")

    m, v = build(pool, args.horizon)
    s, r = run(m, v["n"], args.seconds)
    print(f"[minN] 状态={s.StatusName(r)}")
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("未找到可行解")
        return
    nbest = int(s.ObjectiveValue())
    print(f"  最小架次 N* = {nbest}   下界 = {s.BestObjectiveBound():.0f}   "
          f"用时 {s.WallTime():.1f}s")
    res = dump(pool, s, v, f"minN{nbest}")
    print(f"  该解: Cmax={res['cmax_s']}s  能耗={res['energy_kwh']:.4f}kWh  "
          f"机型={res['models']}")


if __name__ == "__main__":
    main()
