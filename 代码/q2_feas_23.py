# -*- coding: utf-8 -*-
"""Q2 架次数下界探查：仅集合划分（每箱恰好一次），不含时序与资源。

用于快速判断 23 架次在当前 572 候选池中是否存在"纯组批可行"解；
若此松弛问题都无解，则加上时序/资源后必然无解。
"""
from __future__ import annotations
import json
from pathlib import Path
from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
POOL = ROOT / "Q2_开源对比与优化" / "optimization_pool.json"

pool = json.loads(POOL.read_text())
boxes = sorted({b for c in pool for b in c["boxes"]})
bidx = {b: i for i, b in enumerate(boxes)}

m = cp_model.CpModel()
x = [m.NewBoolVar(f"x{i}") for i in range(len(pool))]
cover = [[] for _ in boxes]
for i, c in enumerate(pool):
    for b in c["boxes"]:
        cover[bidx[b]].append(x[i])
for i, lst in enumerate(cover):
    m.AddExactlyOne(lst)

n = sum(x)
m.Minimize(n)

s = cp_model.CpSolver()
s.parameters.max_time_in_seconds = 120
s.parameters.num_search_workers = 8
s.parameters.random_seed = 2026
r = s.Solve(m)
print("状态:", s.StatusName(r))
if r in (cp_model.OPTIMAL, cp_model.FEASIBLE):
    print("最小架次(纯集合划分,无时序/资源) =", int(s.ObjectiveValue()))
    print("下界 =", s.BestObjectiveBound())
    sel = [i for i in range(len(pool)) if s.Value(x[i])]
    import collections
    print("机型分布:", collections.Counter(pool[i]["model"] for i in sel))
    print("总能耗 =", round(sum(pool[i]["energy_kwh"] for i in sel), 4), "kWh")
