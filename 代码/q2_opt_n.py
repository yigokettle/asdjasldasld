# -*- coding: utf-8 -*-
"""Q2 架次数优化：在保持现有推荐解全部约束的前提下，把架次数从 25 压到 23。

口径与 Q2_开源对比与优化/ 推荐解完全一致：
  * 候选池 optimization_pool.json（572 个候选，已含逐段物理量与两阶段充电时长）
  * 每箱恰好交付一次
  * 零延误：被选候选逐箱满足 start + delivery_offset <= min(期望时刻, 硬截止)
    —— 候选字段 latest_zero_delay_start 已预计算该上界
  * 机型实体机库存：A=4, B=2, C=2（题目 U01-U08）
  * 共享电池库存：A=6, B=4, C=4（题目给定）
    电池占用区间 = [start, start + duration + charge]（执行 + 充电）
    无人机占用区间 = [start, start + duration]
  * 架次能耗 <= (1-20%) * E_use（候选池已按 energy_limit_kwh 过滤）

优化策略（字典序）：
  阶段1  固定 N = 23，求可行 + 最小 Cmax
  阶段2  固定 N = 23、Cmax <= 阶段1结果，最小化总能耗
资源用累积约束（AddCumulative）建模，求解后按区间图着色分配具体机号/电池号。
"""
from __future__ import annotations
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from ortools.sat.python import cp_model

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"
OUT.mkdir(parents=True, exist_ok=True)

DRONES = {"A": 4, "B": 2, "C": 2}        # 题目逐架清单 U01-U08
BATTERIES = {"A": 6, "B": 4, "C": 4}     # 题目共享电池库存
SCALE = 1_000_000                        # 能耗整数化：1e-6 kWh


def build(pool, n_target, horizon, cmax_ub=None, energy_ub=None):
    """构建 CP-SAT 模型。返回 (model, vars)。"""
    boxes = sorted({b for c in pool for b in c["boxes"]})
    bidx = {b: i for i, b in enumerate(boxes)}
    m = cp_model.CpModel()

    x, start, end, bat_end = [], [], [], []
    durs, chgs = [], []
    for i, c in enumerate(pool):
        xi = m.NewBoolVar(f"x{i}")
        dur = int(math.ceil(c["duration_s"]))          # 保守取整
        chg = int(math.ceil(c["charge_s"]))
        # 零延误上界：候选池预计算字段
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

    # 每箱恰好一次
    cover = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for b in c["boxes"]:
            cover[bidx[b]].append(x[i])
    for lst in cover:
        m.AddExactlyOne(lst)

    # 架次数
    m.Add(sum(x) == n_target)

    # 资源：按机型分别做累积约束
    for g in "ABC":
        idx = [i for i, c in enumerate(pool) if c["model"] == g]
        if not idx:
            continue
        # 无人机占用 [start, end)
        ivs = [m.NewOptionalIntervalVar(start[i], durs[i], end[i],
                                        x[i], f"uav{i}") for i in idx]
        m.AddCumulative(ivs, [1] * len(idx), DRONES[g])
        # 电池占用 [start, bat_end)  执行 + 充电
        bivs = [m.NewOptionalIntervalVar(start[i], durs[i] + chgs[i], bat_end[i],
                                         x[i], f"bat{i}") for i in idx]
        m.AddCumulative(bivs, [1] * len(idx), BATTERIES[g])

    # Cmax
    cmax = m.NewIntVar(0, horizon, "cmax")
    for i in range(len(pool)):
        m.Add(cmax >= end[i]).OnlyEnforceIf(x[i])
    if cmax_ub is not None:
        m.Add(cmax <= cmax_ub)

    # 能耗
    ecoef = [int(round(c["energy_kwh"] * SCALE)) for c in pool]
    energy = m.NewIntVar(0, sum(ecoef), "energy")
    m.Add(energy == sum(ecoef[i] * x[i] for i in range(len(pool))))
    if energy_ub is not None:
        m.Add(energy <= energy_ub)

    return m, dict(x=x, start=start, end=end, bat_end=bat_end,
                   cmax=cmax, energy=energy, boxes=boxes)


def solve(m, obj, seconds, seed=2026, log=False):
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = seed
    s.parameters.log_search_progress = log
    m.Minimize(obj)
    r = s.Solve(m)
    return s, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--horizon", type=int, default=20000)
    args = ap.parse_args()

    pool = json.loads((SRC / "optimization_pool.json").read_text())
    print(f"候选池 {len(pool)} 个；目标架次 N={args.n}")

    # ---------- 阶段1：可行性 + 最小 Cmax ----------
    m, v = build(pool, args.n, args.horizon)
    s, r = solve(m, v["cmax"], args.seconds)
    print(f"[阶段1 Cmax] 状态={s.StatusName(r)}", end=" ")
    if r not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n=> 当前候选池下 N={} 无可行解（或限时内未找到）".format(args.n))
        return
    cmax1 = int(s.Value(v["cmax"]))
    print(f"Cmax={cmax1}s 下界={s.BestObjectiveBound():.0f} 用时={s.WallTime():.1f}s")

    # ---------- 阶段2：固定 Cmax 上界，最小化能耗 ----------
    m2, v2 = build(pool, args.n, args.horizon, cmax_ub=cmax1)
    s2, r2 = solve(m2, v2["energy"], args.seconds)
    print(f"[阶段2 能耗] 状态={s2.StatusName(r2)}", end=" ")
    if r2 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n阶段2失败，沿用阶段1解")
        s2, v2, m2 = s, v, m
        r2 = r
    else:
        print(f"E={s2.Value(v2['energy'])/SCALE:.4f}kWh "
              f"下界={s2.BestObjectiveBound()/SCALE:.4f} 用时={s2.WallTime():.1f}s")

    sel = [i for i in range(len(pool)) if s2.Value(v2["x"][i])]
    res = dict(
        n=len(sel),
        cmax_s=max(s2.Value(v2["end"][i]) for i in sel),
        energy_kwh=sum(pool[i]["energy_kwh"] for i in sel),
        models=dict(Counter(pool[i]["model"] for i in sel)),
        stage1=dict(status=s.StatusName(r), cmax=cmax1,
                    bound=s.BestObjectiveBound(), wall_s=s.WallTime()),
        stage2=dict(status=s2.StatusName(r2),
                    energy=s2.Value(v2["energy"]) / SCALE,
                    bound=s2.BestObjectiveBound() / SCALE, wall_s=s2.WallTime()),
        trips=[dict(idx=i, model=pool[i]["model"], route=pool[i]["route"],
                    boxes=pool[i]["boxes"],
                    start_s=s2.Value(v2["start"][i]),
                    end_s=s2.Value(v2["end"][i]),
                    bat_end_s=s2.Value(v2["bat_end"][i]),
                    energy_kwh=pool[i]["energy_kwh"]) for i in sel],
    )
    (OUT / f"solution_cN{args.n}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> N={res['n']}  Cmax={res['cmax_s']}s  "
          f"能耗={res['energy_kwh']:.4f}kWh  机型={res['models']}")
    print("已保存", OUT / f"solution_cN{args.n}.json")


if __name__ == "__main__":
    main()
