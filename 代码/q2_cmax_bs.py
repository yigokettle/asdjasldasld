# -*- coding: utf-8 -*-
"""固定 N 下对 Cmax 做二分可行性搜索（比直接 Minimize(cmax) 收敛快得多）。

思路：CP-SAT 在大候选池上「证明最优」很慢，但「判定给定 Cmax 上界是否可行」
      通常快得多。对 Cmax 做二分，每步只问可行性（Solve 到首个可行解即停）。

理论下界（机队配平）：
    设机型 g 的单箱最小边际时长 m_g、机数 K_g，箱数 n_g，
    则 Cmax >= max_g (n_g*m_g/K_g)，在 sum n_g = 80 下配平得
    T* = 80 / sum_g (K_g/m_g)   —— 作为二分下界的参考

用法:
  python q2_cmax_bs.py --n 23 --pool Q2_架次优化/pool_c8.json \
      --lo 3400 --hi 6991 --step-seconds 240
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


def feas(pool, hard, exp, pri, n_fixed, cmax_ub, strict, seconds,
         energy_ub=None, want_energy=False, hint=None):
    """判定 Cmax <= cmax_ub 是否可行；返回 (status, solver, vars) 。

    hint: {(model, boxes_tuple, route_tuple): start_s} —— 已知可行解，作为
          CP-SAT 的 AddHint 暖启动，显著提升大池上找到首个可行解的概率。
    """
    boxes = sorted({b for c in pool for b in c["boxes"]})
    bidx = {b: i for i, b in enumerate(boxes)}
    m = cp_model.CpModel()
    H = cmax_ub

    x, start, end, bat_end = [], [], [], []
    live = []
    for i, c in enumerate(pool):
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
        xi = m.NewBoolVar(f"x{i}")
        if hi < 0:                              # 该候选在此 Cmax 下不可用
            m.Add(xi == 0)
            hi = 0
        else:
            live.append(i)
        s = m.NewIntVar(0, hi, f"s{i}")
        e = m.NewIntVar(0, H, f"e{i}")
        be = m.NewIntVar(0, H + chg, f"be{i}")
        m.Add(e == s + dur)
        m.Add(be == s + dur + chg)
        x.append(xi); start.append(s); end.append(e); bat_end.append(be)

    cover = [[] for _ in boxes]
    for i, c in enumerate(pool):
        for b in c["boxes"]:
            cover[bidx[b]].append(x[i])
    for k, lst in enumerate(cover):
        if not lst:
            return "NO_COVER", None, None
        m.AddExactlyOne(lst)
    m.Add(sum(x) == n_fixed)

    for g in "ABC":
        idx = [i for i, c in enumerate(pool) if c["model"] == g]
        if not idx:
            continue
        m.AddCumulative(
            [m.NewOptionalIntervalVar(start[i], int(math.ceil(pool[i]["duration_s"])),
                                      end[i], x[i], f"u{i}") for i in idx],
            [1] * len(idx), DRONES[g])
        m.AddCumulative(
            [m.NewOptionalIntervalVar(
                start[i], int(math.ceil(pool[i]["duration_s"]))
                + int(math.ceil(pool[i]["charge_s"])), bat_end[i], x[i], f"b{i}")
             for i in idx], [1] * len(idx), BATTERIES[g])

    ec = [int(round(c["energy_kwh"] * SCALE)) for c in pool]
    energy = m.NewIntVar(0, sum(ec), "energy")
    m.Add(energy == sum(ec[i] * x[i] for i in range(len(pool))))
    if energy_ub is not None:
        m.Add(energy <= energy_ub)

    cmax = m.NewIntVar(0, H, "cmax")
    for i in range(len(pool)):
        m.Add(cmax >= end[i]).OnlyEnforceIf(x[i])

    # ---- 暖启动：把已知可行解注入为 hint ----
    if hint:
        nh = 0
        for i, c in enumerate(pool):
            k = (c["model"], tuple(sorted(c["boxes"])), tuple(c["route"]))
            if k in hint:
                st0 = int(hint[k])
                if st0 <= H - int(math.ceil(c["duration_s"])):
                    m.AddHint(x[i], 1)
                    m.AddHint(start[i], st0)
                    nh += 1
            else:
                m.AddHint(x[i], 0)
        if nh:
            print(f"      [hint] 命中 {nh} 个候选", flush=True)

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = seconds
    s.parameters.num_search_workers = 8
    s.parameters.random_seed = 2026
    if want_energy:
        m.Minimize(energy)
    r = s.Solve(m)
    return s.StatusName(r), s, dict(x=x, start=start, end=end,
                                    bat_end=bat_end, cmax=cmax, energy=energy)


def dump(pool, sv, vv, tag, strict, extra=None):
    sel = [i for i in range(len(pool)) if sv.Value(vv["x"][i])]
    res = dict(n=len(sel), timeliness="strict" if strict else "hard",
               cmax_s=max(sv.Value(vv["end"][i]) for i in sel),
               energy_kwh=sum(pool[i]["energy_kwh"] for i in sel),
               weighted_delay=0,
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
    if extra:
        res |= extra
    (OUT / f"solution_{tag}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--pool", default="Q2_架次优化/pool_c8.json")
    ap.add_argument("--timeliness", choices=["strict", "hard"], default="strict")
    ap.add_argument("--lo", type=int, default=None, help="二分下界，默认用配平下界")
    ap.add_argument("--hi", type=int, default=6991, help="二分上界（已知可行值）")
    ap.add_argument("--step-seconds", type=float, default=240)
    ap.add_argument("--gap", type=int, default=60, help="二分终止精度(s)")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--hint-from", default="solution_lexN23.json",
                    help="用于暖启动的已知可行解文件名（在 Q2_架次优化 下）")
    args = ap.parse_args()

    strict = args.timeliness == "strict"
    pool, hard, exp, pri = load(args.pool)
    print(f"候选池 {len(pool)}（{dict(Counter(c['model'] for c in pool))}）")

    # 暖启动解：兼容两种格式（trips 自带 boxes/route，或仅有原池 idx）
    hint = None
    hf = OUT / args.hint_from
    if hf.exists():
        hs = json.loads(hf.read_text())
        opool = None
        hint = {}
        for t in hs["trips"]:
            if "boxes" in t and "route" in t:
                mdl, bxs, rt = t["model"], t["boxes"], t["route"]
            else:
                if opool is None:
                    opool = json.loads((SRC / "optimization_pool.json").read_text())
                c = opool[t["idx"]]
                mdl, bxs, rt = c["model"], c["boxes"], c["route"]
            hint[(mdl, tuple(sorted(bxs)), tuple(rt))] = t["start_s"]
        print(f"暖启动来源 {hf.name}：{len(hint)} 架次，Cmax={hs['cmax_s']}s")

    # 配平下界
    mg = {}
    for c in pool:
        g = c["model"]
        per = c["duration_s"] / len(c["boxes"])
        if g not in mg or per < mg[g]:
            mg[g] = per
    denom = sum(DRONES[g] / mg[g] for g in mg)
    tstar = 80 / denom
    print(f"单箱边际时长 {  {k: round(v,1) for k,v in sorted(mg.items())} }")
    print(f"机队配平下界 T* ≈ {tstar:.0f} s")

    lo = args.lo if args.lo is not None else max(int(tstar), 1)
    hi = args.hi
    print(f"二分区间 [{lo}, {hi}]，每步 {args.step_seconds}s，精度 {args.gap}s\n")

    best = None
    # 先确认上界可行（拿到基线解）
    st, sv, vv = feas(pool, hard, exp, pri, args.n, hi, strict,
                      args.step_seconds, hint=hint)
    print(f"  Cmax<={hi:5d} → {st}")
    if st not in ("OPTIMAL", "FEASIBLE"):
        print("上界都不可行，放宽 --hi 或检查池")
        return
    best = (int(sv.Value(vv["cmax"])), sv, vv)
    hi = best[0]
    print(f"      实际 Cmax={hi}")

    while hi - lo > args.gap:
        mid = (lo + hi) // 2
        st, sv, vv = feas(pool, hard, exp, pri, args.n, mid, strict,
                          args.step_seconds, hint=hint)
        print(f"  Cmax<={mid:5d} → {st}", flush=True)
        if st in ("OPTIMAL", "FEASIBLE"):
            hi = int(sv.Value(vv["cmax"]))
            best = (hi, sv, vv)
            print(f"      实际 Cmax={hi}")
        elif st == "INFEASIBLE":
            lo = mid + 1
        else:                                   # UNKNOWN：判定不了，当作偏紧
            lo = mid + 1
            print("      (UNKNOWN，按不可行处理，可加大 --step-seconds)")

    cm, sv, vv = best
    print(f"\n最优 Cmax ≈ {cm}s（下界 {lo}s，缺口 ≤ {cm-lo}s）")
    # 固定 Cmax 后压能耗
    st, sv2, vv2 = feas(pool, hard, exp, pri, args.n, cm, strict,
                        args.step_seconds * 2, want_energy=True, hint=hint)
    if st in ("OPTIMAL", "FEASIBLE"):
        print(f"固定 Cmax={cm} 压能耗 → {st}  "
              f"E={sv2.Value(vv2['energy'])/SCALE:.4f}kWh")
        sv, vv = sv2, vv2
    tag = args.tag or f"bs{args.timeliness[0]}N{args.n}"
    res = dump(pool, sv, vv, tag, strict, extra=dict(cmax_lb=lo, pool=args.pool))
    print(f"\n=> N={res['n']} Cmax={res['cmax_s']}s "
          f"能耗={res['energy_kwh']:.4f}kWh 机型={res['models']}")
    print("已保存", OUT / f"solution_{tag}.json")


if __name__ == "__main__":
    main()
