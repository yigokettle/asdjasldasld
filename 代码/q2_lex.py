# -*- coding: utf-8 -*-
"""Q2 字典序优化：在给定架次数 N 下依次最小化 Cmax、总能耗。

用法: python q2_lex.py --n 20 --seconds 300
阶段1  固定 N，最小化 Cmax
阶段2  固定 N 且 Cmax <= 阶段1值，最小化总能耗
输出 solution_lexN{n}.json，供 q2_verify.py 还原与验证。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ortools.sat.python import cp_model

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_min_n import build, run, dump, SRC, OUT, SCALE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=300)
    ap.add_argument("--horizon", type=int, default=20000)
    args = ap.parse_args()

    pool = json.loads((SRC / "optimization_pool.json").read_text())
    print(f"候选池 {len(pool)} 个；固定 N={args.n}，字典序优化 Cmax → 能耗")

    # 阶段1：最小 Cmax
    m1, v1 = build(pool, args.horizon, n_fixed=args.n)
    s1, r1 = run(m1, v1["cmax"], args.seconds)
    print(f"[阶段1 Cmax] 状态={s1.StatusName(r1)}", end=" ")
    if r1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n无可行解")
        return
    cmax1 = int(s1.Value(v1["cmax"]))
    print(f"Cmax={cmax1}s  下界={s1.BestObjectiveBound():.0f}  用时={s1.WallTime():.1f}s")

    # 阶段2：固定 Cmax 上界，最小能耗
    m2, v2 = build(pool, args.horizon, n_fixed=args.n, cmax_ub=cmax1)
    s2, r2 = run(m2, v2["energy"], args.seconds)
    print(f"[阶段2 能耗] 状态={s2.StatusName(r2)}", end=" ")
    if r2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"E={s2.Value(v2['energy'])/SCALE:.4f}kWh  "
              f"下界={s2.BestObjectiveBound()/SCALE:.4f}  用时={s2.WallTime():.1f}s")
        sv, vv = s2, v2
        st2 = dict(status=s2.StatusName(r2), energy=s2.Value(v2["energy"]) / SCALE,
                   bound=s2.BestObjectiveBound() / SCALE, wall_s=s2.WallTime())
    else:
        print("失败，沿用阶段1解")
        sv, vv = s1, v1
        st2 = dict(status=s2.StatusName(r2))

    res = dump(pool, sv, vv, f"lexN{args.n}")
    res["stage1"] = dict(status=s1.StatusName(r1), cmax=cmax1,
                         bound=s1.BestObjectiveBound(), wall_s=s1.WallTime())
    res["stage2"] = st2
    (OUT / f"solution_lexN{args.n}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> N={res['n']}  Cmax={res['cmax_s']}s  "
          f"能耗={res['energy_kwh']:.4f}kWh  机型={res['models']}")


if __name__ == "__main__":
    main()
