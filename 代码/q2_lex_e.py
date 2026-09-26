# -*- coding: utf-8 -*-
"""Q2 能耗优先字典序：固定 N，先最小化总能耗，再在该能耗下压 Cmax。

与 q2_lex.py 互补（那个是 Cmax 优先），用于补齐 N-能耗-Cmax 权衡前沿。
用法: python q2_lex_e.py --n 22 --seconds 200
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from ortools.sat.python import cp_model

sys.path.insert(0, str(Path(__file__).resolve().parent))
from q2_min_n import build, run, dump, SRC, OUT, SCALE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seconds", type=float, default=200)
    ap.add_argument("--horizon", type=int, default=20000)
    args = ap.parse_args()

    pool = json.loads((SRC / "optimization_pool.json").read_text())
    print(f"候选池 {len(pool)} 个；固定 N={args.n}，字典序优化 能耗 → Cmax")

    # 阶段1：最小能耗
    m1, v1 = build(pool, args.horizon, n_fixed=args.n)
    s1, r1 = run(m1, v1["energy"], args.seconds)
    print(f"[阶段1 能耗] 状态={s1.StatusName(r1)}", end=" ")
    if r1 not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print("\n无可行解")
        return
    e1 = int(s1.Value(v1["energy"]))
    print(f"E={e1/SCALE:.4f}kWh  下界={s1.BestObjectiveBound()/SCALE:.4f}  "
          f"用时={s1.WallTime():.1f}s")

    # 阶段2：固定能耗上界，压 Cmax
    m2, v2 = build(pool, args.horizon, n_fixed=args.n, energy_ub=e1)
    s2, r2 = run(m2, v2["cmax"], args.seconds)
    print(f"[阶段2 Cmax] 状态={s2.StatusName(r2)}", end=" ")
    if r2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"Cmax={s2.Value(v2['cmax'])}s  下界={s2.BestObjectiveBound():.0f}  "
              f"用时={s2.WallTime():.1f}s")
        sv, vv = s2, v2
        st2 = dict(status=s2.StatusName(r2), cmax=int(s2.Value(v2["cmax"])),
                   bound=s2.BestObjectiveBound(), wall_s=s2.WallTime())
    else:
        print("失败，沿用阶段1解")
        sv, vv = s1, v1
        st2 = dict(status=s2.StatusName(r2))

    res = dump(pool, sv, vv, f"eN{args.n}")
    res["stage1"] = dict(status=s1.StatusName(r1), energy=e1 / SCALE,
                         bound=s1.BestObjectiveBound() / SCALE, wall_s=s1.WallTime())
    res["stage2"] = st2
    (OUT / f"solution_eN{args.n}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=2))
    print(f"\n=> N={res['n']}  Cmax={res['cmax_s']}s  "
          f"能耗={res['energy_kwh']:.4f}kWh  机型={res['models']}")


if __name__ == "__main__":
    main()
