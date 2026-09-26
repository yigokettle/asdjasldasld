# -*- coding: utf-8 -*-
"""把 CP-SAT 求得的 N 架次解还原为完整调度并独立验证。

做两件事：
 1) 资源着色：按区间图着色为每个架次分配具体无人机编号与电池编号
    （无人机占用 [start,end)，电池占用 [start,end+charge)）
 2) 独立复算：箱覆盖、零延误、硬截止、载重体积、能耗上限、资源不重叠
导出与原推荐解同结构的 CSV，便于整套替换。
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
OUT = ROOT / "Q2_架次优化"
DATA = ROOT / "数据" / "无人机应急物资运输基础数据"

DRONES = {"A": 4, "B": 2, "C": 2}
BATTERIES = {"A": 6, "B": 4, "C": 4}
CAP = {"A": (25, 0.060), "B": (30, 0.073), "C": (80, 0.250)}
EUSE = {"A": 4.5, "B": 4.0, "C": 8.0}
RHO = 0.20
# 实体机编号（题目逐架清单）
UAV_IDS = {"A": ["U01", "U02", "U03", "U04"], "B": ["U05", "U06"], "C": ["U07", "U08"]}


def color(intervals, cap, names):
    """区间图着色：按开始时刻贪心分配资源槽位，返回每个区间的资源名。"""
    free = list(names)                      # 可用槽位
    busy = []                               # (释放时刻, 名称)
    out = {}
    for key, s, e in sorted(intervals, key=lambda z: z[1]):
        busy.sort()
        while busy and busy[0][0] <= s + 1e-9:
            free.append(busy.pop(0)[1])
        if not free:
            raise RuntimeError(f"资源不足：{key} 在 {s} 无可用槽位（上限 {cap}）")
        free.sort()
        name = free.pop(0)
        busy.append((e, name))
        out[key] = name
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=23)
    ap.add_argument("--tag", default=None,
                    help="解文件标识，如 lexN20 / eN22 / cN23；默认用 N{n}")
    args = ap.parse_args()

    tag = args.tag or f"N{args.n}"
    src = OUT / f"solution_{tag}.json"
    if not src.exists():
        raise SystemExit(f"找不到解文件 {src}")
    sol = json.loads(src.read_text())
    trips = sol["trips"]
    if sol["n"] != args.n:
        print(f"[提示] 文件 {src.name} 内 N={sol['n']}，以文件为准")
        args.n = sol["n"]
    pool = json.loads((SRC / "optimization_pool.json").read_text())
    # 原推荐解的箱属性表（期望/硬截止/优先系数），用于独立复算
    base = pd.read_csv(SRC / "optimized_box_delivery.csv")
    attr = base.set_index("货箱编号")

    # ---------- 资源着色 ----------
    uav_of, bat_of = {}, {}
    for g in "ABC":
        sub = [t for t in trips if t["model"] == g]
        if not sub:
            continue
        uav_of |= color([(t["idx"], t["start_s"], t["end_s"]) for t in sub],
                        DRONES[g], UAV_IDS[g])
        bat_of |= color([(t["idx"], t["start_s"], t["bat_end_s"]) for t in sub],
                        BATTERIES[g], [f"{g}-B{k+1:02d}" for k in range(BATTERIES[g])])

    # ---------- 导出调度表 ----------
    rows, box_rows = [], []
    for k, t in enumerate(sorted(trips, key=lambda z: (z["start_s"], z["idx"])), 1):
        c = pool[t["idx"]]
        tid = f"OPT-{k:03d}"
        rows.append(dict(
            架次编号=tid, 无人机编号=uav_of[t["idx"]], 机型编号=t["model"],
            电池编号=bat_of[t["idx"]], 开始时刻_s=t["start_s"],
            访问服务区顺序="O01→" + "→".join(c["route"]) + "→O01",
            返回O01时刻_s=t["start_s"] + c["duration_s"],
            架次能耗_kWh=c["energy_kwh"], 货箱数=len(c["boxes"]),
            载货质量_kg=c["mass_kg"], 载货体积_m3=c["volume_m3"],
            充满时刻_s=t["start_s"] + c["duration_s"] + c["charge_s"]))
        for b in c["boxes"]:
            a = attr.loc[b]
            dt = t["start_s"] + c["delivery_offsets"][b]
            box_rows.append(dict(
                货箱编号=b, 架次编号=tid, 服务区编号=a["服务区编号"],
                物资类型=a["物资类型"], 是否首批保障=a["是否首批保障"],
                交付完成时刻_s=dt, 硬截止时刻_s=a["硬截止时刻_s"],
                期望送达时刻_s=a["期望送达时刻_s"],
                相对期望延误_s=max(0.0, dt - a["期望送达时刻_s"]),
                应急优先系数=a["应急优先系数"]))
    ts = pd.DataFrame(rows)
    bd = pd.DataFrame(box_rows)
    ts.to_csv(OUT / f"{tag}_trip_schedule.csv", index=False, encoding="utf-8-sig")
    bd.to_csv(OUT / f"{tag}_box_delivery.csv", index=False, encoding="utf-8-sig")

    # ---------- 独立验证 ----------
    print("=" * 62)
    print(f"【N={args.n} 方案独立验证】")
    v = []
    print(f"架次数        : {len(ts)}")
    print(f"交付箱数      : {len(bd)}  唯一箱号 {bd.货箱编号.nunique()}")
    if bd.货箱编号.nunique() != 80 or len(bd) != 80:
        v.append("箱覆盖不为80")
    cmax = ts["返回O01时刻_s"].max()
    print(f"最晚返航 Cmax : {cmax:.3f} s")
    print(f"总能耗        : {ts['架次能耗_kWh'].sum():.4f} kWh")

    late = (bd["交付完成时刻_s"] - bd["期望送达时刻_s"]).clip(lower=0)
    print(f"延误>0 箱数   : {(late > 1e-6).sum()}   加权延误 {(late*bd.应急优先系数).sum():.4f}")
    if (late > 1e-6).any():
        v.append("存在期望时刻延误")
    hd = bd.dropna(subset=["硬截止时刻_s"])
    bad = (hd["交付完成时刻_s"] - hd["硬截止时刻_s"] > 1e-6).sum()
    print(f"硬截止违反    : {bad}   最小余量 {(hd['硬截止时刻_s']-hd['交付完成时刻_s']).min():.3f} s")
    if bad:
        v.append("硬截止违反")

    for r in ts.itertuples():
        q, vol = CAP[r.机型编号]
        if r.载货质量_kg > q + 1e-9:
            v.append(f"{r.架次编号} 超载")
        if r.载货体积_m3 > vol + 1e-9:
            v.append(f"{r.架次编号} 超体积")
        if r.架次能耗_kWh > (1 - RHO) * EUSE[r.机型编号] + 1e-9:
            v.append(f"{r.架次编号} 超能量上限")
    marg = min((1 - RHO) * EUSE[r.机型编号] - r.架次能耗_kWh for r in ts.itertuples())
    print(f"载重/体积/能量: {'全部合规' if not v else v}   最小能耗余量 {marg:.6f} kWh")

    # 资源不重叠
    for col, endcol, label in [("无人机编号", "返回O01时刻_s", "无人机"),
                               ("电池编号", "充满时刻_s", "电池")]:
        cnt = 0
        for name, g in ts.groupby(col):
            g = g.sort_values("开始时刻_s")
            s_, e_ = g["开始时刻_s"].values, g[endcol].values
            for i in range(len(g) - 1):
                if s_[i + 1] < e_[i] - 1e-6:
                    cnt += 1
        print(f"{label}重叠冲突  : {cnt}")
        if cnt:
            v.append(f"{label}资源重叠")

    print(f"实体机使用    : {ts.无人机编号.nunique()} 架  {sorted(ts.无人机编号.unique())}")
    print(f"电池使用      : {ts.电池编号.nunique()} 组")
    for g in "ABC":
        s = ts[ts.机型编号 == g]
        if len(s):
            print(f"  {g}型: {s.无人机编号.nunique()}/{DRONES[g]} 架, "
                  f"{s.电池编号.nunique()}/{BATTERIES[g]} 组电池, {len(s)} 架次")
            if s.无人机编号.nunique() > DRONES[g]:
                v.append(f"{g}型超机队")
            if s.电池编号.nunique() > BATTERIES[g]:
                v.append(f"{g}型超电池")

    # 需求覆盖
    dem = pd.read_excel(DATA / "物资需求与配送时限.xlsx")
    need = dem.groupby(["服务区编号", "物资类型"])["总需求箱数"].sum()
    got = bd.groupby(["服务区编号", "物资类型"]).size()
    cmp = pd.DataFrame({"需求": need, "交付": got}).fillna(0).astype(int)
    mism = cmp[cmp.需求 != cmp.交付]
    print(f"需求覆盖      : {len(cmp)-len(mism)}/{len(cmp)} 项匹配")
    if len(mism):
        v.append("需求覆盖不匹配")
        print(mism.to_string())

    print("=" * 62)
    print("结论:", "全部约束通过 ✅" if not v else f"存在问题 ❌ {v}")
    print("已导出:", OUT / f"{tag}_trip_schedule.csv")
    print("        ", OUT / f"{tag}_box_delivery.csv")


if __name__ == "__main__":
    main()
