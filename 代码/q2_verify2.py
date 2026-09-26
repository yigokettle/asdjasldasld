# -*- coding: utf-8 -*-
"""独立验证自带完整字段的解（solution_*.json 内 trips 已含 duration/charge/
offsets/mass/volume），不依赖候选池索引。

做两件事：
 1) 资源着色：区间图贪心为每架次分配实体无人机编号与共享电池编号
    无人机占用 [start, start+duration)，电池占用 [start, start+duration+charge)
 2) 独立复算 16 项：箱覆盖、期望延误、硬截止、载重、体积、能量上限、
    资源不重叠、机队/电池上限、需求覆盖
导出与原推荐解同结构的两张 CSV。
"""
from __future__ import annotations
import argparse
import json
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
UAV_IDS = {"A": ["U01", "U02", "U03", "U04"], "B": ["U05", "U06"],
           "C": ["U07", "U08"]}


def color(intervals, cap, names):
    free, busy, out = list(names), [], {}
    for key, s, e in sorted(intervals, key=lambda z: z[1]):
        busy.sort()
        while busy and busy[0][0] <= s + 1e-6:
            free.append(busy.pop(0)[1])
        if not free:
            raise RuntimeError(f"资源不足：{key} 在 {s:.0f}s 无可用槽位（上限 {cap}）")
        free.sort()
        nm = free.pop(0)
        busy.append((e, nm))
        out[key] = nm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()

    src = OUT / f"solution_{args.tag}.json"
    if not src.exists():
        raise SystemExit(f"找不到 {src}")
    sol = json.loads(src.read_text())
    trips = sol["trips"]
    attr = pd.read_csv(SRC / "optimized_box_delivery.csv").set_index("货箱编号")

    uav_of, bat_of = {}, {}
    for g in "ABC":
        sub = [t for t in trips if t["model"] == g]
        if not sub:
            continue
        uav_of |= color([(t["idx"], t["start_s"],
                          t["start_s"] + t["duration_s"]) for t in sub],
                        DRONES[g], UAV_IDS[g])
        bat_of |= color([(t["idx"], t["start_s"],
                          t["start_s"] + t["duration_s"] + t["charge_s"])
                         for t in sub], BATTERIES[g],
                        [f"{g}-B{k+1:02d}" for k in range(BATTERIES[g])])

    rows, box_rows = [], []
    for k, t in enumerate(sorted(trips, key=lambda z: (z["start_s"], z["idx"])), 1):
        tid = f"OPT-{k:03d}"
        ret = t["start_s"] + t["duration_s"]
        rows.append(dict(
            架次编号=tid, 无人机编号=uav_of[t["idx"]], 机型编号=t["model"],
            电池编号=bat_of[t["idx"]], 开始时刻_s=t["start_s"],
            访问服务区顺序="O01→" + "→".join(t["route"]) + "→O01",
            返回O01时刻_s=ret, 架次能耗_kWh=t["energy_kwh"],
            货箱数=len(t["boxes"]), 载货质量_kg=t["mass_kg"],
            载货体积_m3=t["volume_m3"], 充满时刻_s=ret + t["charge_s"]))
        for b in t["boxes"]:
            a = attr.loc[b]
            dt = t["start_s"] + t["delivery_offsets"][b]
            box_rows.append(dict(
                货箱编号=b, 架次编号=tid, 服务区编号=a["服务区编号"],
                物资类型=a["物资类型"], 是否首批保障=a["是否首批保障"],
                交付完成时刻_s=dt, 硬截止时刻_s=a["硬截止时刻_s"],
                期望送达时刻_s=a["期望送达时刻_s"],
                相对期望延误_s=max(0.0, dt - a["期望送达时刻_s"]),
                应急优先系数=a["应急优先系数"]))
    ts, bd = pd.DataFrame(rows), pd.DataFrame(box_rows)
    ts.to_csv(OUT / f"{args.tag}_trip_schedule.csv", index=False,
              encoding="utf-8-sig")
    bd.to_csv(OUT / f"{args.tag}_box_delivery.csv", index=False,
              encoding="utf-8-sig")

    print("=" * 64)
    print(f"【{args.tag} 独立验证】")
    v = []
    print(f"架次数        : {len(ts)}")
    print(f"交付箱数      : {len(bd)}  唯一 {bd.货箱编号.nunique()}")
    if len(bd) != 80 or bd.货箱编号.nunique() != 80:
        v.append("箱覆盖不为80")
    print(f"最晚返航 Cmax : {ts['返回O01时刻_s'].max():.3f} s")
    print(f"总能耗        : {ts['架次能耗_kWh'].sum():.4f} kWh")

    late = (bd["交付完成时刻_s"] - bd["期望送达时刻_s"]).clip(lower=0)
    print(f"延误>0 箱数   : {(late>1e-6).sum()}   加权延误 "
          f"{(late*bd.应急优先系数).sum():.4f}   按期 {(late<=1e-6).sum()}/80")
    if (late > 1e-6).any():
        v.append("存在期望时刻延误")
    hd = bd.dropna(subset=["硬截止时刻_s"])
    bad = (hd["交付完成时刻_s"] - hd["硬截止时刻_s"] > 1e-6).sum()
    print(f"硬截止({len(hd)}箱) : 违反 {bad}   最小余量 "
          f"{(hd['硬截止时刻_s']-hd['交付完成时刻_s']).min():.3f} s")
    if bad:
        v.append("硬截止违反")

    for r in ts.itertuples():
        q, vol = CAP[r.机型编号]
        if r.载货质量_kg > q + 1e-9:
            v.append(f"{r.架次编号} 超载")
        if r.载货体积_m3 > vol + 1e-9:
            v.append(f"{r.架次编号} 超体积")
        if r.架次能耗_kWh > (1 - RHO) * EUSE[r.机型编号] + 1e-9:
            v.append(f"{r.架次编号} 超能量")
    marg = min((1 - RHO) * EUSE[r.机型编号] - r.架次能耗_kWh for r in ts.itertuples())
    print(f"载重/体积/能量: {'全部合规' if not v else v}   最小能量余量 {marg:.6f} kWh")

    for col, ec, lb in [("无人机编号", "返回O01时刻_s", "无人机"),
                        ("电池编号", "充满时刻_s", "电池")]:
        cnt = 0
        for _nm, g in ts.groupby(col):
            g = g.sort_values("开始时刻_s")
            s_, e_ = g["开始时刻_s"].values, g[ec].values
            cnt += sum(1 for i in range(len(g) - 1) if s_[i + 1] < e_[i] - 1e-6)
        print(f"{lb}重叠冲突  : {cnt}")
        if cnt:
            v.append(f"{lb}资源重叠")

    print(f"实体机使用    : {ts.无人机编号.nunique()} 架 {sorted(ts.无人机编号.unique())}")
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

    dem = pd.read_excel(DATA / "物资需求与配送时限.xlsx")
    need = dem.groupby(["服务区编号", "物资类型"])["总需求箱数"].sum()
    got = bd.groupby(["服务区编号", "物资类型"]).size()
    cmp = pd.DataFrame({"需求": need, "交付": got}).fillna(0).astype(int)
    mism = cmp[cmp.需求 != cmp.交付]
    print(f"需求覆盖      : {len(cmp)-len(mism)}/{len(cmp)} 项匹配")
    if len(mism):
        v.append("需求覆盖不匹配")
        print(mism.to_string())

    print("=" * 64)
    print("结论:", "全部约束通过 ✅" if not v else f"存在问题 ❌ {v}")
    print("已导出:", OUT / f"{args.tag}_trip_schedule.csv")
    print("        ", OUT / f"{args.tag}_box_delivery.csv")


if __name__ == "__main__":
    main()
