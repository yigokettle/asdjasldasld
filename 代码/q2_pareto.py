# -*- coding: utf-8 -*-
"""按题目问题二的四目标口径，对已求得的各解做帕累托筛选与 TOPSIS 排序。

题目口径（问题二第1条）：
    综合考虑「配送及时性、全部任务完成时间、运输能耗、架次数」四个指标优化，
    并说明各指标之间的权衡关系。
其中：
  * 配送及时性 —— 医疗物资期望送达、首批保障截止为硬约束；
                  其他物资期望送达时间用于「衡量」及时性（软指标）
  * 全部任务完成时间 Cmax —— 所有无人机完成最后架次并返回 O01 的最晚时刻
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "Q2_架次优化"
SRC = ROOT / "Q2_开源对比与优化"

rows = []
# 原基准（开源对比得到的推荐解）
t0 = pd.read_csv(SRC / "optimized_trip_schedule.csv")
b0 = pd.read_csv(SRC / "optimized_box_delivery.csv")
late0 = (b0["交付完成时刻_s"] - b0["期望送达时刻_s"]).clip(lower=0)
rows.append(dict(方案="原基准25", N=len(t0),
                 Cmax=t0["返回O01时刻_s"].max(),
                 能耗=t0["架次能耗_kWh"].sum(),
                 加权延误=(late0 * b0["应急优先系数"]).sum(),
                 按期箱=(late0 <= 1e-6).sum()))

for f in sorted(OUT.glob("*_trip_schedule.csv")):
    tag = f.name.replace("_trip_schedule.csv", "")
    ts = pd.read_csv(f)
    bd = pd.read_csv(OUT / f"{tag}_box_delivery.csv")
    late = (bd["交付完成时刻_s"] - bd["期望送达时刻_s"]).clip(lower=0)
    rows.append(dict(方案=tag, N=len(ts),
                     Cmax=ts["返回O01时刻_s"].max(),
                     能耗=ts["架次能耗_kWh"].sum(),
                     加权延误=(late * bd["应急优先系数"]).sum(),
                     按期箱=(late <= 1e-6).sum()))

df = pd.DataFrame(rows)

# ---------- 帕累托筛选（四目标全部越小越好；及时性用加权延误代表）----------
OBJ = ["加权延误", "Cmax", "能耗", "N"]
M = df[OBJ].values
dom = np.zeros(len(df), dtype=bool)
for i in range(len(df)):
    for j in range(len(df)):
        if i == j:
            continue
        # j 支配 i：所有目标 <= 且至少一项 <
        if np.all(M[j] <= M[i] + 1e-9) and np.any(M[j] < M[i] - 1e-9):
            dom[i] = True
            break
df["受支配"] = dom
df["帕累托"] = np.where(dom, "", "★")

print("=" * 88)
print("【按题目四目标口径的解集】（加权延误/Cmax/能耗/架次数 均越小越好）")
print("=" * 88)
print(df.sort_values(["N", "Cmax"])[
    ["方案", "N", "Cmax", "能耗", "加权延误", "按期箱", "帕累托"]
].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

pf = df[~df.受支配].copy()
print(f"\n帕累托前沿 {len(pf)} 个解，受支配 {df.受支配.sum()} 个")

# ---------- TOPSIS（四目标等权，及时性因全部零延误而退化为常数）----------
use = ["Cmax", "能耗", "N"]          # 加权延误全为0，不参与区分
X = pf[use].values.astype(float)
Z = X / np.sqrt((X ** 2).sum(axis=0))   # 向量归一化
w = np.array([1 / 3, 1 / 3, 1 / 3])
Z = Z * w
best, worst = Z.min(axis=0), Z.max(axis=0)
dp = np.sqrt(((Z - best) ** 2).sum(axis=1))
dn = np.sqrt(((Z - worst) ** 2).sum(axis=1))
pf["TOPSIS得分"] = dn / (dp + dn)
pf = pf.sort_values("TOPSIS得分", ascending=False)

print("\n" + "=" * 88)
print("【帕累托前沿 TOPSIS 排序】（Cmax/能耗/架次数 等权，越高越均衡）")
print("=" * 88)
print(pf[["方案", "N", "Cmax", "能耗", "TOPSIS得分"]].to_string(
    index=False, float_format=lambda x: f"{x:.4f}"))

# ---------- 相对原基准的变化 ----------
b = df[df.方案 == "原基准25"].iloc[0]
print("\n" + "=" * 88)
print("【帕累托解相对原基准25的变化】")
print("=" * 88)
for r in pf.itertuples():
    print(f"{r.方案:10s} 架次 {b.N:.0f}→{r.N:<3.0f}({(r.N-b.N)/b.N*100:+6.1f}%)  "
          f"Cmax {b.Cmax:.0f}→{r.Cmax:<8.0f}({(r.Cmax-b.Cmax)/b.Cmax*100:+6.1f}%)  "
          f"能耗 {b.能耗:.2f}→{r.能耗:<6.2f}({(r.能耗-b.能耗)/b.能耗*100:+6.1f}%)")

df.to_csv(OUT / "pareto_analysis.csv", index=False, encoding="utf-8-sig")
print("\n已保存", OUT / "pareto_analysis.csv")
