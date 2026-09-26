# -*- coding: utf-8 -*-
"""Q2 候选池扩充：按与 optimization_pool.json 完全一致的物理口径生成更多候选。

目的：原池仅 572 个候选，单架次时长 1070~3201 s，导致 N=23 时 Cmax 卡在 6991 s。
扩充后给求解器更多「短时长」组合可选，以压低全部任务完成时间。

口径（逐项对齐原池 legs 字段，脚本内含自校验）：
  * 距离   球面距离，半径 6378137 m（WGS84 长半轴，由原池 legs 反解校准）
  * 巡航海拔 航段沿线 DEM 像元峰值 + 50 m 净空
  * 作业高度 O01 = 地面海拔；服务区 = 地面海拔 + 30 m
  * 时间   水平段 d/v_cr + 爬升 climb/v_up + 下降 descent/v_down
           + 工位准备 T_setup + 每箱装载 T_load
           + 每站基础交接 T_handover + 每箱增量 T_handover_p
  * 能耗   水平 E_use*d/L_g(q) + 爬升 (M_g0+q)*g*climb/eta_up/3.6e6，g=9.81
  * 载荷   逐站卸货，去程带剩余载荷，返程空载
  * 安全   E_trip <= (1-rho)*E_use
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Q2_开源对比与优化"
DATA = ROOT / "数据" / "无人机应急物资运输基础数据"
DEM_TIF = (ROOT / "数据" / "镇龙乡地理空间数据" / "镇龙乡及周边地理数据"
           / "数字高程模型数据（DEM）" / "镇龙乡及周边30米DEM.tif")
OUT = ROOT / "Q2_架次优化"
OUT.mkdir(parents=True, exist_ok=True)

G = 9.81
KWH_J = 3.6e6
CLEAR = 50.0        # 巡航净空
OPS_S = 30.0        # 服务区作业高度
R_EARTH = 6378137.0  # 原池口径：WGS84 长半轴（由原池 legs 反解，std<1e-5）
PEAK_STEP = 1.0     # 沿线 DEM 峰值采样步长（1 m 等价于逐像元遍历，已全量核验）

# ---------------- 机型参数 ----------------
GT = {
    "A": dict(M0=70.0, Q=25.0, V=0.060, vc=12.0, L0=25000.0, LF=20000.0,
              Euse=4.5, rho=0.20, eta=0.72, v_up=3.0, v_dn=2.5,
              t_prep=300.0, t_load=30.0, t_hand=150.0, t_hand_box=30.0,
              full_charge=1800.0),
    "B": dict(M0=65.0, Q=30.0, V=0.073, vc=15.0, L0=28000.0, LF=16000.0,
              Euse=4.0, rho=0.20, eta=0.72, v_up=3.0, v_dn=2.5,
              t_prep=300.0, t_load=30.0, t_hand=150.0, t_hand_box=30.0,
              full_charge=2400.0),
    "C": dict(M0=69.9, Q=80.0, V=0.250, vc=15.0, L0=26000.0, LF=12000.0,
              Euse=8.0, rho=0.20, eta=0.72, v_up=2.5, v_dn=2.0,
              t_prep=300.0, t_load=30.0, t_hand=180.0, t_hand_box=36.0,
              full_charge=3000.0),
}

# ---------------- 节点与 DEM ----------------
# 用 Q1 已核验过的节点表（与 xlsx 单元格值逐项一致）
# The original development checkout used a review-only CSV.  In the released
# repository the authoritative node table is the first sheet of the attached
# workbook; read it positionally because the workbook may be opened with a
# legacy Chinese code page on Windows.
_raw_nd = pd.read_excel(DATA / "调度中心与服务区.xlsx", header=None)
_rows = _raw_nd.iloc[2:3].values.tolist() + _raw_nd.iloc[6:21].values.tolist()
NODES = {str(r[0]): (float(r[2]), float(r[3]), float(r[4])) for r in _rows
         if pd.notna(r[0])}

im = Image.open(DEM_TIF)
DEM = np.array(im, dtype=np.float64)
NY, NX = DEM.shape
PS = 1.0 / 3600.0
LON0, LAT0 = 109.03277777777778, 23.224722222222223


def dem_elev(lon, lat):
    c = (np.asarray(lon) - LON0) / PS
    r = (LAT0 - np.asarray(lat)) / PS
    rr = np.clip(np.round(r).astype(int), 0, NY - 1)
    cc = np.clip(np.round(c).astype(int), 0, NX - 1)
    return DEM[rr, cc]


def sphere_m(lo0, la0, lo1, la1):
    p0, p1 = math.radians(la0), math.radians(la1)
    dp, dl = p1 - p0, math.radians(lo1 - lo0)
    a = math.sin(dp / 2) ** 2 + math.cos(p0) * math.cos(p1) * math.sin(dl / 2) ** 2
    return 2 * R_EARTH * math.asin(math.sqrt(a))


_GEO: dict = {}


def leg_geo(i, j):
    """有向航段几何（距离/巡航海拔/爬升/下降），按节点对缓存。"""
    key = (i, j)
    if key in _GEO:
        return _GEO[key]
    lo0, la0, _ = NODES[i]
    lo1, la1, _ = NODES[j]
    d = sphere_m(lo0, la0, lo1, la1)
    n = max(int(math.ceil(d / PEAK_STEP)) + 1, 2)
    t = np.linspace(0, 1, n)
    zc = float(np.nanmax(dem_elev(lo0 + (lo1 - lo0) * t,
                                  la0 + (la1 - la0) * t))) + CLEAR
    g = (d, zc, max(0.0, zc - ops_alt(i)), max(0.0, zc - ops_alt(j)))
    _GEO[key] = g
    return g


def peak_alt(lo0, la0, lo1, la1, step_m=PEAK_STEP):
    d = sphere_m(lo0, la0, lo1, la1)
    n = max(int(math.ceil(d / step_m)) + 1, 2)
    t = np.linspace(0, 1, n)
    return float(np.nanmax(dem_elev(lo0 + (lo1 - lo0) * t, la0 + (la1 - la0) * t)))


def ops_alt(nid):
    return NODES[nid][2] + (0.0 if nid == "O01" else OPS_S)


def equiv_range(g, q):
    p = GT[g]
    q = min(max(q, 0.0), p["Q"])
    return p["L0"] - (p["L0"] - p["LF"]) / p["Q"] ** 1.5 * q ** 1.5


def eval_trip(g, route, boxes, binfo):
    """按原池口径计算一个候选的时间/能耗/逐箱交付偏移。"""
    p = GT[g]
    mass = sum(binfo[b]["mass"] for b in boxes)
    vol = sum(binfo[b]["vol"] for b in boxes)
    if mass > p["Q"] + 1e-9 or vol > p["V"] + 1e-9:
        return None
    # 每站货箱
    at = {s: [b for b in boxes if binfo[b]["area"] == s] for s in route}
    if any(not at[s] for s in route):
        return None

    t = p["t_prep"] + p["t_load"] * len(boxes)      # 工位准备 + 装载
    e = 0.0
    legs, offsets = [], {}
    payload = mass
    seq = ["O01"] + list(route) + ["O01"]
    for k in range(len(seq) - 1):
        i, j = seq[k], seq[k + 1]
        d, zc, climb, desc = leg_geo(i, j)
        tt = d / p["vc"] + climb / p["v_up"] + desc / p["v_dn"]
        eh = d / equiv_range(g, payload) * p["Euse"]
        ec = (p["M0"] + payload) * G * climb / p["eta"] / KWH_J
        t += tt
        e += eh + ec
        legs.append(dict(**{"from": i, "to": j}, payload_kg=payload,
                         distance_m=d, climb_m=climb, descent_m=desc,
                         cruise_alt_m=zc, time_s=tt,
                         horizontal_energy_kwh=eh, climb_energy_kwh=ec,
                         energy_kwh=eh + ec))
        if j != "O01":
            # 到站交接
            t += p["t_hand"] + p["t_hand_box"] * len(at[j])
            for b in at[j]:
                offsets[b] = t
            payload -= sum(binfo[b]["mass"] for b in at[j])
    if e > (1 - p["rho"]) * p["Euse"] + 1e-12:
        return None
    return dict(model=g, boxes=list(boxes), route=list(route),
                mass_kg=mass, volume_m3=vol, duration_s=t, return_s=t,
                energy_kwh=e, energy_limit_kwh=(1 - p["rho"]) * p["Euse"],
                delivery_offsets=offsets, delivery_times=dict(offsets),
                legs=legs,
                charge_s=e / p["Euse"] * p["full_charge"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-boxes", type=int, default=9)
    ap.add_argument("--selftest", action="store_true", help="仅做口径自校验")
    args = ap.parse_args()

    old = json.loads((SRC / "optimization_pool.json").read_text())
    bd = pd.read_csv(SRC / "optimized_box_delivery.csv")
    binfo = {}
    dem = pd.read_excel(DATA / "物资需求与配送时限.xlsx")
    mcol = [c for c in dem.columns if "单箱质量" in str(c)][0]
    vcol = [c for c in dem.columns if "单箱体积" in str(c)][0]
    mv = dem.set_index(["服务区编号", "物资类型"])[[mcol, vcol]]
    for r in bd.itertuples():
        m_, v_ = mv.loc[(r.服务区编号, r.物资类型)]
        binfo[r.货箱编号] = dict(area=r.服务区编号, mass=float(m_), vol=float(v_),
                               exp=float(r.期望送达时刻_s),
                               hard=None if pd.isna(r.硬截止时刻_s) else float(r.硬截止时刻_s))

    # ---------- 口径自校验：复算原池全部候选 ----------
    print("=" * 62)
    print(f"【口径自校验】复算原池 {len(old)} 个候选，对比 duration/energy/legs")
    dt, de, dl = [], [], []
    for c in old:
        r = eval_trip(c["model"], c["route"], c["boxes"], binfo)
        if r is None:
            dt.append(9e9)
            continue
        dt.append(abs(r["duration_s"] - c["duration_s"]))
        de.append(abs(r["energy_kwh"] - c["energy_kwh"]))
        for a, b in zip(c["legs"], r["legs"]):
            dl.append(max(abs(b[k] - a[k]) for k in
                          ("distance_m", "climb_m", "descent_m", "cruise_alt_m",
                           "time_s", "energy_kwh")))
    print(f"  时长最大差 {max(dt):.8f} s   能耗最大差 {max(de):.10f} kWh   "
          f"逐段字段最大差 {max(dl):.10f}")
    if max(dt) > 1e-6 or max(de) > 1e-9:
        print("  ⚠️ 口径不一致，终止扩充")
        return
    print("  ✅ 口径与原池逐字段一致，可安全扩充")
    if args.selftest:
        return

    # ---------- 枚举扩充 ----------
    boxes_by_area = {}
    for b, v in binfo.items():
        boxes_by_area.setdefault(v["area"], []).append(b)
    for s in boxes_by_area:
        boxes_by_area[s].sort()
    areas = sorted(boxes_by_area)
    print(f"\n服务区 {len(areas)} 个，货箱 {len(binfo)} 个")

    seen = {(c["model"], tuple(c["route"]), tuple(sorted(c["boxes"])))
            for c in old}
    new = []

    def emit(g, route, bs):
        key = (g, tuple(route), tuple(sorted(bs)))
        if key in seen:
            return
        r = eval_trip(g, route, bs, binfo)
        if r is None:
            return
        seen.add(key)
        new.append(r)

    # 单站：每个服务区的任意箱子子集（按箱数上限截断）
    for g in "ABC":
        cap = GT[g]["Q"]
        for s in areas:
            bl = boxes_by_area[s]
            kmax = min(len(bl), args.max_boxes)
            for k in range(1, kmax + 1):
                for combo in itertools.combinations(bl, k):
                    if sum(binfo[b]["mass"] for b in combo) > cap + 1e-9:
                        continue
                    emit(g, [s], combo)
    print(f"单站枚举后新增 {len(new)}")

    # 两站：所有有序服务区对 × 两侧箱子子集
    n1 = len(new)
    for g in "ABC":
        cap, vcap = GT[g]["Q"], GT[g]["V"]
        for s1, s2 in itertools.permutations(areas, 2):
            b1, b2 = boxes_by_area[s1], boxes_by_area[s2]
            for k1 in range(1, min(len(b1), args.max_boxes) + 1):
                for c1 in itertools.combinations(b1, k1):
                    m1 = sum(binfo[b]["mass"] for b in c1)
                    v1 = sum(binfo[b]["vol"] for b in c1)
                    if m1 > cap + 1e-9 or v1 > vcap + 1e-9:
                        continue
                    kr = min(len(b2), args.max_boxes - k1)
                    for k2 in range(1, kr + 1):
                        for c2 in itertools.combinations(b2, k2):
                            if m1 + sum(binfo[b]["mass"] for b in c2) > cap + 1e-9:
                                continue
                            if v1 + sum(binfo[b]["vol"] for b in c2) > vcap + 1e-9:
                                continue
                            emit(g, [s1, s2], c1 + c2)
    print(f"两站枚举后新增 {len(new)}（其中两站 {len(new)-n1}）")

    merged = old + new
    (OUT / "pool_expanded.json").write_text(
        json.dumps(merged, ensure_ascii=False))
    ds = [c["duration_s"] for c in merged]
    print(f"\n合并池 {len(merged)} 个候选（原 {len(old)} + 新 {len(new)}）")
    print(f"  时长范围 {min(ds):.1f} ~ {max(ds):.1f} s")
    print(f"  机型分布 {dict(Counter(c['model'] for c in merged))}")
    print("已保存", OUT / "pool_expanded.json")


if __name__ == "__main__":
    main()
