"""问题三V3独立结果核验：原始硬时限、真实物理、资源和实际直连优先。

不调用旧版结果核验，不覆盖prefix_通信保障.csv。精确通信单独输出到
prefix_精确_*；本程序核验结果输出到prefix_V3_*。
python 问题三_V3_结果核验.py --prefix 问题三_冻结推荐方案
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from 问题二_物理模型与输入 import read_inputs, pairwise_legs, route, charge_seconds
from 问题三_通信几何 import Scene, LIMIT


def resolve(prefix):
    p = Path(prefix)
    return p if p.is_absolute() else ROOT / p


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def original_file(name):
    candidates = [ROOT/name, ROOT/'q1_flat'/name,
                  ROOT.parent/'q3_source'/'华为杯'/name,
                  ROOT.parent/'q3_source'/'original'/'数据'/'无人机应急物资运输基础数据'/name]
    return next((p for p in candidates if p.is_file()), None)


def close(value, expected, label, tol=1e-7):
    assert math.isfinite(float(value)) and abs(float(value)-float(expected)) <= tol, (
        label, value, expected)


def peak(frame, start, end):
    events = []
    for a, b in frame[[start, end]].itertuples(index=False, name=None):
        assert b >= a >= 0, (start, end, a, b)
        events += [(float(a), 1), (float(b), -1)]
    current = maximum = 0
    for _, delta in sorted(events):
        current += delta
        maximum = max(maximum, current)
    assert current == 0
    return maximum


def disjoint(frame, resource, start, end, allowed):
    assert set(frame[resource]).issubset(set(allowed)), (resource, set(frame[resource])-set(allowed))
    for rid, sub in frame.groupby(resource):
        sub = sub.sort_values(start)
        assert np.all(sub[start].to_numpy()[1:] >= sub[end].to_numpy()[:-1]-1e-8), (
            resource, rid, '占用或充电重叠')


def relay_parameters():
    p = original_file('中继无人机数据.xlsx')
    c = original_file('通信链路参数.xlsx')
    assert p and c, '缺少原始中继或通信Excel'
    raw = pd.read_excel(p, sheet_name='数据', header=None)
    r = raw.iloc[2]
    params = dict(mass=float(r[4]), speed=float(r[5]), cruise_kw=float(r[6]),
                  battery=float(r[7]), reserve=float(r[8])/100, prep=float(r[9]),
                  setup=float(r[10]), turnaround=float(r[11]), climb=float(r[12]),
                  descend=float(r[13]), efficiency=float(r[14]), hover_kw=float(r[16]),
                  comm_kw=float(r[17]), max_agl=float(r[18]),
                  components=int(raw.iloc[11, 1]), full_charge=float(raw.iloc[11, 2]))
    ids = [str(raw.iloc[k, 0]) for k in [6, 7]]
    link = pd.read_excel(c, sheet_name='数据', header=None)
    values = {(str(row[0]), str(row[1])): float(row[4])
              for row in link.iloc[2:16].itertuples(index=False, name=None)}
    f = values['传播参数', '载波频率（MHz）']
    loss = values['传播参数', '系统损耗（dB）']
    threshold = values['接收参数', '接收灵敏度（dBm）'] + values['接收参数', '衰落裕量（dB）']
    def budget(a, b):
        return min(values[a, '发射功率（dBm）'] + values[a, '天线增益（dBi）']
                   + values[b, '天线增益（dBi）'] - loss - threshold,
                   values[b, '发射功率（dBm）'] + values[b, '天线增益（dBi）']
                   + values[a, '天线增益（dBi）'] - loss - threshold)
    limits = dict(direct=budget('运输无人机', '固定网关 G01'),
                  access=budget('运输无人机', '中继接入端'),
                  backhaul=budget('中继回传端', '固定网关 G01'))
    assert limits == LIMIT and f == 2400 and values['传播参数', '地形遮挡附加损耗（dB）'] == 10
    assert values['固定网关 G01', '天线离地高度（m）'] == 20
    return params, ids, [p, c], limits


def physics(prefix):
    base = resolve(prefix)
    trans = str(base) + '_运输'
    paths = [Path(trans+'_架次.csv'), Path(trans+'_逐箱.csv'), Path(trans+'_航段.csv'),
             Path(str(base)+'_中继架次.csv')]
    F, D, L, R = [pd.read_csv(p) for p in paths]
    N, B, G, U, BT = read_inputs()
    # 两种硬规则同时适用时取更早者；普通期望时间不硬化。
    hard = np.minimum(np.where(B.first_batch.eq('是'), B.first_deadline_s, np.inf),
                      np.where(B.material_type.eq('医疗物资'), B.target_delivery_s, np.inf))
    B['hard_deadline_s'] = hard
    assert np.isfinite(hard).sum() == 31 and np.isinf(hard).sum() == 49
    assert D['货箱编号'].is_unique and set(D['货箱编号']) == set(B.index) and len(D) == 80
    assert F['架次编号'].is_unique and set(D['架次编号']) == set(F['架次编号'])
    assert set(L['架次编号']) == set(F['架次编号'])
    assert set(F['机型']).issubset(set(G.index))
    legs = pairwise_legs(N)
    energies = []; box_checks = []; resource_checks = []
    trans_energy = 0.; trans_soc = []; max_arc_error = 0.
    for rec in F.to_dict('records'):
        fid = rec['架次编号']; model = G.loc[rec['机型']]
        dl = D[D['架次编号'].eq(fid)]
        ids = str(rec['货箱编号']).split(',')
        assert len(ids) == len(set(ids)) and set(ids) == set(dl['货箱编号']), fid
        seq = str(rec['访问顺序']).split('→')
        assert seq[0] == seq[-1] == 'O01' and len(seq) >= 3, fid
        stops = seq[1:-1]
        assert set(stops).issubset(set(N.index)-{'O01'}) and len(stops) == len(set(stops)), fid
        alloc = {site: [bid for bid in ids if B.loc[bid, 'node_id'] == site] for site in stops}
        assert all(alloc.values()) and sum(map(len, alloc.values())) == len(ids), fid
        assert rec['开始秒'] >= 0
        rr = route(stops, alloc, rec['机型'], B, G, legs, at=float(rec['开始秒']))
        assert rr is not None, (fid, '载荷/体积/硬时限/安全能量违反')
        close(rec['质量kg'], rr['mass_kg'], fid+'质量')
        close(rec['体积m3'], rr['volume_m3'], fid+'体积')
        close(rec['能耗kWh'], rr['energy_kwh'], fid+'能耗')
        close(rec['返航SOC%'], 100*rr['remaining_soc'], fid+'SOC')
        assert rr['remaining_soc'] >= model.reserve_percent/100-1e-9
        elapsed = int(model.fixed_prep_s+model.load_per_box_s*len(ids))
        observed_legs = L[L['架次编号'].eq(fid)].reset_index(drop=True)
        assert len(observed_legs) == len(rr['leg_rows']), fid
        for j, actual in enumerate(rr['leg_rows']):
            elapsed += math.ceil(actual['flight_s']-1e-10)
            obs = observed_legs.iloc[j]; u=actual['from_node']; v=actual['to_node']
            assert (obs['起点'], obs['终点']) == (u, v), fid
            for col, val in [('剩余载荷kg', actual['remaining_payload_kg']),
                             ('距离m', actual['distance_m']), ('巡航海拔m', actual['cruise_altitude_m']),
                             ('爬升m', legs[u,v]['climb_m']), ('下降m', legs[u,v]['descent_m']),
                             ('飞行秒', math.ceil(actual['flight_s']-1e-10)),
                             ('航段能耗kWh', actual['energy_kwh'])]:
                close(obs[col], val, fid+col)
            max_arc_error = max(max_arc_error, abs(float(obs['航段能耗kWh'])-actual['energy_kwh']))
            if v != 'O01':
                elapsed += int(model.handoff_base_s+model.handoff_per_box_s*len(alloc[v]))
                for bid in alloc[v]:
                    row = dl.loc[dl['货箱编号'].eq(bid)].iloc[0]
                    arrival = float(rec['开始秒'])+elapsed
                    close(row['送达秒'], arrival, bid+'送达')
                    assert row['服务区'] == B.loc[bid, 'node_id']
                    deadline = float(B.loc[bid, 'hard_deadline_s']); target = float(B.loc[bid, 'target_delivery_s'])
                    assert arrival <= deadline+1e-8, (bid, arrival, deadline)
                    weight = float(B.loc[bid, 'priority'])
                    penalty = weight*max(0, arrival-target)/target
                    close(row['归一化加权延误'], penalty, bid+'延误')
                    close(row['期望送达秒'], target, bid+'期望')
                    close(row['优先系数'], weight, bid+'优先系数')
                    if math.isfinite(deadline): close(row['硬截止秒'], deadline, bid+'硬截止')
                    else: assert pd.isna(row['硬截止秒']) or math.isinf(float(row['硬截止秒'])), bid
                    box_checks.append(dict(货箱编号=bid, 服务区=row['服务区'], 运输架次=fid,
                        实际送达秒=arrival, 原始硬截止秒=deadline if math.isfinite(deadline) else None,
                        期望送达秒=target, 硬时限货箱=math.isfinite(deadline),
                        硬余量秒=deadline-arrival if math.isfinite(deadline) else None,
                        期望余量秒=target-arrival, 软延误秒=max(0, arrival-target),
                        归一化加权延误=penalty))
        close(rec['返回秒'], float(rec['开始秒'])+elapsed, fid+'返航')
        assert rec['无人机'] in U and rec['电池'] in BT
        assert U[rec['无人机']]['model'] == rec['机型'] == BT[rec['电池']]['model'], fid
        charge = math.ceil(charge_seconds(rr['remaining_soc'], BT[rec['电池']]['full_charge_s'])-1e-9)
        assert rec['电池充满秒'] >= rec['返回秒']+charge-1e-8, fid
        trans_energy += rr['energy_kwh']; trans_soc.append(rr['remaining_soc']*100)
        energies.append(dict(类别='运输', 架次=fid, 实算能耗kWh=rr['energy_kwh'],
            实算返航SOC百分比=rr['remaining_soc']*100, 实算充电秒=charge,
            保守额外充电秒=float(rec['电池充满秒']-rec['返回秒']-charge)))
    close(F['质量kg'].sum(), B.mass_kg.sum(), '全部货箱质量')
    close(F['体积m3'].sum(), B.volume_m3.sum(), '全部货箱体积')
    disjoint(F, '无人机', '开始秒', '返回秒', U)
    disjoint(F, '电池', '开始秒', '电池充满秒', BT)
    for g in G.index:
        sub = F[F['机型'].eq(g)]
        for label, end, inventory in [('运输无人机', '返回秒', sum(x['model']==g for x in U.values())),
                                      ('运输共享电池', '电池充满秒', sum(x['model']==g for x in BT.values()))]:
            pk=peak(sub,'开始秒',end);assert pk<=inventory
            resource_checks.append(dict(资源=label, 机型=g, 峰值=pk, 库存=inventory, 通过=True))
    params, relay_ids, extra_sources, limits = relay_parameters()
    assert R['中继架次'].is_unique
    scene = Scene(); z0 = float(N.loc['O01','ground_elevation_m']); relay_energy=0.; relay_soc=[]
    for rec in R.to_dict('records'):
        rid=rec['中继架次'];ground=scene.ground(float(rec['经度']),float(rec['纬度']))
        agl=float(rec['悬停海拔m'])-ground
        assert math.isfinite(ground) and 0<=agl<=params['max_agl']+1e-7, rid
        station=scene.station(str(rec['站点']),float(rec['经度']),float(rec['纬度']),agl)
        assert station['backhaul_ok'], (rid, '回传不可用')
        h=max(station['line_max_z']+50,float(rec['悬停海拔m']),z0)
        d=station['distance_m'];z=float(rec['悬停海拔m'])
        out=math.ceil((h-z0)/params['climb']+d/params['speed']+(h-z)/params['descend']-1e-10)
        back=math.ceil((h-z)/params['climb']+d/params['speed']+(h-z0)/params['descend']-1e-10)
        assert rec['开始秒']>=0 and rec['服务结束秒']>=rec['建链完成秒']
        close(rec['建链完成秒'],rec['开始秒']+params['prep']+out+params['setup'],rid+'建链')
        close(rec['服务时长秒'],rec['服务结束秒']-rec['建链完成秒'],rid+'服务时长')
        close(rec['返回秒'],rec['服务结束秒']+back,rid+'返回')
        e=params['cruise_kw']*(2*d/params['speed'])/3600
        e+=params['mass']*9.80665*((h-z0)+(h-z))/(3.6e6*params['efficiency'])
        e+=(params['hover_kw']+params['comm_kw'])*(params['setup']+rec['服务时长秒'])/3600
        soc=1-e/params['battery'];assert soc>=params['reserve']-1e-9
        close(rec['能耗kWh'],e,rid+'能耗');close(rec['返航SOC%'],100*soc,rid+'SOC')
        charge=math.ceil(charge_seconds(soc,params['full_charge'])-1e-9)
        assert rec['组件充满秒']>=rec['返回秒']+charge-1e-8
        close(rec['无人机可用秒'],rec['返回秒']+params['turnaround'],rid+'周转')
        close(rec['充电秒'],rec['组件充满秒']-rec['返回秒'],rid+'充电记录')
        relay_energy+=e;relay_soc.append(100*soc)
        energies.append(dict(类别='中继',架次=rid,实算能耗kWh=e,实算返航SOC百分比=100*soc,
            实算充电秒=charge,保守额外充电秒=float(rec['组件充满秒']-rec['返回秒']-charge)))
    disjoint(R,'中继无人机','开始秒','无人机可用秒',relay_ids)
    disjoint(R,'能源组件','开始秒','组件充满秒',[f'H{i:02d}'for i in range(1,params['components']+1)])
    for label,end,inventory in [('中继无人机','无人机可用秒',len(relay_ids)),
                                ('中继能源组件','组件充满秒',params['components'])]:
        pk=peak(R,'开始秒',end);assert pk<=inventory
        resource_checks.append(dict(资源=label,机型='R',峰值=pk,库存=inventory,通过=True))
    BC=pd.DataFrame(box_checks); hardbox=BC[BC['硬时限货箱']];softbox=BC[~BC['硬时限货箱']]
    transport_end=float(F['返回秒'].max());relay_end=float(R['返回秒'].max()) if len(R) else 0.
    result=dict(status='物理与资源通过',source_prefix=str(base),运输架次=len(F),中继架次=len(R),
        两类架次分别报告=dict(运输=len(F),中继=len(R)),合计架次=len(F)+len(R),交付货箱=len(D),
        原始硬时限货箱=len(hardbox),普通软时限货箱=len(softbox),硬时限全部满足=True,
        软时限迟到箱数=int(softbox['软延误秒'].gt(1e-8).sum()),
        软时限延误总秒=float(softbox['软延误秒'].sum()),
        归一化加权延误=float(BC['归一化加权延误'].sum()),
        最小硬余量秒=float(hardbox['硬余量秒'].min()),
        运输完成秒=transport_end,中继完成秒=relay_end,联合完成秒=max(transport_end,relay_end),
        最后货箱交付秒=float(D['送达秒'].max()),运输能耗kWh=trans_energy,中继能耗kWh=relay_energy,
        总能耗kWh=trans_energy+relay_energy,运输最小返航SOC百分比=min(trans_soc),
        中继最小返航SOC百分比=min(relay_soc) if relay_soc else None,
        逐航段能耗最大误差kWh=max_arc_error,质量kg=float(B.mass_kg.sum()),体积m3=float(B.volume_m3.sum()),
        资源峰值=resource_checks,双向链路门限dB=limits,
        原始时限口径='医疗期望及首批截止为硬约束；49个普通货箱期望仅计软延误，未将期望硬化。')
    summary_path=Path(str(base)+'_结果.json')
    if summary_path.exists():
        saved=json.loads(summary_path.read_text(encoding='utf-8'))
        for field,value in [('N_transport',len(F)),('N_relay',len(R)),('transport_energy_kwh',trans_energy),
                ('relay_energy_kwh',relay_energy),('total_energy_kwh',trans_energy+relay_energy),
                ('transport_finish_s',transport_end),('joint_finish_s',max(transport_end,relay_end)),
                ('weighted_lateness',float(BC['归一化加权延误'].sum()))]:
            if field in saved: close(saved[field],value,'结果汇总'+field,1e-6)
        paths.append(summary_path)
    paths += extra_sources+[ROOT/'q1_flat'/n for n in ['最终工作DEM.tif','调度中心与服务区.xlsx',
                                                   '物资需求与配送时限.xlsx','运输无人机数据.xlsx']]
    result['source_sha256']={str(p):digest(p)for p in paths}
    BC.to_csv(str(base)+'_V3_逐箱核验.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(resource_checks).to_csv(str(base)+'_V3_资源核验.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(energies).to_csv(str(base)+'_V3_能源核验.csv',index=False,encoding='utf-8-sig')
    return result


def run(prefix, communication=True, backend='auto'):
    base=resolve(prefix);started=time.perf_counter()
    old=Path(str(base)+'_通信保障.csv');oldhash=digest(old)if old.exists()else None
    result=physics(base)
    if communication:
        from 问题三_V3_通信状态 import run as exact_communication
        report=exact_communication(str(base),output_prefix=str(base)+'_精确',backend=backend)
        assert report['status']=='通过'and report['failed_coverage_count']==0
        assert report['direct_priority_mismatches']==0
        result['精确通信核验']=report
        result['精确通信文件SHA256']={str(p):digest(p)for p in sorted(base.parent.glob(base.name+'_精确_*'))
                                    if p.suffix in ['.csv','.json']}
        result['status']='全部通过'
    else:
        result['精确通信核验']='未运行；本次不声明通信已核验'
    assert (digest(old)if old.exists()else None)==oldhash,'不得改写旧保守保障记录'
    result['旧通信记录保持不变']=True
    result['seconds']=time.perf_counter()-started
    result['scope']='原始表、工作DEM及题定物理规则独立复算；不提供全局最优证明。精确通信另存_精确_*，不以旧保守标签代替实际状态。'
    Path(str(base)+'_V3_独立核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items()if k not in ['source_sha256','精确通信核验','精确通信文件SHA256']},ensure_ascii=False,indent=2))
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='问题三_冻结推荐方案')
    ap.add_argument('--physics-only',action='store_true');ap.add_argument('--backend',choices=['auto','python'],default='auto')
    a=ap.parse_args()
    try:run(a.prefix,not a.physics_only,a.backend)
    except Exception as exc:
        failure={'status':'失败','error':str(exc),'traceback':traceback.format_exc()}
        Path(str(resolve(a.prefix))+'_V3_独立核验.json').write_text(json.dumps(failure,ensure_ascii=False,indent=2),encoding='utf-8')
        raise
