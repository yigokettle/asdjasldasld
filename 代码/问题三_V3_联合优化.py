"""问题三V3：真实硬时限＋普通物资软延误，有限路线池与离散中继点联合CP-SAT。

输入问题三_通信联合输入.json：
  stations: {id, lon, lat, altitude_m, outbound_s, return_s,
             fixed_energy_kwh}。fixed_energy包含来回飞行和30秒建链，不含服务。
  routes: {pool_index, blind_intervals:[{lo,hi,stations:[id,...]}]}。
  每一通信盲区区间向外取整至整数秒，且必须被同一中继任务完整覆盖。

这是一种保守的、可检查的有限模型；OPTIMAL 仅适用于本模型，
不表示连续候选位置及所有潜在运输路线的全局最优。
"""
from __future__ import annotations
import argparse, collections, hashlib, json, math, sys, time
from fractions import Fraction
from pathlib import Path
from ortools.sat.python import cp_model
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent
sys.path.insert(0,str(P))
from 问题二_地形ALNS_CPSAT统一实验 import Engine, plan, export_solution


def charge_seconds(energy_kwh):
    soc=1-energy_kwh/3.2
    return math.ceil((1800*(.65*(.9-soc)/.9+.35) if soc<.9 else
                      1800*.35*(1-soc)/.1)-1e-9)


def load_problem(geometry_file, pool_file, fixed=False, context_file=None):
    context=json.loads(Path(context_file or P/'问题二_统一实验输入.json').read_text(encoding='utf-8'))
    pool=json.loads(Path(pool_file).read_text(encoding='utf-8'))
    geom=json.loads(Path(geometry_file).read_text(encoding='utf-8'))
    keys=[plan(*k) for k in pool['keys']]
    if fixed:keys=keys[:19]
    eng=Engine(context)
    # 医疗及首批硬时限保留原值；其余期望时限进入迟到变量。
    routes=[eng.evaluate(k) for k in keys]
    index={int(r['pool_index']):r for r in geom['routes']}
    assert len(index)>=len(keys),'通信几何记录未覆盖全部路线池'
    return context,keys,routes,eng,geom,index


def objective_vector(snapshot,eng):
    sol=snapshot['transport'];kk=[plan(*k) for k in sol['keys']]
    mm=eng.metrics(kk,sol['starts']);rr=snapshot['relay_records']
    c=max([s+eng.evaluate(k).duration for k,s in zip(kk,sol['starts'])]+[r['返回秒'] for r in rr])
    scaled=sum(int(eng.weight[b])*max(0,int(s)+dt-int(eng.target[b]))
               for k,s in zip(kk,sol['starts']) for b,dt in eng.evaluate(k).delta.items())
    snapshot['late_scaled']=scaled;snapshot['late_denominator']=eng.denom
    return [scaled/eng.denom,int(c),float(mm[2]+sum(r['能耗kWh'] for r in rr)),len(kk),len(rr)]


def update_archive(archive,snapshot):
    v=np.array(snapshot['objective_vector'],float)
    def dom(a,b):return np.all(a<=b+1e-9) and np.any(a<b-1e-9)
    if any(dom(np.array(r['objective_vector']),v) for r in archive):return
    if any(np.allclose(np.array(r['objective_vector']),v,atol=1e-9,rtol=0) for r in archive):return
    archive[:]=[r for r in archive if not dom(v,np.array(r['objective_vector']))]
    archive.append(snapshot)


def ranking(s,eng,goal):
    f=objective_vector(s,eng)
    if goal=='time':return (f[1],f[2],f[0],f[3],f[4])
    if goal=='energy':return (f[2],f[1],f[0],f[3],f[4])
    if goal=='relay':return (f[4],f[1],f[2],f[0],f[3])
    if goal=='transport':return (f[3],f[1],f[2],f[0],f[4])
    if goal=='count':return (f[3]+f[4],f[1],f[2],f[0])
    return (f[0],f[1],f[2],f[3],f[4])


def normalize_hint(h,eng,keys,index,args,latecap):
    if 'transport' not in h or 'relay_records' not in h:return None
    h=json.loads(json.dumps(h));sol=h['transport'];sol['keys']=[plan(*k) for k in sol['keys']]
    try:eng.validate(sol)
    except AssertionError:return None
    lookup={k:i for i,k in enumerate(keys)}
    if any(k not in lookup for k in sol['keys']):return None
    f=objective_vector(h,eng)
    if len(sol['keys'])>args.transport_cap or len(h['relay_records'])>args.relay_cap:return None
    if args.relay_epsilon is not None and len(h['relay_records'])>args.relay_epsilon:return None
    if args.transport_epsilon is not None and len(sol['keys'])>args.transport_epsilon:return None
    if f[1]>args.horizon or (args.time_cap is not None and f[1]>args.time_cap):return None
    if args.energy_cap is not None and f[2]>args.energy_cap+1e-9:return None
    if latecap is not None and h['late_scaled']>latecap:return None
    # 每份热启动必须按当前候选范围复查所需通信区間，不能把旧可行直接当作新可行。
    for key,start in zip(sol['keys'],sol['starts']):
        for iv in index[lookup[key]]['blind_intervals']:
            lo=start+math.floor(iv['lo']+1e-9);hi=start+math.ceil(iv['hi']-1e-9)
            if not any(str(r['站点'])in set(map(str,iv['stations'])) and r['建链完成秒']<=lo and r['服务结束秒']>=hi for r in h['relay_records']):return None
    h['selected_pool_indices']=[lookup[k] for k in sol['keys']]
    return {'transport':sol,'relay_records':h['relay_records'],'selected_pool_indices':h['selected_pool_indices'],
            'late_scaled':h['late_scaled'],'late_denominator':eng.denom}


def export_snapshot(context,keys,geom,snapshot,prefix,metadata=None):
    eng=Engine(context);s=json.loads(json.dumps(snapshot));sol=s['transport']
    sol['keys']=[plan(*k) for k in sol['keys']];eng.validate(sol)
    sol['metrics']=eng.metrics(sol['keys'],sol['starts']).tolist()
    export_solution(context,sol,prefix+'_运输')
    (P/(prefix+'_运输.json')).write_text(json.dumps(sol,ensure_ascii=False,indent=2),encoding='utf-8')
    rr=sorted(s['relay_records'],key=lambda r:r['开始秒']);ufree=[0]*2;bfree=[0]*6
    stations={str(r['id']):r for r in geom['stations']}
    for num,r in enumerate(rr,1):
        st=stations[str(r['站点'])];a=r['开始秒'];ret=r['返回秒'];dd=r['服务结束秒']-r['建链完成秒']
        ee=float(st['fixed_energy_kwh'])+1.1*dd/3600
        eupper=math.ceil(st['fixed_energy_kwh']*1e6-1e-9)+math.ceil(1.1*dd/3600*1e6-1e-9)
        ch=charge_seconds(eupper/1e6)
        u=next(i for i,t in enumerate(ufree) if t<=a);b=next(i for i,t in enumerate(bfree) if t<=a)
        ufree[u]=ret+300;bfree[b]=max(r.get('组件充满秒',0),ret+ch)
        r.update({'中继架次':f'J{num:02d}','中继无人机':f'R{u+1:02d}','能源组件':f'H{b+1:02d}',
                  '经度':st['lon'],'纬度':st['lat'],'悬停海拔m':st['altitude_m'],'服务时长秒':dd,
                  '无人机可用秒':ret+300,'组件充满秒':bfree[b],'充电秒':bfree[b]-ret,
                  '能耗kWh':ee,'返航SOC%':100*(1-ee/3.2)})
        assert ee<=2.56+1e-9
    pd.DataFrame(rr).to_csv(P/(prefix+'_中继架次.csv'),index=False,encoding='utf-8-sig')
    index={r['pool_index']:r for r in geom['routes']};lookup={plan(*k):j for j,k in enumerate(keys)}
    support=[]
    for rn,(key,start) in enumerate(sorted(zip(sol['keys'],sol['starts']),key=lambda t:t[1]),1):
        j=lookup[key]
        for b,iv in enumerate(index[j]['blind_intervals']):
            lo=start+math.floor(iv['lo']+1e-9);hi=start+math.ceil(iv['hi']-1e-9)
            suitable=[r for r in rr if str(r['站点'])in set(map(str,iv['stations'])) and r['建链完成秒']<=lo and r['服务结束秒']>=hi]
            assert suitable,(prefix,j,b,lo,hi)
            r=suitable[0]
            support.append({'运输架次':f'R{rn:02d}','路线池索引':j,'盲区序号':b,'开始秒':lo,'结束秒':hi,
                            '保障方式':'保守中继预留','中继架次':r['中继架次'],'站点':r['站点']})
    pd.DataFrame(support).to_csv(P/(prefix+'_中继保障.csv'),index=False,encoding='utf-8-sig')
    f=objective_vector({'transport':sol,'relay_records':rr},eng)
    result={**(metadata or {}),'status':(metadata or {}).get('status','FEASIBLE'),
            'transport':sol,'relay_records':rr,'support_records':support,
            'selected_pool_indices':[lookup[k] for k in sol['keys']],
            'N_transport':f[3],'N_relay':f[4],'joint_finish_s':f[1],'total_energy_kwh':f[2],
            'transport_energy_kwh':sol['metrics'][2],'relay_energy_kwh':sum(r['能耗kWh'] for r in rr),
            'transport_finish_s':max(start+eng.evaluate(key).duration for key,start in zip(sol['keys'],sol['starts'])),
            'relay_finish_s':max((r['返回秒'] for r in rr),default=0),'weighted_lateness':f[0],
            'late_scaled':s['late_scaled'],'late_denominator':eng.denom,'objective_vector':f,
            'dem_sha256':context['dem_sha256'],'selected_snapshot_origin':s.get('origin','saved_snapshot')}
    if result['status'] not in ('FEASIBLE','OPTIMAL'):result['status']='FEASIBLE_FROM_PRESERVED_HINT'
    (P/(prefix+'_结果.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    return result


def solve(args):
    output=Path(args.prefix);output=output if output.is_absolute() else P/output
    if output.resolve()==(P/'问题三_V3_最终方案').resolve():
        raise ValueError('请将复算结果写入新的 --prefix，勿覆盖正式冻结方案。')
    output.parent.mkdir(parents=True,exist_ok=True)
    args.prefix=str(output)
    for name in ['geometry','pool','hint','context']:
        value=getattr(args,name,None)
        if value:
            path=Path(value);setattr(args,name,str(path if path.is_absolute() else P/path))
    context,keys,routes,eng,geom,index=load_problem(args.geometry,args.pool,args.fixed,getattr(args,'context',None))
    stations=geom['stations'];station_by_id={str(v['id']):v for v in stations}
    H=args.horizon;SCALE=1000000;M=cp_model.CpModel()
    cm=M.NewIntVar(0,H,'joint_makespan')
    X=[];S=[];ENDS=[];UI=[[]for _ in range(3)];BI=[[]for _ in range(3)]
    boxterms=[[] for _ in range(80)]
    hintdata=json.loads(Path(args.hint).read_text(encoding='utf-8')) if args.hint else None
    hint=(hintdata.get('transport',hintdata) if hintdata else
          json.loads((P/'全局认证_冻结最终十九架次.json').read_text(encoding='utf-8')))
    hintmap={plan(*k):int(s) for k,s in zip(hint['keys'],hint['starts'])}
    for j,(key,r) in enumerate(zip(keys,routes)):
        assert r is not None
        latest=min(r.latest,H-r.duration)
        xx=M.NewBoolVar(f'transport_{j}')
        ss=M.NewIntVar(0,max(0,latest),f'transport_start_{j}')
        ee=M.NewIntVar(r.duration,H+r.duration,f'transport_end_{j}')
        be=M.NewIntVar(r.duration+r.charge,H+r.duration+r.charge,f'transport_full_{j}')
        M.Add(ee==ss+r.duration);M.Add(be==ee+r.charge)
        M.Add(cm>=ee).OnlyEnforceIf(xx);M.Add(ss==0).OnlyEnforceIf(xx.Not())
        if latest<0:M.Add(xx==0)
        UI[r.g].append(M.NewOptionalIntervalVar(ss,r.duration,ee,xx,f'tu_{j}'))
        BI[r.g].append(M.NewOptionalIntervalVar(ss,r.duration+r.charge,be,xx,f'tb_{j}'))
        for b in r.boxes:boxterms[b].append(xx)
        X.append(xx);S.append(ss);ENDS.append(ee)
        M.AddHint(xx,int(key in hintmap))
        if key in hintmap:M.AddHint(ss,hintmap[key])
    for b,terms in enumerate(boxterms):
        assert terms,(b,'uncovered')
        M.AddExactlyOne(terms)
    for g,m in enumerate(context['models']):
        M.AddCumulative(UI[g],[1]*len(UI[g]),m['n_drones'])
        M.AddCumulative(BI[g],[1]*len(BI[g]),m['n_batteries'])
    M.Add(sum(X)<=args.transport_cap)
    arrivals=[];tardiness=[]
    for b in range(80):
        arrival=M.NewIntVar(0,H,f'arrival_{b}')
        for j,r in enumerate(routes):
            if b in r.delta:M.Add(arrival==S[j]+r.delta[b]).OnlyEnforceIf(X[j])
        M.Add(arrival<=min(H,int(eng.hard[b])))
        tard=M.NewIntVar(0,H,f'late_{b}');M.AddMaxEquality(tard,[0,arrival-int(eng.target[b])])
        arrivals.append(arrival);tardiness.append(tard)
    latecost=sum(int(eng.weight[b])*tardiness[b] for b in range(80))
    latecap=None
    if args.lateness_cap is not None:
        cap=Fraction(str(args.lateness_cap))*eng.denom
        latecap=cap.numerator//cap.denominator
        M.Add(latecost<=latecap)


    # 紧凑站点选择：每个中继任务槽一个站点整数变量；不复制“站点×任务槽”区间。
    missions=[];relayUI=[];relayBI=[];slots=[]
    relayhint=sorted((hintdata or {}).get('relay_records',[]),key=lambda z:z['开始秒'])
    sid_to_idx={str(st['id']):h for h,st in enumerate(stations)}
    leads=[180+int(st['outbound_s'])+30 for st in stations]
    backs=[int(st['return_s']) for st in stations]
    bases=[math.ceil(float(st['fixed_energy_kwh'])*SCALE-1e-9) for st in stations]
    assert stations and min(bases)<2560000
    maxservice=max(math.floor((2560000-v)*3600/1100000) for v in bases)
    for k in range(args.relay_cap):
        x=M.NewBoolVar(f'relay_present_{k}');slots.append(x)
        station=M.NewIntVar(0,len(stations)-1,f'relay_station_{k}')
        lead=M.NewIntVar(min(leads),max(leads),f'relay_lead_{k}')
        back=M.NewIntVar(min(backs),max(backs),f'relay_back_{k}')
        base=M.NewIntVar(min(bases),max(bases),f'relay_base_{k}')
        M.AddElement(station,leads,lead);M.AddElement(station,backs,back);M.AddElement(station,bases,base)
        a=M.NewIntVar(0,H,f'relay_start_{k}')
        ready=M.NewIntVar(0,H+max(leads),f'relay_ready_{k}')
        d=M.NewIntVar(0,maxservice,f'relay_service_duration_{k}')
        se=M.NewIntVar(0,H+max(leads)+maxservice,f'relay_service_end_{k}')
        ret=M.NewIntVar(0,H+max(leads)+maxservice+max(backs),f'relay_return_{k}')
        endu=M.NewIntVar(0,2*H+20000,f'relay_drone_free_{k}')
        full=M.NewIntVar(0,2*H+20000,f'relay_component_full_{k}')
        service_e=M.NewIntVar(0,2560000,f'relay_service_energy_{k}')
        M.AddDivisionEquality(service_e,1100000*d+3599,3600)
        e=M.NewIntVar(0,5120000,f'relay_energy_{k}');M.Add(e==base+service_e)
        M.Add(e<=2560000).OnlyEnforceIf(x)
        low=M.NewBoolVar(f'relay_low_consumption_{k}')
        M.Add(e<=320000).OnlyEnforceIf(low);M.Add(e>=320001).OnlyEnforceIf(low.Not())
        c_low=M.NewIntVar(0,11000,f'relay_charge_low_{k}')
        c_high=M.NewIntVar(0,11000,f'relay_charge_high_{k}')
        ch=M.NewIntVar(0,11000,f'relay_charge_{k}')
        M.AddDivisionEquality(c_low,63*e+31999,32000);M.AddDivisionEquality(c_high,13*e+31999,32000)
        M.Add(ch==c_low).OnlyEnforceIf(low);M.Add(ch==500+c_high).OnlyEnforceIf(low.Not())
        M.Add(ready==a+lead);M.Add(se==ready+d);M.Add(ret==se+back)
        M.Add(endu==ret+300);M.Add(full==ret+ch)
        M.Add(a==0).OnlyEnforceIf(x.Not());M.Add(d==0).OnlyEnforceIf(x.Not());M.Add(station==0).OnlyEnforceIf(x.Not())
        M.Add(d>=1).OnlyEnforceIf(x);M.Add(cm>=ret).OnlyEnforceIf(x)
        udur=M.NewIntVar(0,2*H+20000,f'relay_uduration_{k}')
        bdur=M.NewIntVar(0,2*H+20000,f'relay_bduration_{k}')
        M.Add(udur==endu-a);M.Add(bdur==full-a)
        relayUI.append(M.NewOptionalIntervalVar(a,udur,endu,x,f'ru_{k}'))
        relayBI.append(M.NewOptionalIntervalVar(a,bdur,full,x,f'rb_{k}'))
        active_e=M.NewIntVar(0,2560000,f'relay_active_energy_{k}')
        M.Add(active_e==e).OnlyEnforceIf(x);M.Add(active_e==0).OnlyEnforceIf(x.Not())
        mis=dict(slot=k,station_index=station,present=x,start=a,ready=ready,duration=d,
                 service_end=se,returned=ret,drone_free=endu,full=full,charge=ch,energy=e,active_energy=active_e)
        missions.append(mis)
        if k:M.Add(slots[k-1]>=x);M.Add(missions[k-1]['start']<=a).OnlyEnforceIf(x)
        h=relayhint[k] if k<len(relayhint) else None
        M.AddHint(x,int(h is not None));M.AddHint(station,sid_to_idx[str(h['站点'])] if h else 0)
        if h:
            for vn,hn in [('start','开始秒'),('ready','建链完成秒'),('duration','服务时长秒'),('service_end','服务结束秒'),('returned','返回秒')]:M.AddHint(mis[vn],int(h[hn]))
        if args.lock_relay_stations:
            M.Add(x==int(h is not None))
            if h:M.Add(station==sid_to_idx[str(h['站点'])])
    M.AddCumulative(relayUI,[1]*len(relayUI),2);M.AddCumulative(relayBI,[1]*len(relayBI),6)
    if args.relay_epsilon is not None:M.Add(sum(slots)<=args.relay_epsilon)
    if args.transport_epsilon is not None:M.Add(sum(X)<=args.transport_epsilon)
    assignments=[];blind_count=0
    for j,r in enumerate(routes):
        for b,iv in enumerate(index[j]['blind_intervals']):
            lo=math.floor(float(iv['lo'])+1e-9);hi=math.ceil(float(iv['hi'])-1e-9)
            if hi<=lo:continue
            blind_count+=1;allowed=[sid_to_idx[str(v)] for v in iv['stations'] if str(v)in sid_to_idx];terms=[]
            if not allowed:M.Add(X[j]==0);continue
            domain=cp_model.Domain.FromValues(allowed)
            for mi,mis in enumerate(missions):
                aa=M.NewBoolVar(f'cover_{j}_{b}_{mi}');M.Add(aa<=mis['present']);M.Add(aa<=X[j])
                M.AddLinearExpressionInDomain(mis['station_index'],domain).OnlyEnforceIf(aa)
                M.Add(mis['ready']<=S[j]+lo).OnlyEnforceIf(aa)
                M.Add(mis['service_end']>=S[j]+hi).OnlyEnforceIf(aa)
                terms.append(aa);assignments.append((aa,j,b,mi,lo,hi))
            M.Add(sum(terms)==X[j])

    energy=sum(math.ceil(r.energy*SCALE-1e-9)*x for r,x in zip(routes,X))+sum(m['active_energy'] for m in missions)
    count=sum(X)+sum(slots)
    if args.time_cap is not None:M.Add(cm<=args.time_cap)
    if args.energy_cap is not None:M.Add(energy<=int(args.energy_cap*SCALE))
    # 有界整数尺度实现字典序，无模糊权重混合。
    energy_bound=(args.transport_cap*8+args.relay_cap*3)*SCALE
    if args.goal=='time':obj=(energy_bound+100)*cm+energy
    elif args.goal=='energy':obj=(H+1)*energy+cm
    elif args.goal=='relay':obj=(energy_bound+100)*(H+1)*sum(slots)+(energy_bound+100)*cm+energy
    elif args.goal=='transport':obj=(energy_bound+100)*(H+1)*sum(X)+(energy_bound+100)*cm+energy
    elif args.goal=='late':obj=(H+1)*latecost+cm
    elif args.goal=='count':obj=(energy_bound+100)*(H+1)*count+(energy_bound+100)*cm+energy
    else:obj=energy
    M.Minimize(obj)
    err=M.Validate();assert not err,err
    print(json.dumps({'routes':len(routes),'stations':len(stations),'max_relay_missions':args.relay_cap,
                      'blind_intervals':blind_count,'cover_variables':len(assignments),
                      'vars':len(M.Proto().variables),'constraints':len(M.Proto().constraints)},ensure_ascii=False),flush=True)
    solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=args.seconds
    solver.parameters.num_search_workers=args.threads;solver.parameters.random_seed=20260923
    solver.parameters.log_search_progress=True
    trace=[];archive=[];snapshots=[]
    recoverable_path=P/(args.prefix+'_可恢复轨迹.jsonl');recoverable_path.write_text('',encoding='utf-8')
    def register(snapshot):
        f=objective_vector(snapshot,eng)
        snapshot['objective_vector']=f
        snapshots.append(snapshot)
        with recoverable_path.open('a',encoding='utf-8') as fh:fh.write(json.dumps(snapshot,ensure_ascii=False)+'\n')
        update_archive(archive,snapshot)
        (P/(args.prefix+'_非支配完整档案.json')).write_text(json.dumps(archive,ensure_ascii=False,indent=2),encoding='utf-8')
    if hintdata:
        baseline=normalize_hint(hintdata,eng,keys,index,args,latecap)
        if baseline is not None:
            baseline['origin']='provided_complete_feasible_hint';baseline['seconds']=0.0
            register(baseline)
            print('PRESERVED_FEASIBLE_HINT',json.dumps(baseline['objective_vector']),flush=True)
    class Log(cp_model.CpSolverSolutionCallback):
        def on_solution_callback(self):
            jj=[j for j,x in enumerate(X) if self.Value(x)]
            snapshot={'seconds':self.WallTime(),'transport':{'keys':[keys[j] for j in jj],
                          'starts':[self.Value(S[j]) for j in jj]},'selected_pool_indices':jj,
                      'relay_records':[],'late_scaled':self.Value(latecost),'late_denominator':eng.denom,
                      'solver_objective':self.ObjectiveValue(),'bound':self.BestObjectiveBound(),
                      'origin':'solver_incumbent'}
            for m in missions:
                if not self.Value(m['present']):continue
                st=stations[self.Value(m['station_index'])];dd=self.Value(m['duration'])
                snapshot['relay_records'].append({'站点':st['id'],'开始秒':self.Value(m['start']),
                      '建链完成秒':self.Value(m['ready']),'服务时长秒':dd,
                      '服务结束秒':self.Value(m['service_end']),'返回秒':self.Value(m['returned']),
                      '无人机可用秒':self.Value(m['drone_free']),'组件充满秒':self.Value(m['full']),
                      '能耗kWh':float(st['fixed_energy_kwh'])+1.1*dd/3600,
                      '经度':st['lon'],'纬度':st['lat'],'悬停海拔m':st['altitude_m'],
                      '充电秒':self.Value(m['charge'])})
            register(snapshot)
            z={'seconds':self.WallTime(),'objective_vector':snapshot['objective_vector'],
               'objective':self.ObjectiveValue(),'bound':self.BestObjectiveBound()}
            trace.append(z);print('JOINT_INCUMBENT',json.dumps(z),flush=True)
    status=solver.Solve(M,Log());name=solver.StatusName(status)
    report={'solver_status':name,'status':name,'scope':'finite concrete-box route pool and discrete relay candidates; hard deadlines retained; soft tardiness optimized',
            'goal':args.goal,'seconds':solver.WallTime(),'objective':solver.ObjectiveValue(),
            'bound':solver.BestObjectiveBound(),'trace':trace,'response':solver.ResponseStats(),
            'geometry_file':str(args.geometry),'pool_file':str(args.pool),'route_pool_size':len(routes),
            'station_count':len(stations),'configuration':vars(args),'lateness_denominator':eng.denom,
            'lateness_cap_scaled':latecap,'lateness_scaling_error':0,
            'energy_solver_quantization':'ceil micro-kWh; conservative, <= (NT+2*NR)*1e-6 kWh',
            'complete_snapshot_count':len(snapshots),'nondominated_snapshot_count':len(archive),
            'method':'finite-pool joint CP-SAT; no branch-price or Benders certificate'}
    if not snapshots:
        (P/(args.prefix+'_结果.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');return report
    best=min(snapshots,key=lambda a:ranking(a,eng,args.goal))
    result=export_snapshot(context,keys,geom,best,args.prefix,report)
    print('FINAL_JOINT_RESULT',json.dumps({k:result[k] for k in ['status','N_transport','N_relay','total_energy_kwh','joint_finish_s','weighted_lateness']},ensure_ascii=False),flush=True)
    return result


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--context',default=str(P/'问题二_统一实验输入.json'))
    ap.add_argument('--geometry',default=str(P/'问题三_V3_通信联合输入.json'))
    ap.add_argument('--prefix',default='recomputed/q3_joint')
    ap.add_argument('--pool',default=str(P/'问题三_V3_路线池.json'))
    ap.add_argument('--seconds',type=float,default=90);ap.add_argument('--threads',type=int,default=3)
    ap.add_argument('--horizon',type=int,default=18000);ap.add_argument('--time-cap',type=int)
    ap.add_argument('--energy-cap',type=float);ap.add_argument('--relay-cap',type=int,default=8)
    ap.add_argument('--hint')
    ap.add_argument('--relay-epsilon',type=int,help='中继架次epsilon上限；与库存6组件不同')
    ap.add_argument('--transport-epsilon',type=int,help='运输架次epsilon上限')
    ap.add_argument('--lateness-cap',type=str,help='F_late上界；精确Fraction×LCM转换；0为显式零延误切片')
    ap.add_argument('--lock-relay-stations',action='store_true',help='局部时序重算：保留hint的中继站点及任务数')
    ap.add_argument('--transport-cap',type=int,default=32);ap.add_argument('--fixed',action='store_true')
    ap.add_argument('--goal',choices=['time','energy','relay','transport','late','count'],default='time')
    solve(ap.parse_args())
