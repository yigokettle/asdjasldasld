"""原表读取和独立物理评价内核；沿用原始连续时间模型，用于复核整数调度。"""
from pathlib import Path
import math, sys, itertools
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'q1_flat' if (ROOT / 'q1_flat').is_dir() else ROOT
sys.path.insert(0, str(DATA))
import 精确算法 as q1


def read_inputs():
    node_raw = pd.read_excel(DATA/'调度中心与服务区.xlsx', sheet_name='数据', header=None)
    node_rows = []
    for row in node_raw.itertuples(index=False, name=None):
        if isinstance(row[0], str) and (row[0]=='O01' or row[0].startswith('S0')):
            node_rows.append(dict(node_id=row[0], longitude_deg=float(row[2]),
                latitude_deg=float(row[3]), ground_elevation_m=float(row[4]),
                operating_altitude_m=float(row[4]) + (0 if row[0]=='O01' else 30)))
    nodes = pd.DataFrame(node_rows).set_index('node_id')
    box_data = pd.read_excel(DATA/'物资需求与配送时限.xlsx',sheet_name='逐箱货箱清单')
    rename = {'货箱编号':'box_id','服务区编号':'node_id','物资类型':'material_type',
        '单箱质量（kg）':'mass_kg','单箱体积（m³）':'volume_m3','是否首批保障':'first_batch',
        '首批截止时间（s）':'first_deadline_s','期望送达时间（s）':'target_delivery_s',
        '应急优先系数':'priority'}
    boxes = box_data.rename(columns=rename).set_index('box_id')
    boxes['hard_deadline_s'] = np.where(
        boxes.first_batch.eq('是'), boxes.first_deadline_s,
        np.where(boxes.material_type.eq('医疗物资'), boxes.target_delivery_s, np.inf))
    model_raw = pd.read_excel(DATA/'运输无人机数据.xlsx', sheet_name='数据', header=None)
    cols = ['model_id','model_name','empty_mass_kg','max_payload_kg','max_volume_m3',
        'cruise_speed_m_s','empty_range_m','full_range_m','battery_kwh','reserve_percent',
        'fixed_prep_s','load_per_box_s','handoff_base_s','handoff_per_box_s',
        'climb_speed_m_s','descent_speed_m_s','climb_efficiency','descent_efficiency']
    models = pd.DataFrame(model_raw.iloc[2:5,:18].to_numpy(),columns=cols)
    models[cols[2:]]=models[cols[2:]].apply(pd.to_numeric)
    models=models.set_index('model_id')
    drones={str(row[0]):dict(model=str(row[1]),ready=0.) for row in model_raw.iloc[8:16,:3].itertuples(index=False,name=None)}
    batteries={}
    for row in model_raw.iloc[19:22,:3].itertuples(index=False,name=None):
        model,n,charge = str(row[0]),int(row[1]),float(row[2])
        for i in range(1,n+1):
            batteries[f'{model}-B{i:02d}']=dict(model=model,ready=0.,full_charge_s=charge)
    assert len(boxes)==80 and len(drones)==8 and len(batteries)==14
    return nodes,boxes,models,drones,batteries


def pairwise_legs(nodes):
    dem,left,top,dx,dy=q1.read_dem(DATA/'最终工作DEM.tif')
    coords={s:((float(n.longitude_deg)-left)/dx,(top-float(n.latitude_deg))/dy,
        q1.utm49(float(n.longitude_deg),float(n.latitude_deg))) for s,n in nodes.iterrows()}
    legs={}
    for u,v in itertools.permutations(nodes.index,2):
        x0,y0,a=coords[u];x1,y1,b=coords[v]
        touched=q1.touched_cells(x0,y0,x1,y1,dem.shape[1],dem.shape[0])
        assert touched, (u,v)
        peak=max(float(dem[row,col]) for row,col in touched)
        altitude=peak+50.
        legs[u,v]=dict(distance_m=math.dist(a,b),peak_m=peak,cruise_altitude_m=altitude,
            climb_m=max(0.,altitude-float(nodes.loc[u,'operating_altitude_m'])),
            descent_m=max(0.,altitude-float(nodes.loc[v,'operating_altitude_m'])))
    return legs


def charge_seconds(soc,full):
    assert -1e-8<=soc<=1+1e-8
    soc=float(np.clip(soc,0,1))
    return full*(.65*(.9-soc)/.9+.35) if soc<.9 else full*.35*(1-soc)/.1


def route(stops,alloc,model_id,boxes,models,legs,at=0.):
    model=models.loc[model_id]
    allids=[b for site in stops for b in alloc[site]]
    assert len(allids)==len(set(allids)) and len(set(stops))==len(stops)
    mass=float(boxes.loc[allids,'mass_kg'].sum())
    volume=float(boxes.loc[allids,'volume_m3'].sum())
    if mass>model.max_payload_kg+1e-8 or volume>model.max_volume_m3+1e-8:return None
    elapsed=float(model.fixed_prep_s+model.load_per_box_s*len(allids))
    remaining=mass;energy=0.;src='O01';arrivals={};leg_rows=[]
    for dst in (*stops,'O01'):
        leg=legs[src,dst]
        d=float(leg['distance_m']);qmax=float(model.max_payload_kg)
        effective_range=float(model.empty_range_m-(model.empty_range_m-model.full_range_m)
            *(remaining/qmax)**1.5)
        flight=d/model.cruise_speed_m_s+leg['climb_m']/model.climb_speed_m_s+leg['descent_m']/model.descent_speed_m_s
        horizontal=float(model.battery_kwh)*d/effective_range
        climb=(float(model.empty_mass_kg)+remaining)*9.80665*leg['climb_m']/(3.6e6*model.climb_efficiency)
        energy+=horizontal+climb
        elapsed+=flight
        leg_rows.append(dict(from_node=src,to_node=dst,remaining_payload_kg=remaining,
            distance_m=d,cruise_altitude_m=leg['cruise_altitude_m'],energy_kwh=horizontal+climb,
            flight_s=flight))
        if dst!='O01':
            elapsed+=model.handoff_base_s+model.handoff_per_box_s*len(alloc[dst])
            for bid in alloc[dst]:
                if at+elapsed>float(boxes.loc[bid,'hard_deadline_s'])+1e-7:return None
                arrivals[bid]=elapsed
            remaining-=float(boxes.loc[alloc[dst],'mass_kg'].sum())
        src=dst
    if energy > model.battery_kwh*(1-model.reserve_percent/100)+1e-8:return None
    return dict(stops=tuple(stops), boxes=tuple(allids), deliveries=arrivals,
        mass_kg=mass,volume_m3=volume,energy_kwh=energy,duration_s=elapsed,
        remaining_soc=1-energy/float(model.battery_kwh),leg_rows=leg_rows)
