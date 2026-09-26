"""问题三 V3：仅绘最终方案的九面板地图，读取解析通信状态。

python 问题三_V3_基础绘图.py --prefix 问题三_V3_最终方案 \
    --geometry 问题三_V3_最终方案_精确 --stations 问题三_V3_站点.csv

--geometry 是包含“_通信解析区间.csv”“_通信保障.csv”及
“_连续通信复核.json”的前缀，也可直接指向通信解析区间 CSV。
不回退到旧保守状态，不显示未采用的候选路线和悬停点。
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path

# 旧公共样式模块会在导入时读取运输表；将其指向本次V3方案，避免依赖旧Q2结果。
_early=argparse.ArgumentParser(add_help=False)
_early.add_argument('--prefix',default='问题三_V3_最终方案')
_early_args,_=_early.parse_known_args()
os.environ.setdefault('Q2_CERTIFIED_PREFIX',_early_args.prefix+'_运输')

from 问题三_V2_基础绘图 import (
    P,N,CM,INK,BLUE,RED,plt,np,pd,MapStyle,nodes,draw_segments,draw_routes,
    relay_nodes,clip_intervals,Line2D,ScalarMappable,patheffects,read_dem,
)
from PIL import Image


def resolve(value):
    p=Path(value)
    return p if p.is_absolute() else P/p


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry_prefix(value,prefix):
    p=resolve(value if value else str(prefix)+'_精确')
    tail='_通信解析区间.csv'
    return Path(str(p)[:-len(tail)]) if str(p).endswith(tail) else p


def load_v3(prefix,geometry=None,stations='问题三_V3_站点.csv'):
    p=resolve(prefix);gp=geometry_prefix(geometry,p);sf=resolve(stations)
    paths={'transport':Path(str(p)+'_运输_架次.csv'),'boxes':Path(str(p)+'_运输_逐箱.csv'),
        'missions':Path(str(p)+'_中继架次.csv'),'states':Path(str(gp)+'_通信解析区间.csv'),
        'guarantee':Path(str(gp)+'_通信保障.csv'),'communication_qa':Path(str(gp)+'_连续通信复核.json'),'stations':sf}
    for k,v in paths.items():
        if not v.exists():raise FileNotFoundError(f'缺少 V3 {k} 数据：{v}')
    f=pd.read_csv(paths['transport']);b=pd.read_csv(paths['boxes']);m=pd.read_csv(paths['missions'])
    t=pd.read_csv(paths['states']);c=pd.read_csv(paths['guarantee']);s=pd.read_csv(sf)
    qa=json.loads(paths['communication_qa'].read_text(encoding='utf-8'))
    if qa.get('status')!='通过' or qa.get('failed_coverage_count',1)!=0 or qa.get('direct_priority_mismatches',1)!=0:
        raise ValueError('精确通信状态尚未通过完整核验，停止绘图')
    m=m.rename(columns={'中继架次':'relay_mission_id','中继无人机':'relay_drone','能源组件':'energy_component',
        '站点':'station_id','经度':'lon','纬度':'lat','悬停海拔m':'z','开始秒':'prep_start_s','建链完成秒':'service_start_s',
        '服务结束秒':'service_end_s','返回秒':'return_s','无人机可用秒':'drone_ready_s','组件充满秒':'component_ready_s',
        '充电秒':'charge_s','服务时长秒':'service_duration_s','能耗kWh':'energy_kwh','返航SOC%':'return_soc_pct'})
    assert len(m)>0 and m.relay_mission_id.is_unique
    if 'station_id' not in s and 'id' in s:s=s.rename(columns={'id':'station_id'})
    s.station_id=s.station_id.astype(str);m.station_id=m.station_id.astype(str)
    assert s.station_id.is_unique
    ss=s.set_index('station_id')
    assert set(m.station_id).issubset(ss.index), '最终采用中继点不在所给几何站点表内'
    m['ground_z']=m.station_id.map(ss.ground_z);m['agl']=m.z-m.ground_z
    assert m.agl.between(-1e-7,300+1e-7).all()
    for _,r in m.iterrows():
        st=ss.loc[r.station_id]
        assert abs(r.lon-st.lon)<1e-9 and abs(r.lat-st.lat)<1e-9 and abs(r.z-st.z)<1e-6
    m['takeoff_s']=m.prep_start_s+180;m['arrival_s']=m.service_start_s-30
    m['link_service_energy_kwh']=1.1*(m.service_duration_s+30)/3600
    m['transit_energy']=m.energy_kwh-m.link_service_energy_kwh
    assert len(b)==80 and b['货箱编号'].nunique()==80
    assert set(f['架次编号'])==set(t.route)
    required=['model','stage','start_s','end_s','direct_ok','relay_id','from_node','to_node']
    required += [p+'_'+v for p in ['p0','p1'] for v in ['px','py','east','north','z']]
    assert set(required).issubset(t.columns)
    t['direct_ok']=t.direct_ok.astype(str).str.lower().isin(['true','1'])
    t['relay_mission_id']=t.relay_id.fillna('').astype(str)
    t['duration_s']=t.end_s-t.start_s
    assert (t.duration_s>0).all()
    assert np.array_equal(t.direct_ok.to_numpy(),t['保障方式'].eq('直连').to_numpy())
    assert not t['保障方式'].eq('中断').any()
    assert set(t.loc[~t.direct_ok,'relay_mission_id']).issubset(set(m.relay_mission_id))
    for _,r in f.iterrows():
        q=t[t.route.eq(r['架次编号'])].sort_values('start_s')
        assert set(q.model)=={r['机型']}
        begin=r['开始秒']+300+30*len(r['货箱编号'].split(','))
        assert abs(q.start_s.iloc[0]-begin)<1e-6 and abs(q.end_s.iloc[-1]-r['返回秒'])<1e-6
        assert np.max(np.abs(q.end_s.to_numpy()[:-1]-q.start_s.to_numpy()[1:]),initial=0)<1e-6
        pairs=[]
        for pair in zip(q.from_node,q.to_node):
            if not pairs or pair!=pairs[-1]:pairs.append(pair)
        seq=r['访问顺序'].split('→')
        assert pairs==list(zip(seq[:-1],seq[1:])), '精确通信数据的路线顺序与最终运输方案不一致'
    for mode,flag in [('直连',True),('中继',False)]:
        actual=float(t.loc[t.direct_ok.eq(flag),'duration_s'].sum())
        assert abs(actual-qa['communication_seconds'][mode])<1e-6
    c=c.rename(columns={'运输架次编号':'route','通信阶段':'stage','开始时刻（s）':'start_s',
        '结束时刻（s）':'end_s','保障方式':'mode','中继架次编号':'relay_mission_id'})
    provenance={'prefix':str(p),'geometry':str(gp),'station_file':str(sf),
        'input_sha256':{k:sha(v) for k,v in paths.items()},'communication_qa':qa}
    return f,b,m,c,t,s,provenance


def windows(f,m):
    end=max(float(f['返回秒'].max()),float(m.return_s.max()))
    step=int(np.ceil(end/3/300)*300)
    return [(k*step,(k+1)*step) for k in range(3)]


def relay_height_labels(ax,m):
    """同一平面位置不同高度共享真实位置标记，逐任务列出离地高度。"""
    for (lon,lat),group in m.groupby(['lon','lat'],sort=False):
        ax.scatter(lon,lat,s=155,marker='D',c=[RED],ec='white',lw=1.2,zorder=20)
        lines=[]
        for agl,part in group.groupby('agl',sort=True):
            label='/'.join(part.relay_mission_id)
            lines.append(f'{label}  {agl:.0f} m')
        xlo,xhi=ax.get_xlim();right=lon>xlo+.72*(xhi-xlo)
        txt=ax.annotate('\n'.join(lines),(lon,lat),xytext=(-8 if right else 8,6),textcoords='offset points',
            fontsize=12.5,zorder=21,ha='right' if right else 'left',va='bottom')
        txt.set_path_effects([patheffects.withStroke(linewidth=2.5,foreground='white')])


def save(fig,path):
    path=Path(path);temp=path.with_name(path.stem+'.tmp.png')
    fig.savefig(temp,dpi=340,bbox_inches='tight',pad_inches=.08);plt.close(fig)
    with Image.open(temp) as im:im.verify()
    with Image.open(temp) as im:im.load()
    temp.replace(path);print(path.name,flush=True)


def geography(f,m,t,s,out):
    sty=MapStyle(m);t=sty.coordinates(t)
    covered=t[~t.direct_ok].copy()
    covered['relay_drone']=covered.relay_mission_id.map(m.set_index('relay_mission_id').relay_drone)
    fig=plt.figure(figsize=(21.8,17.5))
    gs=fig.add_gridspec(3,3,left=.054,right=.985,bottom=.116,top=.970,wspace=.12,hspace=.16)
    periods=windows(f,m)
    titles=['运输任务与工作地形','直连与中继通信保障','悬停点与离地高度','回传链路与中继部署',
        'R01 保障的运输航段','R02 保障的运输航段']+[f'{lo//60}—{hi//60} 分钟通信保障' for lo,hi in periods]
    axs=[fig.add_subplot(gs[i//3,i%3]) for i in range(9)]
    for i,ax in enumerate(axs):sty.base(ax,i,titles[i])
    draw_routes(axs[0],f);nodes(axs[0]);sty.north(axs[0])
    draw_segments(axs[1],t[t.direct_ok],BLUE,2,.72);draw_segments(axs[1],covered,RED,3.2,.95);nodes(axs[1])
    draw_routes(axs[2],f);nodes(axs[2],labels=False);relay_height_labels(axs[2],m)
    draw_routes(axs[3],f);relay_nodes(axs[3],m,backhaul=True);nodes(axs[3])
    for k,drone in enumerate(['R01','R02']):
        draw_routes(axs[4+k],f);draw_segments(axs[4+k],covered[covered.relay_drone.eq(drone)],RED,3.3)
        relay_nodes(axs[4+k],m[m.relay_drone.eq(drone)]);nodes(axs[4+k])
    for k,(lo,hi) in enumerate(periods):
        seg=clip_intervals(t,lo,hi)
        draw_segments(axs[6+k],seg[seg.direct_ok],BLUE,2,.68)
        draw_segments(axs[6+k],seg[~seg.direct_ok],RED,3,.98)
        active=m[(m.service_start_s<hi)&(m.service_end_s>lo)]
        relay_nodes(axs[6+k],active,backhaul=True);nodes(axs[6+k])
    handles=[Line2D([],[],color='#656565',lw=1.3,label='运输航线'),Line2D([],[],color=BLUE,lw=3,label='直连保障'),
        Line2D([],[],color=RED,lw=3,label='中继保障'),Line2D([],[],color=INK,lw=2,ls='--',label='中继回传'),
        Line2D([],[],ls='',marker='D',mfc=RED,mec=INK,ms=9,label='采用悬停点'),
        Line2D([],[],ls='',marker='*',mfc=INK,mec='white',ms=13,label='中心与基站')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.007),ncol=6,frameon=False,fontsize=14.5,columnspacing=1.6)
    cb=fig.colorbar(ScalarMappable(norm=sty.norm,cmap=CM),cax=fig.add_axes([.33,.060,.34,.009]),orientation='horizontal')
    cb.set_ticks([200,400,600,800,1000]);cb.ax.tick_params(labelsize=12,width=1,length=3,pad=2);cb.outline.set_linewidth(1)
    fig.text(.695,.063,'高程（m）',fontsize=14,ha='left',va='center')
    save(fig,str(out)+'_图01_九面板通信空间图.png')
    return {'windows_s':periods,'candidate_pool_count_not_drawn':len(s),'displayed_hover_locations':int(m[['lon','lat']].drop_duplicates().shape[0]),
        'displayed_hover_states':int(m[['lon','lat','z']].drop_duplicates().shape[0]),
        'relay_heights_agl_m':sorted(float(x) for x in np.unique(np.round(m.agl,7))),'shared_elevation_color_range_m':[sty.norm.vmin,sty.norm.vmax]}


def parser():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prefix',default='问题三_V3_最终方案')
    ap.add_argument('--geometry',default=None)
    ap.add_argument('--stations',default='问题三_V3_站点.csv')
    ap.add_argument('--output-prefix',default='问题三_V3')
    return ap


def main():
    a=parser().parse_args();f,b,m,c,t,s,provenance=load_v3(a.prefix,a.geometry,a.stations)
    out=resolve(a.output_prefix);qa=geography(f,m,t,s,out)
    qa.update({'transport_sorties':len(f),'relay_sorties':len(m),'boxes':len(b),
        'direct_seconds':float(t.loc[t.direct_ok,'duration_s'].sum()),'relay_seconds':float(t.loc[~t.direct_ok,'duration_s'].sum()),
        'communication_scope':'解析遮挡和距离事件恢复直连优先；边界瞬时单独记录，不计入累计时长',**provenance})
    Path(str(out)+'_地图绘图核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
