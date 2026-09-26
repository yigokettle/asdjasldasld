"""问题三静态学术图：真实通信几何、联合调度及中继资源记录。

运行示例：
python 问题三_结果绘图.py --transport-prefix 问题三_联合方案 \
    --relay-prefix 问题三_联合方案 --geometry-prefix q3_geometry

不构造示意最优解或仿真数据。地形和字体直接继承问题二最终图件。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from 认证更新_地图剖面公用 import (
    P, N, B, CM, read_dem, gradient_line, gradient_patch, gradient_fill,
    GradientKey, plt, np, pd, patheffects,
)
from matplotlib.colors import LightSource, Normalize
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

INK = '#171717'
BLUE, RED = CM(.075), CM(.96)
MODEL_COLOR = {'A': CM(.10), 'B': CM(.75), 'C': CM(.96)}
OFFSETS = {'S001':(6,7),'S002':(6,6),'S003':(6,-16),'S004':(7,5),
    'S005':(-8,7),'S006':(-5,-17),'S007':(-6,-17),'S008':(-8,7),
    'S009':(-6,-17),'S010':(6,3),'S011':(-12,7),'S012':(-3,7),
    'S013':(5,7),'S014':(-6,-17),'S015':(-8,7)}
COORD = {s:np.array([r.longitude_deg,r.latitude_deg]) for s,r in N.iterrows()}


def resolve(s):
    p = Path(s)
    return p if p.is_absolute() else P/p


def as_bool(series):
    return series.astype(str).str.lower().isin(['true','1','1.0'])


def load_inputs(transport_prefix, relay_prefix, geometry_prefix):
    tp, rp, gp = map(resolve, [transport_prefix,relay_prefix,geometry_prefix])
    transport = pd.read_csv(str(tp)+'_架次.csv')
    boxes = pd.read_csv(str(tp)+'_逐箱.csv')
    missions = pd.read_csv(str(rp)+'_中继架次.csv')
    guarantee = pd.read_csv(str(rp)+'_通信保障.csv')
    checked_trajectory=Path(str(rp)+'_通信区间复核.csv')
    trajectory = pd.read_csv(checked_trajectory if checked_trajectory.exists() else str(gp)+'_intervals.csv')
    stations = pd.read_csv(str(gp)+'_stations.csv')
    report_file=next((f for f in [Path(str(rp)+'_完整核验.json'),Path(str(rp)+'_结果.json'),Path(str(rp)+'.json')] if f.exists()))
    report = json.loads(report_file.read_text(encoding='utf-8'))
    missions=missions.rename(columns={'中继架次':'relay_mission_id','中继无人机':'relay_drone','能源组件':'energy_component',
        '站点':'station_id','经度':'lon','纬度':'lat','悬停海拔m':'z','开始秒':'prep_start_s','建链完成秒':'service_start_s',
        '服务结束秒':'service_end_s','返回秒':'return_s','无人机可用秒':'drone_ready_s','组件充满秒':'component_ready_s',
        '充电秒':'charge_s','服务时长秒':'service_duration_s','能耗kWh':'energy_kwh','返航SOC%':'return_soc_pct'})
    if 'takeoff_s' not in missions:missions['takeoff_s']=missions.prep_start_s+180
    if 'arrival_s' not in missions:missions['arrival_s']=missions.service_start_s-30
    if 'link_service_energy_kwh' not in missions:
        missions['link_service_energy_kwh']=1.1*(missions.service_duration_s+30)/3600
        missions['transit_energy']=missions.energy_kwh-missions.link_service_energy_kwh
    guarantee=guarantee.rename(columns={'运输架次编号':'route','通信阶段':'stage','开始时刻（s）':'start_s',
        '结束时刻（s）':'end_s','保障方式':'mode','中继架次编号':'relay_mission_id'})
    assert len(boxes)==80 and boxes['货箱编号'].nunique()==80
    assert set(transport['架次编号']) == set(trajectory['route'])
    assert len(missions)>0, '没有中继架次；此绘图脚本不补造任务'
    for _,r in transport.iterrows():
        frame=trajectory[trajectory.route.eq(r['架次编号'])].sort_values('relative_start_s')
        actual=[]
        for pair in zip(frame.from_node,frame.to_node):
            if not actual or pair!=actual[-1]:actual.append(pair)
        visit=r['访问顺序'].split('→')
        assert actual==list(zip(visit[:-1],visit[1:])), '运输路线变更后必须重算通信几何'
        assert abs(frame.relative_end_s.max()-(r['返回秒']-r['开始秒']))<1e-5, '任务长度与通信几何不一致'
    trajectory['direct_ok'] = as_bool(trajectory.direct_ok)
    starts = transport.set_index('架次编号')['开始秒']
    # 几何列与路线相同而联合解重新安排开始时刻时，只平移时轴。
    trajectory['start_s'] = trajectory.relative_start_s + trajectory.route.map(starts)
    trajectory['end_s'] = trajectory.relative_end_s + trajectory.route.map(starts)
    trajectory['duration_s'] = trajectory.end_s - trajectory.start_s
    assert (trajectory.duration_s>0).all()
    # 最终核验表在任务与服务事件处重新分段；按该表恢复真正使用的保障方式。
    if 'mode' in guarantee:
        raw=trajectory
        split=[]
        for _,g in guarantee.iterrows():
            w=clip_intervals(raw[raw.route.eq(g.route)],g.start_s,g.end_s)
            w['direct_ok']=g['mode']=='直连';w['relay_mission_id']=g.relay_mission_id
            split.append(w)
        trajectory=pd.concat(split,ignore_index=True)
        assert abs(trajectory.duration_s.sum()-raw.duration_s.sum())<1e-5
        guarantee=guarantee[guarantee['mode'].eq('中继')].copy()
    return transport,boxes,missions,guarantee,trajectory,stations,report


class MapStyle:
    def __init__(self, missions):
        z,self.left,self.top,self.dx,self.dy = read_dem(P/'q1_flat'/'最终工作DEM.tif')
        lo = np.r_[N.longitude_deg.to_numpy(), missions.lon.to_numpy()]
        la = np.r_[N.latitude_deg.to_numpy(), missions.lat.to_numpy()]
        self.ext = [lo.min()-.009,lo.max()+.009,la.min()-.008,la.max()+.008]
        self.aspect = 1/np.cos(np.deg2rad(N.latitude_deg.mean()))
        x0,x1,y0,y1 = self.ext
        r0=max(0,int((self.top-y1)/self.dy));r1=min(z.shape[0],int((self.top-y0)/self.dy)+2)
        c0=max(0,int((x0-self.left)/self.dx));c1=min(z.shape[1],int((x1-self.left)/self.dx)+2)
        zz=z[r0:r1,c0:c1]
        self.zext=[self.left+c0*self.dx,self.left+c1*self.dx,self.top-r1*self.dy,self.top-r0*self.dy]
        shade=LightSource(azdeg=305,altdeg=54).hillshade(zz,dx=30,dy=30)
        self.norm=Normalize(float(np.nanmin(z)),float(np.nanmax(z)))
        norm=self.norm
        terrain=CM(norm(zz))[:,:,:3]*(.85+.15*shade[:,:,None])
        self.rgb=.54+.46*terrain

    def base(self,ax,index,title):
        ax.imshow(self.rgb,extent=self.zext,origin='upper',interpolation='bilinear',zorder=0)
        ax.set_xlim(self.ext[:2]);ax.set_ylim(self.ext[2:]);ax.set_aspect(self.aspect)
        ax.set_xticks([109.16,109.20,109.24,109.28]);ax.set_yticks([23.00,23.04,23.08])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=12.8,pad=3.5,width=1.15,length=3.6)
        for spine in ax.spines.values():spine.set_linewidth(1.4);spine.set_color(INK)
        ax.set_title(f'{chr(97+index)}  {title}',loc='left',fontsize=19,pad=8)
        if index%3==0:ax.set_ylabel('纬度（°）',fontsize=16,labelpad=2)
        if index//3==2:ax.set_xlabel('经度（°）',fontsize=16,labelpad=3)

    def coordinates(self, trajectory):
        t=trajectory.copy()
        for p in ['p0','p1']:
            t[p+'_lon']=self.left+t[p+'_px']*self.dx
            t[p+'_lat']=self.top-t[p+'_py']*self.dy
        return t

    def north(self,ax):
        ax.annotate('',xy=(.085,.90),xytext=(.085,.79),xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-|>',color='black',lw=1.3,mutation_scale=17))
        ax.text(.085,.925,'北',ha='center',transform=ax.transAxes,fontsize=13)
        x=self.ext[0]+.005;y=self.ext[2]+.004
        d=2000/(111320*np.cos(np.deg2rad(N.latitude_deg.mean())))
        ax.plot([x,x+d],[y,y],color=INK,lw=2.1,zorder=20)
        for a in [x,x+d]:ax.plot([a,a],[y-.0005,y+.0005],color=INK,lw=1.2)
        ax.text(x+d/2,y+.0012,'2 km',ha='center',fontsize=11.5)


def nodes(ax,labels=True):
    for s,xy in COORD.items():
        if s=='O01':
            ax.scatter(*xy,s=240,marker='*',c=INK,ec='white',lw=1,zorder=18)
            text=ax.annotate('O01 / G01',xy,xytext=(0,-17),textcoords='offset points',ha='center',fontsize=12.5,zorder=19)
        else:
            ax.scatter(*xy,s=48,c='white',ec=INK,lw=.85,zorder=12)
            if not labels:continue
            ox,oy=OFFSETS[s]
            text=ax.annotate(s,xy,xytext=(ox,oy),textcoords='offset points',ha='right' if ox<0 else 'left',va='bottom',fontsize=12.5,zorder=15)
        text.set_path_effects([patheffects.withStroke(linewidth=2,foreground='white')])


def draw_segments(ax,df,color=BLUE,lw=2.3,alpha=.95,zorder=4):
    if not len(df):return
    a=df[['p0_lon','p0_lat']].to_numpy();b=df[['p1_lon','p1_lat']].to_numpy()
    move=np.linalg.norm(a-b,axis=1)>1e-11
    if move.any():
        colors=color if isinstance(color,str) or np.ndim(color)==1 else np.asarray(color)[move]
        ax.add_collection(LineCollection(np.stack([a[move],b[move]],axis=1),colors=colors,lw=lw,alpha=alpha,zorder=zorder))
    fixed=~move
    if fixed.any():
        xy=np.unique(np.round(a[fixed],9),axis=0)
        ax.scatter(xy[:,0],xy[:,1],s=23,c=[color] if np.ndim(color)==1 and not isinstance(color,str) else color,alpha=alpha,zorder=zorder)


def draw_routes(ax,transport):
    for _,r in transport.iterrows():
        xy=np.array([COORD[s] for s in r['访问顺序'].split('→')])
        ax.plot(xy[:,0],xy[:,1],color='#656565',lw=1.25,alpha=.55,zorder=2)


def relay_nodes(ax,missions,backhaul=False,labels=True):
    for i,((station,lon,lat),frame) in enumerate(missions.groupby(['station_id','lon','lat'],sort=False)):
        color=CM(.92)
        if backhaul:
            origin=COORD['O01']
            ax.plot([origin[0],lon],[origin[1],lat],color=INK,lw=1.9,ls='--',zorder=6)
        ax.scatter(lon,lat,s=155,marker='D',c=[color],ec='white',lw=1.2,zorder=20)
        if labels:
            label='/'.join(frame.relay_drone.drop_duplicates())
            text=ax.annotate(label,(lon,lat),xytext=(9,3),textcoords='offset points',fontsize=13.5,fontweight='bold',zorder=21)
            text.set_path_effects([patheffects.withStroke(linewidth=2.4,foreground='white')])


def clip_intervals(t,lo,hi):
    w=t[(t.start_s<hi)&(t.end_s>lo)].copy()
    if not len(w):return w
    a=((lo-w.start_s)/w.duration_s).clip(0,1).to_numpy()
    b=((hi-w.start_s)/w.duration_s).clip(0,1).to_numpy()
    for q in ['px','py','east','north','lon','lat','z']:
        if 'p0_'+q not in w:continue
        p0=w['p0_'+q].to_numpy();delta=w['p1_'+q].to_numpy()-p0
        w['p0_'+q]=p0+a*delta;w['p1_'+q]=p0+b*delta
    w['start_s']=w.start_s.clip(lower=lo);w['end_s']=w.end_s.clip(upper=hi)
    w['duration_s']=w.end_s-w.start_s
    return w


def match_guarantee(trajectory, guarantee):
    """以时间相交取保障子段，显示范围不向外延伸。"""
    pieces=[]
    for _,g in guarantee.iterrows():
        t=trajectory[(trajectory.route==g.route)&(~trajectory.direct_ok)]
        p=clip_intervals(t,float(g.start_s),float(g.end_s))
        if len(p):p=p.assign(relay_mission_id=g.relay_mission_id);pieces.append(p)
    return pd.concat(pieces,ignore_index=True) if pieces else trajectory.iloc[:0].assign(relay_mission_id='')


def save(fig,path,dpi=340):
    path=Path(path);temp=path.with_name(path.stem+'.tmp.png')
    fig.savefig(temp,dpi=dpi,bbox_inches='tight',pad_inches=.08)
    plt.close(fig);temp.replace(path);print(path.name,flush=True)


def geography(transport,missions,guarantee,trajectory,stations,out):
    sty=MapStyle(missions);t=sty.coordinates(trajectory)
    covered=match_guarantee(t,guarantee)
    covered['relay_drone']=covered.relay_mission_id.map(missions.set_index('relay_mission_id').relay_drone)
    relay_ids=sorted(missions.relay_drone.unique())
    fig=plt.figure(figsize=(21.8,17.5))
    gs=fig.add_gridspec(3,3,left=.054,right=.985,bottom=.116,top=.970,wspace=.12,hspace=.16)
    end=float(max(transport['返回秒'].max(),missions.return_s.max()))
    step=int(np.ceil(end/3/300)*300)
    windows=[(k*step,(k+1)*step) for k in range(3)]
    titles=['运输任务与工作地形','直连保障与中继需求','候选悬停点与采用位置','回传链路与中继部署',
        'R01 保障的运输航段','R02 保障的运输航段']+[f'{lo//60}—{hi//60} 分钟通信保障' for lo,hi in windows]
    axs=[fig.add_subplot(gs[i//3,i%3]) for i in range(9)]
    for i,ax in enumerate(axs):sty.base(ax,i,titles[i])
    draw_routes(axs[0],transport);nodes(axs[0]);sty.north(axs[0])
    draw_segments(axs[1],t[t.direct_ok],BLUE,2.0,.72)
    draw_segments(axs[1],t[~t.direct_ok],RED,3.2,.95);nodes(axs[1])
    draw_routes(axs[2],transport)
    axs[2].scatter(stations.lon,stations.lat,s=28,c=[CM(.30)],ec=INK,lw=.35,zorder=3)
    relay_nodes(axs[2],missions,labels=False);nodes(axs[2])
    draw_routes(axs[3],transport);relay_nodes(axs[3],missions,backhaul=True);nodes(axs[3])
    for k,drone in enumerate(['R01','R02']):
        draw_routes(axs[4+k],transport)
        draw_segments(axs[4+k],covered[covered.relay_drone==drone],RED,3.3)
        relay_nodes(axs[4+k],missions[missions.relay_drone==drone]);nodes(axs[4+k])
    for k,(lo,hi) in enumerate(windows):
        seg=clip_intervals(t,lo,hi)
        draw_segments(axs[6+k],seg[seg.direct_ok],BLUE,2,.68)
        draw_segments(axs[6+k],seg[~seg.direct_ok],RED,3,.98)
        m=missions[(missions.service_start_s<hi)&(missions.service_end_s>lo)]
        relay_nodes(axs[6+k],m,backhaul=True);nodes(axs[6+k])
    handles=[Line2D([],[],color='#656565',lw=1.3,label='运输航线'),Line2D([],[],color=BLUE,lw=3,label='直连保障'),Line2D([],[],color=RED,lw=3,label='中继保障'),
        Line2D([],[],color=INK,lw=2,ls='--',label='中继回传'),
        Line2D([],[],ls='',marker='D',mfc=RED,mec=INK,ms=9,label='采用悬停点'),
        Line2D([],[],ls='',marker='*',mfc=INK,mec='white',ms=13,label='中心与基站')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.007),ncol=6,frameon=False,fontsize=14.5,columnspacing=1.6)
    cax=fig.add_axes([.33,.060,.34,.009])
    cb=fig.colorbar(ScalarMappable(norm=sty.norm,cmap=CM),cax=cax,orientation='horizontal')
    cb.set_ticks([0,200,400,600,800,1000] if sty.norm.vmin<=0 else [200,400,600,800,1000])
    cb.ax.tick_params(labelsize=12,width=1,length=3,pad=2);cb.outline.set_linewidth(1)
    fig.text(.695,.063,'高程（m）',fontsize=14,ha='left',va='center')
    save(fig,str(out)+'_图01_九面板通信空间图.png')
    return {'windows_s':windows,'map_candidates':len(stations),'adopted_stations':int(missions.station_id.nunique())}


def panel(ax,k,title):
    ax.set_title(f'{chr(97+k)}  {title}',loc='left',fontsize=19,pad=10)
    ax.spines[['top','right']].set_visible(False)
    for sp in ax.spines.values():sp.set_linewidth(1.4)
    ax.tick_params(labelsize=13,width=1.2,length=4)


def bar_gradient(ax,left,width,y,height,lo,hi,hatch=None):
    patch=Rectangle((left,y-height/2),width,height,ec=INK,lw=.7,fc='none',hatch=hatch,zorder=3)
    ax.add_patch(patch)
    gradient_patch(ax,patch,lo,hi,orientation='horizontal',alpha=.95,zorder=2.8)
    return patch


def resource(transport,boxes,missions,guarantee,trajectory,out):
    fig,axs=plt.subplots(2,3,figsize=(20.8,10.5))
    fig.subplots_adjust(left=.060,right=.986,bottom=.110,top=.955,wspace=.24,hspace=.36)
    ax=axs.flat[0];panel(ax,0,'中继无人机任务与周转')
    end=max(missions.drone_ready_s.max(),missions.component_ready_s.max())/60
    ax.set_xlim(0,np.ceil(end/10)*10);ax.set_ylim(-.6,1.6)
    # 准备、飞行/建链、服务、返航、周转均有真实起止时刻。
    for _,r in missions.iterrows():
        y=['R01','R02'].index(r.relay_drone)
        intervals=[(r.prep_start_s,r.takeoff_s,.04,.13),(r.takeoff_s,r.service_start_s,.19,.38),
            (r.service_start_s,r.service_end_s,.69,.96),(r.service_end_s,r.return_s,.19,.38),
            (r.return_s,r.drone_ready_s,.40,.48)]
        for a,b,lo,hi in intervals:bar_gradient(ax,a/60,(b-a)/60,y,.43,lo,hi)
        ax.text((r.service_start_s+r.service_end_s)/120,y,r.relay_mission_id,ha='center',va='center',fontsize=13,color='black',zorder=5)
    ax.set_yticks([0,1],['R01','R02']);ax.set_xlabel('时间（min）');ax.invert_yaxis()
    ax=axs.flat[1];panel(ax,1,'能源组件占用与充电')
    comp=sorted(missions.energy_component.unique())
    ax.set_xlim(0,np.ceil(end/10)*10);ax.set_ylim(-.6,len(comp)-.4)
    for _,r in missions.iterrows():
        y=comp.index(r.energy_component)
        bar_gradient(ax,r.prep_start_s/60,(r.return_s-r.prep_start_s)/60,y,.52,.70,.96)
        bar_gradient(ax,r.return_s/60,r.charge_s/60,y,.52,.04,.32,hatch='///')
        ax.text((r.prep_start_s+r.return_s)/120,y,r.relay_mission_id,ha='center',va='center',fontsize=13,zorder=5)
    ax.set_yticks(range(len(comp)),comp);ax.set_xlabel('时间（min）');ax.invert_yaxis()
    ax=axs.flat[2];panel(ax,2,'中继架次能耗组成')
    m=missions.sort_values('relay_mission_id');xx=np.arange(len(m))
    for n,(_,r) in enumerate(m.iterrows()):
        a=ax.bar(n,r.transit_energy,width=.63,ec=INK,lw=.8,zorder=3)[0]
        b=ax.bar(n,r.link_service_energy_kwh,bottom=r.transit_energy,width=.63,ec=INK,lw=.8,zorder=3)[0]
        gradient_patch(ax,a,.05,.32);gradient_patch(ax,b,.65,.97)
        ax.text(n,r.energy_kwh+.055,f'{r.energy_kwh:.2f}',ha='center',fontsize=12)
    ax.axhline(2.56,c=INK,ls='--',lw=1.2)
    ax.set_xlim(-.7,len(m)-.3);ax.set_ylim(0,max(2.8,float(m.energy_kwh.max())*1.15));ax.set_xticks(xx,m.relay_mission_id);ax.set_ylabel('能耗（kWh）')
    ax.legend(handles=[GradientKey('往返飞行',.05,.32),GradientKey('建链与悬停',.65,.97)],loc='upper left',ncol=2,frameon=False,fontsize=11)
    ax=axs.flat[3];panel(ax,3,'返航能源余量')
    ax.set_xlim(-.7,len(m)-.3);ax.set_ylim(0,min(100,max(45,float(m.return_soc_pct.max())*1.13)))
    for n,(_,r) in enumerate(m.iterrows()):
        b=ax.bar(n,r.return_soc_pct,width=.60,ec=INK,lw=.8,zorder=3)[0];gradient_patch(ax,b,.04,.93)
        ax.scatter(n,r.return_soc_pct,c=INK,s=25,zorder=4)
        ax.text(n,r.return_soc_pct+1.2,f'{r.return_soc_pct:.1f}%',ha='center',fontsize=12)
    ax.axhline(20,c=RED,ls='--',lw=1.5);ax.set_xticks(xx,m.relay_mission_id);ax.set_ylabel('返航 SOC（%）')
    ax=axs.flat[4];panel(ax,4,'运输架次通信保障组成')
    ct=trajectory.groupby(['route','direct_ok']).duration_s.sum().unstack(fill_value=0)/60
    ct=ct.reindex(sorted(ct.index));yy=np.arange(len(ct))
    ax.set_ylim(-.65,len(ct)-.35);ax.set_xlim(0,ct.sum(axis=1).max()*1.06)
    for j,(route,row) in enumerate(ct.iterrows()):
        direct=float(row.get(True,0));relay=float(row.get(False,0))
        if direct:bar_gradient(ax,0,direct,j,.68,.04,.30)
        if relay:bar_gradient(ax,direct,relay,j,.68,.68,.97)
    ax.set_yticks(yy,ct.index,fontsize=11.5);ax.set_xlabel('累计通信保障时间（min）');ax.invert_yaxis()
    ax.legend(handles=[GradientKey('直连',.04,.30),GradientKey('中继',.68,.97)],loc='upper right',ncol=2,frameon=False,fontsize=11)
    ax=axs.flat[5];panel(ax,5,'运输与通信联合完成时间')
    values=[float(boxes['送达秒'].max()),float(transport['返回秒'].max()),float(missions.return_s.max())]
    labels=['全部物资送达','运输机全部返航','中继机全部返航']
    ax.set_xlim(0,max(values)/60*1.16);ax.set_ylim(-.7,2.7)
    for j,v in enumerate(values):
        bar_gradient(ax,0,v/60,j,.56,.05,.94)
        ax.text(v/60+1.5,j,f'{v/60:.1f}',va='center',fontsize=14)
    ax.set_yticks(range(3),labels,fontsize=13);ax.set_xlabel('时间（min）');ax.invert_yaxis()
    fig.legend(handles=[GradientKey('准备／飞行',.04,.38),GradientKey('中继服务',.69,.96),GradientKey('地面周转',.40,.48),GradientKey('返航后充电',.04,.32,hatch='///')],
        ncol=4,loc='lower center',bbox_to_anchor=(.5,.005),frameon=False,fontsize=14.5,columnspacing=2)
    save(fig,str(out)+'_图02_中继资源与通信保障.png')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--transport-prefix',default='问题三_冻结推荐方案_运输')
    ap.add_argument('--relay-prefix',default='问题三_冻结推荐方案')
    ap.add_argument('--geometry-prefix',default='q3_geometry')
    ap.add_argument('--output-prefix',default='问题三_V2')
    args=ap.parse_args()
    f,b,m,c,t,s,result=load_inputs(args.transport_prefix,args.relay_prefix,args.geometry_prefix)
    out=resolve(args.output_prefix)
    geography_qa=geography(f,m,c,t,s,out)
    resource(f,b,m,c,t,out)
    qa={'transport_prefix':args.transport_prefix,'relay_prefix':args.relay_prefix,'geometry_prefix':args.geometry_prefix,
        'transport_sorties':len(f),'boxes':len(b),'relay_sorties':len(m),'transport_energy_kwh':float(f['能耗kWh'].sum()),
        'relay_energy_kwh':float(m.energy_kwh.sum()),'all_return_s':float(max(f['返回秒'].max(),m.return_s.max())),
        'min_relay_soc_pct':float(m.return_soc_pct.min()),'direct_transport_seconds':float(t.loc[t.direct_ok,'duration_s'].sum()),
        'conservative_relay_transport_seconds':float(t.loc[~t.direct_ok,'duration_s'].sum()),'dem':'q1_flat/最终工作DEM.tif',
        'communication_scope':'直连采用全区间充分证书；未获得直连证书的时段预留中继，累计时长为保守中继保障时长，并非精确失联时长',
        'style':'沿用RdYlBu_r渐变、黑字、紧凑小面板；无热力矩阵；无总图名',**geography_qa}
    Path(str(out)+'_绘图核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__=='__main__':main()
