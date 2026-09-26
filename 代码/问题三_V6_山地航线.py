"""三维真实山地上的实际飞行航线：仅重绘经审计的冻结方案。

python 问题三_V6_山地航线.py --dpi 340
横纵坐标为 O01 相对投影距离，竖轴为实际海拔。DEM 与航线的高程值
均直接采用原数据；三个时间窗只用于筛选航线，不充当第三坐标。
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from PIL import Image
from matplotlib.colors import Normalize,LightSource
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator,FixedLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from 问题三_V3_基础绘图 import load_v3,resolve,clip_intervals,plt,P
from 认证更新_地图剖面公用 import N,CM,read_dem,CN
from q1_flat.精确算法 import utm49

INK='#111111'
COLORS={'J01':CM(.025),'J02':CM(.79),'J03':CM(.98)}
WINDOWS=[(0.,2700.),(2700.,5400.),(5400.,8100.)]
plt.rcParams.update({'font.family':['STIXGeneral',CN],'font.size':19,
    'axes.labelsize':18,'xtick.labelsize':15,'ytick.labelsize':15,
    'text.color':INK,'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'figure.facecolor':'white','savefig.facecolor':'white','mathtext.fontset':'stix',
    'axes.unicode_minus':False})


def projected_grid(lon,lat):
    """与计算内核 utm49 同公式的向量化实现。"""
    a,e2,k0=6378137.,0.0066943799901413165,.9996
    phi=np.deg2rad(lat);dl=np.deg2rad(lon-111.)
    ep2=e2/(1-e2);s=np.sin(phi);c=np.cos(phi);t=np.tan(phi)**2
    C=ep2*c*c;A=c*dl;n=a/np.sqrt(1-e2*s*s)
    m=a*((1-e2/4-3*e2**2/64-5*e2**3/256)*phi
        -(3*e2/8+3*e2**2/32+45*e2**3/1024)*np.sin(2*phi)
        +(15*e2**2/256+45*e2**3/1024)*np.sin(4*phi)
        -(35*e2**3/3072)*np.sin(6*phi))
    x=500000+k0*n*(A+(1-t+C)*A**3/6+(5-18*t+t*t+72*C-58*ep2)*A**5/120)
    y=k0*(m+n*np.tan(phi)*(A*A/2+(5-t+9*C+4*C*C)*A**4/24
        +(61-58*t+t*t+600*C-330*ep2)*A**6/720))
    return x,y


class Scene:
    def __init__(self,missions):
        self.origin=np.array(utm49(float(N.loc['O01','longitude_deg']),float(N.loc['O01','latitude_deg'])))
        self.node={s:np.r_[(np.array(utm49(r.longitude_deg,r.latitude_deg))-self.origin)/1000,r.operating_altitude_m] for s,r in N.iterrows()}
        self.z0=float(N.loc['O01','ground_elevation_m']);self.gateway=np.array([0.,0.,self.z0+20.])
        dem,left,top,dx,dy=read_dem(P/'q1_flat'/'最终工作DEM.tif')
        lon=np.r_[N.longitude_deg,missions.lon];lat=np.r_[N.latitude_deg,missions.lat]
        ext=[lon.min()-.009,lon.max()+.009,lat.min()-.008,lat.max()+.008]
        r0=max(0,int((top-ext[3])/dy));r1=min(dem.shape[0],int((top-ext[2])/dy)+2)
        c0=max(0,int((ext[0]-left)/dx));c1=min(dem.shape[1],int((ext[1]-left)/dx)+2)
        rr=np.arange(r0,r1,2);cc=np.arange(c0,c1,2)
        ll,aa=np.meshgrid(left+(cc+.5)*dx,top-(rr+.5)*dy)
        xx,yy=projected_grid(ll,aa)
        self.x=(xx-self.origin[0])/1000;self.y=(yy-self.origin[1])/1000
        self.z=dem[np.ix_(rr,cc)]
        self.xlim=(float(self.x.min()),float(self.x.max()));self.ylim=(float(self.y.min()),float(self.y.max()))
        self.zlim=(0.,float(np.ceil(max(1200.,self.z.max())/200)*200))
        self.norm=Normalize(float(dem.min()),float(dem.max()))
        # 真正的 RdYlBu_r 高程色，不掺白；光照只调节亮度，z 不改动。
        shade=LightSource(azdeg=305,altdeg=47).hillshade(self.z,dx=60,dy=60,vert_exag=1.)
        rgb=CM(self.norm(self.z))[:,:,:3]
        rgb=np.clip(rgb*(.58+.54*shade[:,:,None]),0,1)
        self.facecolors=np.dstack([rgb,np.ones(self.z.shape)])

    def hover_xyz(self,r):
        xy=(np.array(utm49(r.lon,r.lat))-self.origin)/1000;return np.r_[xy,r.z]



def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

Z_EXAGGERATION=3.5
DIRECT='#333b47'

def xyz(scene,frame,prefix):
    out=frame[[prefix+'_east',prefix+'_north',prefix+'_z']].to_numpy(float)
    out[:,:2]=(out[:,:2]-scene.origin)/1000
    return out


def base(ax,scene,letter,title):
    ax.computed_zorder=False;ax.patch.set_alpha(0)
    ax.plot_surface(scene.x,scene.y,scene.z,facecolors=scene.facecolors,
        rstride=1,cstride=1,linewidth=0,edgecolor='none',shade=False,
        antialiased=False,alpha=1,zorder=1)
    ax.set_xlim(scene.xlim);ax.set_ylim(scene.ylim);ax.set_zlim(scene.zlim)
    ax.set_box_aspect((np.ptp(scene.xlim),np.ptp(scene.ylim),
        np.ptp(scene.zlim)/1000*Z_EXAGGERATION),zoom=1.04)
    ax.set_proj_type('ortho');ax.view_init(elev=44,azim=-62)
    ax.xaxis.set_major_locator(MultipleLocator(5));ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.zaxis.set_major_locator(MultipleLocator(400))
    ax.set_xlabel('东向距离（千米）',labelpad=10)
    ax.set_ylabel('北向距离（千米）',labelpad=10)
    ax.set_zlabel('')
    ax.text2D(.99,.745,'海拔（米）',transform=ax.transAxes,ha='right',fontsize=18)
    ax.tick_params(axis='both',labelsize=16,pad=1,length=3)
    for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
        axis.pane.fill=False;axis.pane.set_edgecolor('#bfc4ca')
        axis.line.set_color(INK);axis.line.set_linewidth(1.3)
        axis._axinfo['grid'].update({'color':(.40,.44,.48,.13),'linewidth':.6})
    ax.text2D(.015,.99,letter,transform=ax.transAxes,fontsize=25,fontweight='bold',va='top')
    ax.text2D(.079,.987,title,transform=ax.transAxes,fontsize=21,va='top')
    ax.scatter(0.,0.,scene.z0,marker='*',s=145,c=INK,ec='white',lw=1.0,depthshade=False,zorder=15)
    ax.text(0.,-.65,scene.z0,'O01',fontsize=15,ha='center',zorder=20)


def draw_routes(ax,scene,frame):
    a=xyz(scene,frame,'p0');b=xyz(scene,frame,'p1')
    moving=np.linalg.norm((b-a)*np.array([1.,1.,.001]),axis=1)>1e-9
    details=[]
    for mode,partmask in [('直连',frame.direct_ok.to_numpy()),('中继',~frame.direct_ok.to_numpy())]:
        mask=moving&partmask
        if not mask.any():continue
        seg=np.stack([a[mask],b[mask]],axis=1)
        color=[DIRECT if mode=='直连' else COLORS[k] for k in frame.loc[mask,'relay_mission_id']]
        width=2.25 if mode=='直连' else 3.15
        order=6 if mode=='直连' else 8
        ax.add_collection3d(Line3DCollection(seg,colors='white',linewidths=width+1.0,alpha=1,zorder=order))
        ax.add_collection3d(Line3DCollection(seg,colors=color,linewidths=width,alpha=1,zorder=order+1))
        details.append({'mode':mode,'moving_segments':int(mask.sum())})
    # 服务区位置仍为作业海拔，不把节点压到绘图平面。
    sites=sorted((set(frame.from_node)|set(frame.to_node))-{'O01'})
    if sites:
        p=np.array([scene.node[s] for s in sites])
        ax.scatter(p[:,0],p[:,1],p[:,2],s=31,c='white',ec=INK,lw=.95,depthshade=False,zorder=12)
    return details


def draw_relays(ax,scene,missions,lo,hi):
    represented=[]
    for r in missions.itertuples():
        if min(hi,r.service_end_s)<=max(lo,r.service_start_s):continue
        p=scene.hover_xyz(r);col=COLORS[r.relay_mission_id]
        # 虚线为真实悬停高度的垂线；下端用该站点工作DEM地面高程。
        ax.plot([p[0],p[0]],[p[1],p[1]],[r.ground_z,p[2]],c='white',lw=3.1,ls='--',zorder=9)
        ax.plot([p[0],p[0]],[p[1],p[1]],[r.ground_z,p[2]],c=col,lw=1.7,ls='--',zorder=10)
        ax.scatter(*p,marker='D',s=100,c=[col],ec='white',lw=1.35,depthshade=False,zorder=15)
        ax.text(p[0]+.18,p[1],p[2]+62,r.relay_mission_id,fontsize=16,fontweight='bold',zorder=21)
        represented.append({'mission':r.relay_mission_id,'hover_altitude_m':float(r.z),
            'ground_altitude_m':float(r.ground_z),'agl_m':float(r.agl)})
    return represented


def export(fig,path,dpi):
    path=Path(path);tmp=path.with_suffix('.tmp.png')
    fig.savefig(tmp,dpi=dpi,bbox_inches='tight',pad_inches=.10);plt.close(fig)
    with Image.open(tmp) as im:im.verify()
    with Image.open(tmp) as im:im.load();size=list(im.size)
    tmp.replace(path);return {'file':path.name,'size_px':size,'dpi':dpi,'decode':'PASS'}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--prefix',default='问题三_V3_最终方案')
    ap.add_argument('--geometry',default=None)
    ap.add_argument('--stations',default='问题三_V3_站点.csv')
    ap.add_argument('--output-prefix',default='问题三_V6')
    ap.add_argument('--dpi',type=int,default=340)
    args=ap.parse_args();out=resolve(args.output_prefix)
    preserved=list(P.glob('问题三_V4*'))+list(P.glob('问题三_V5_图08*'))+list(P.glob('问题三_V5_图10*'))
    frozen={p.name:sha(p) for p in preserved if p.is_file() and '.tmp.' not in p.name}
    f,b,m,c,t,s,provenance=load_v3(args.prefix,args.geometry,args.stations)
    assert len(t)==443 and len(t[~t.direct_ok])==138
    scene=Scene(m)
    maxz=max(scene.zlim[1],t.p0_z.max(),t.p1_z.max(),m.z.max())
    scene.zlim=(0.,float(np.ceil(maxz/200)*200))
    fig=plt.figure(figsize=(22.5,8.1))
    gs=fig.add_gridspec(1,3,left=.006,right=.984,bottom=.118,top=.925,wspace=.025)
    parts=[];summary=[]
    for i,(lo,hi) in enumerate(WINDOWS):
        frame=clip_intervals(t,lo,hi)
        frame['time_window']=f'{int(lo/60)}—{int(hi/60)}分钟'
        frame['plot_z0_m']=frame.p0_z;frame['plot_z1_m']=frame.p1_z
        # 明确输出作图坐标，便于与原始航段逐一核对。
        frame['plot_x0_km']=(frame.p0_east-scene.origin[0])/1000
        frame['plot_y0_km']=(frame.p0_north-scene.origin[1])/1000
        frame['plot_x1_km']=(frame.p1_east-scene.origin[0])/1000
        frame['plot_y1_km']=(frame.p1_north-scene.origin[1])/1000
        parts.append(frame)
        ax=fig.add_subplot(gs[0,i],projection='3d')
        base(ax,scene,chr(97+i),f'{int(lo/60)}—{int(hi/60)} 分钟')
        routeparts=draw_routes(ax,scene,frame)
        missions=draw_relays(ax,scene,m,lo,hi)
        summary.append({'window_s':[lo,hi],'segments':len(frame),'transport_sorties':int(frame.route.nunique()),
            'seconds':float((frame.end_s-frame.start_s).sum()),'drawn_parts':routeparts,'relay_points':missions})
    handles=[Line2D([],[],c=DIRECT,lw=2.8,label='直连保障航段')]
    handles += [Line2D([],[],c=COLORS[k],lw=3.2,label=k+' 保障航段') for k in COLORS]
    handles.append(Line2D([],[],c=INK,marker='D',mec='white',lw=0,ms=8,label='中继悬停'))
    fig.legend(handles=handles,ncol=5,loc='upper center',bbox_to_anchor=(.5,1.002),
        frameon=False,fontsize=18,columnspacing=1.8,handlelength=2.0)
    cax=fig.add_axes([.346,.046,.285,.013])
    cb=fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),cax=cax,orientation='horizontal')
    cb.set_ticks([100,400,700,1000]);cb.ax.tick_params(labelsize=16,length=3,pad=3);cb.outline.set_linewidth(.8)
    fig.text(.646,.052,'地形海拔（米）',fontsize=18,va='center')
    png=export(fig,str(out)+'_图09_三维山地真实航线.png',args.dpi)
    combined=pd.concat(parts,ignore_index=True)
    total=float(t.duration_s.sum());drawn=float((combined.end_s-combined.start_s).sum())
    assert abs(total-drawn)<1e-7
    assert np.array_equal(combined.plot_z0_m.to_numpy(),combined.p0_z.to_numpy())
    assert np.array_equal(combined.plot_z1_m.to_numpy(),combined.p1_z.to_numpy())
    assert frozen=={p.name:sha(p) for p in preserved if p.is_file() and '.tmp.' not in p.name}
    combined.to_csv(str(out)+'_图09_绘图区间.csv',index=False,encoding='utf-8-sig')
    qa={**provenance,'source_interval_count':len(t),'source_relay_intervals':138,
        'drawn_split_interval_count':len(combined),'all_flight_phase_seconds':total,'drawn_phase_seconds':drawn,
        'transport_sorties':len(f),'relay_sorties':len(m),'boxes':len(b),
        'coordinate_system':'O01-relative WGS84/UTM49N easting and northing, km',
        'vertical_axis':'actual sea-level altitude in metres: unchanged p0_z / p1_z',
        'vertical_exaggeration':Z_EXAGGERATION,'time_is_only_a_panel_filter':True,
        'terrain_display':'actual corrected DEM heights, stride 2 for display only, no smoothing',
        'dem_sha256':sha(P/'q1_flat'/'最终工作DEM.tif'),
        'dem_color_range_m':[scene.norm.vmin,scene.norm.vmax],
        'windows':summary,'png':png,'other_approved_figures_byte_identical':True,
        'preserved_sha256':frozen,'new_optimization_performed':False}
    Path(str(out)+'_图09_核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    Path(str(out)+'_图09_说明.md').write_text(f"""图09改为真实三维山地与实际飞行航迹叠加。竖轴为海拔（米），三个任务时段仅用于分面筛选。地形和航线共用同一空间坐标，DEM曲面采用最终工作DEM的原始高程，飞行线采用443个精确通信解析区间的 p0_z/p1_z。时间窗切分后为{len(combined)}段，累计{total:.9f}架次秒；绘制坐标中没有高程偏移、贴地替换或时间高度。

航线保留真实的爬升、水平巡航和下降段，直连保障为深灰，J01/J02/J03保障航段分别为蓝、橙、红；悬停点按真实海拔定位，向下虚线对应真实离地高度。各时段内的点表示其在该时段曾经提供服务，不表示始终同时服务。

所有面板同视角、同范围、同海拔色标。纵向比例统一夸张{Z_EXAGGERATION:g}倍，仅为增强地形可读性；数据高程未改变。DEM每两像元抽样只用于显示，未进行额外平滑。白色细描边仅增强航线可读性。运输停留阶段位置与相邻飞行段重合，不另画装饰性空中曲线。

本图不重新优化。原有V4图件及V5图08、图10保持不变。
""",encoding='utf-8')
    print(json.dumps({'output':png,'seconds':total,'panels':summary},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
