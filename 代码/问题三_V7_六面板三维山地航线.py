"""V7：六面板真实三维山地航线。
python 问题三_V7_六面板三维山地航线.py --dpi 340
冻结最终结果，仅调整绘图；三维轴使用真实海拔，不把时间作为高度。
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from matplotlib.colors import LightSource,Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from 问题三_V3_基础绘图 import load_v3,resolve,parser,windows,clip_intervals,save,plt,P
from 认证更新_地图剖面公用 import N,CM,read_dem,CN
from q1_flat.精确算法 import utm49
from 问题三_V6_山地航线 import projected_grid
import hashlib

Z_EXAGGERATION=3.5
INK='#111111'
MC={'A':CM(.025),'B':CM(.79),'C':CM(.98)}
plt.rcParams.update({'font.family':['STIXGeneral',CN], 'font.size':19,
    'axes.labelsize':17,'xtick.labelsize':15,'ytick.labelsize':15,
    'axes.titlesize':20,'legend.fontsize':17,'text.color':INK,
    'axes.labelcolor':INK,'xtick.color':INK,'ytick.color':INK,
    'axes.linewidth':1.2,'figure.facecolor':'white','savefig.facecolor':'white',
    'axes.unicode_minus':False,'mathtext.fontset':'stix'})

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

    def base(self,ax,letter,title,axis_labels=True):
        ax.computed_zorder=False
        ax.patch.set_alpha(0)
        ax.plot_surface(self.x,self.y,self.z,facecolors=self.facecolors,
            rstride=1,cstride=1,linewidth=0,edgecolor='none',shade=False,
            antialiased=False,alpha=1,zorder=1)
        ax.set_xlim(self.xlim);ax.set_ylim(self.ylim);ax.set_zlim(self.zlim)
        ax.set_box_aspect((np.ptp(self.xlim),np.ptp(self.ylim),np.ptp(self.zlim)/1000*Z_EXAGGERATION),zoom=1.02)
        ax.view_init(elev=44,azim=-62);ax.set_proj_type('ortho')
        ax.xaxis.set_major_locator(MultipleLocator(5));ax.yaxis.set_major_locator(MultipleLocator(5));ax.zaxis.set_major_locator(MultipleLocator(400))
        ax.set_xlabel('东向距离（千米）' if axis_labels else '',labelpad=9);ax.set_ylabel('北向距离（千米）' if axis_labels else '',labelpad=9)
        ax.set_zlabel('');ax.text2D(.985,.72,'海拔（米）',transform=ax.transAxes,ha='right',fontsize=18)
        ax.tick_params(axis='both',pad=1,length=3,labelsize=15)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:
            axis.pane.fill=False;axis.pane.set_edgecolor('#c0c3c6')
            axis.line.set_color(INK);axis.line.set_linewidth(1.2)
            axis._axinfo['grid'].update({'color':(.42,.45,.50,.13),'linewidth':.55})
        ax.text2D(.018,.98,letter,transform=ax.transAxes,fontsize=23,fontweight='bold',va='top')
        ax.text2D(.076,.978,title,transform=ax.transAxes,fontsize=20,va='top')
        base=np.array([0.,0.,self.z0])
        ax.scatter(*base,s=155,marker='*',c=INK,ec='white',lw=1.0,zorder=15,depthshade=False)
        ax.text(base[0],base[1]-.65,base[2],'O01',fontsize=15,ha='center',zorder=20)

    def xyz(self,frame,prefix):
        p=frame[[prefix+'_east',prefix+'_north',prefix+'_z']].to_numpy(float)
        p[:,:2]=(p[:,:2]-self.origin)/1000;return p

    def flight(self,ax,frame,width=3.0,direction=False):
        if not len(frame):return
        a=self.xyz(frame,'p0');b=self.xyz(frame,'p1')
        moving=np.linalg.norm((b-a)*np.array([1,1,.001]),axis=1)>1e-9
        if not moving.any():return
        seg=np.stack([a[moving],b[moving]],axis=1)
        ax.add_collection3d(Line3DCollection(seg,colors='white',linewidths=width+1.05,alpha=1,zorder=7))
        colors=[MC[g] for g in frame.loc[moving,'model']]
        ax.add_collection3d(Line3DCollection(seg,colors=colors,linewidths=width,alpha=1,zorder=8))
        if direction:
            choices=[]
            for _,part in frame[frame.stage.eq('巡航')].groupby(['route','from_node','to_node'],sort=False):
                part=part.sort_values('start_s');p=self.xyz(part.iloc[[0]],'p0')[0];q=self.xyz(part.iloc[[-1]],'p1')[0]
                length=np.linalg.norm((q-p)[:2])
                if length>1:choices.append((length,p,q,MC[part.model.iloc[0]]))
            selected=[]
            for _,p,q,c in sorted(choices,key=lambda x:x[0],reverse=True):
                mid=(p+q)/2
                if any(np.linalg.norm((mid-k)[:2])<1.5 for k in selected):continue
                v=(q-p)/np.linalg.norm((q-p)[:2])*.46
                ax.quiver(*(mid-v/2),*v,color=c,arrow_length_ratio=.65,linewidth=1.8,length=1,normalize=False,zorder=10)
                selected.append(mid)
                if len(selected)==2:break

    def nodes(self,ax,frame,labels=True):
        sites=sorted((set(frame.from_node)|set(frame.to_node))-{'O01'})
        if not sites:return
        p=np.array([self.node[s] for s in sites])
        ax.scatter(p[:,0],p[:,1],p[:,2],s=34,c='white',ec=INK,lw=.95,depthshade=False,zorder=11)
        if labels:
            for s in sorted(sites,key=lambda s:self.node[s][2],reverse=True)[:2]:
                q=self.node[s];ax.text(q[0]+.10,q[1],q[2]+75,s,fontsize=15,color=INK,zorder=19)

    def hover_xyz(self,r):
        xy=(np.array(utm49(r.lon,r.lat))-self.origin)/1000;return np.r_[xy,r.z]

    def relay(self,ax,missions,stations,flight=True):
        for (_,_,_),group in missions.groupby(['lon','lat','z'],sort=False):
            r=group.iloc[0];p=self.hover_xyz(r)
            if flight:
                st=stations.loc[r.station_id].to_dict();st['station_id']=r.station_id
                f={'cruise_z':max(float(st['line_max_z'])+50,float(st['z']),self.z0)}
                track=np.array([[0,0,self.z0],[0,0,f['cruise_z']],[p[0],p[1],f['cruise_z']],p])
                ax.plot(*track.T,c='white',lw=4.0,zorder=8)
                ax.plot(*track.T,c=INK,lw=2.8,zorder=9)
            ends=np.vstack([self.gateway,p])
            ax.plot(*ends.T,c='white',lw=3.0,ls=(0,(4,3)),zorder=8)
            ax.plot(*ends.T,c=INK,lw=1.7,ls=(0,(4,3)),zorder=9)
            ax.scatter(*p,s=112,marker='D',c=[CM(.99)],ec='white',lw=1.4,depthshade=False,zorder=15)
            label='/'.join(group.relay_mission_id)+f'  {r.agl:.0f} 米'
            ax.text(p[0]+.18,p[1],p[2]+72,label,fontsize=15,fontweight='bold',zorder=20)


def shared(fig,scene,relay_flight=True):
    handles=[Line2D([],[],c=MC[g],lw=3.3,label=f'{g} 型运输') for g in 'ABC']
    if relay_flight:handles.append(Line2D([],[],c=INK,lw=2.8,label='中继航迹'))
    handles.extend([Line2D([],[],c=INK,lw=1.7,ls='--',label='回传链路'),Line2D([],[],marker='D',ls='',mfc=CM(.99),mec='white',ms=10,label='中继悬停')])
    fig.legend(handles=handles,ncol=len(handles),loc='upper center',bbox_to_anchor=(.5,1.002),frameon=False,fontsize=18,columnspacing=1.7,handlelength=2.1)
    cax=fig.add_axes([.35,.041,.29,.013])
    cb=fig.colorbar(ScalarMappable(norm=scene.norm,cmap=CM),cax=cax,orientation='horizontal')
    cb.set_ticks([100,400,700,1000]);cb.ax.tick_params(labelsize=15,length=3,pad=3);cb.outline.set_linewidth(.8)
    fig.text(.653,.045,'地形海拔（米）',fontsize=18,va='center')


def save_v7(fig,path,dpi):
    from PIL import Image
    p=Path(path);tmp=p.with_suffix('.tmp.png')
    fig.savefig(tmp,dpi=dpi,bbox_inches='tight',pad_inches=.09)
    plt.close(fig)
    with Image.open(tmp) as im:im.verify()
    with Image.open(tmp) as im:im.load()
    tmp.replace(p);print(p.name,flush=True)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap=parser();ap.set_defaults(output_prefix='问题三_V7');ap.add_argument('--dpi',type=int,default=340);a=ap.parse_args()
    preserved=[p for p in P.glob('*.png') if p.name.startswith(('问题三_V4_','问题三_V5_','问题三_V6_')) and '.tmp.' not in p.name]
    hashes={p.name:sha(p) for p in preserved}
    f,b,m,c,t,s,provenance=load_v3(a.prefix,a.geometry,a.stations)
    assert len(t)>0 and len(f)==22 and len(m)==4 and len(b)==80
    out=resolve(a.output_prefix);stations=s.set_index('station_id');scene=Scene(m)
    used=stations.loc[m.station_id.unique()]
    maxz=max(scene.zlim[1],float(t.p0_z.max()),float(t.p1_z.max()),float(m.z.max()),float(used.cruise_z.max()))
    scene.zlim=(0.,float(np.ceil(maxz/200)*200))
    covered=t[~t.direct_ok].copy();covered['relay_drone']=covered.relay_mission_id.map(m.set_index('relay_mission_id').relay_drone)
    fig=plt.figure(figsize=(22.7,13.8))
    gs=fig.add_gridspec(2,3,left=.005,right=.990,bottom=.057,top=.955,wspace=.025,hspace=-.025)
    frames=[];panels=[]
    for i,title in enumerate(['全部航线','A 型运输','B 型运输','C 型运输','R01 中继','R02 中继']):
        ax=fig.add_subplot(gs[i//3,i%3],projection='3d');scene.base(ax,chr(97+i),title,axis_labels=i>=3)
        missions=m.iloc[0:0]
        if i==0:
            frame=t.copy();scene.flight(ax,frame,width=2.9,direction=True);scene.nodes(ax,frame)
        elif i<4:
            frame=t[t.model.eq('ABC'[i-1])].copy()
            scene.flight(ax,frame,width=3.25,direction=True);scene.nodes(ax,frame)
        else:
            drone=['R01','R02'][i-4];frame=covered[covered.relay_drone.eq(drone)].copy()
            scene.flight(ax,frame,width=3.0);scene.nodes(ax,frame,labels=False)
            missions=m[m.relay_drone.eq(drone)]
            scene.relay(ax,missions,stations)
        pa=scene.xyz(frame,'p0');pb=scene.xyz(frame,'p1')
        assert np.array_equal(pa[:,2],frame.p0_z.to_numpy(float))
        assert np.array_equal(pb[:,2],frame.p1_z.to_numpy(float))
        frame['panel']=chr(97+i);frame['panel_name']=title
        frame['plot_x0_km']=pa[:,0];frame['plot_y0_km']=pa[:,1];frame['plot_z0_m']=pa[:,2]
        frame['plot_x1_km']=pb[:,0];frame['plot_y1_km']=pb[:,1];frame['plot_z1_m']=pb[:,2]
        frames.append(frame)
        panels.append({'panel':chr(97+i),'title':title,'interval_count':len(frame),
            'transport_sorties':int(frame.route.nunique()),'transport_phase_seconds':float(frame.duration_s.sum()),
            'relay_missions':list(missions.relay_mission_id)})
    shared(fig,scene)
    png=Path(str(out)+'_图03_六面板三维山地航线.png')
    save_v7(fig,png,a.dpi)
    from PIL import Image
    with Image.open(png) as im:im.load();size=list(im.size)
    import pandas as pd
    pd.concat(frames,ignore_index=True).to_csv(str(out)+'_图03_绘图区间.csv',index=False,encoding='utf-8-sig')
    assert hashes=={p.name:sha(p) for p in preserved}
    assert abs(sum(p['transport_phase_seconds'] for p in panels[1:4])-panels[0]['transport_phase_seconds'])<1e-6
    assert abs(sum(p['transport_phase_seconds'] for p in panels[4:])-float(covered.duration_s.sum()))<1e-6
    qa={**provenance,'transport_sorties':len(f),'relay_sorties':len(m),'boxes':len(b),
        'source_interval_count':len(t),'source_relay_intervals':len(covered),'panels':panels,
        'coordinate_system':'O01-relative WGS84/UTM49N easting and northing, km',
        'vertical_axis':'unchanged actual metres from p0_z/p1_z',
        'visual_vertical_exaggeration':Z_EXAGGERATION,
        'terrain_display':'stride-2 original corrected DEM; no smoothing, no height offsets',
        'dem_sha256':sha(P/'q1_flat'/'最终工作DEM.tif'),
        'shared_elevation_color_range_m':[scene.norm.vmin,scene.norm.vmax],
        'geometry_range':{'east_km':scene.xlim,'north_km':scene.ylim,'altitude_m':scene.zlim},
        'relay_hover_altitude_m':{r.relay_mission_id:float(r.z) for r in m.itertuples()},
        'relay_transport_scope':'only analytic direct-priority relay demand intervals assigned to the stated relay drone',
        'relay_transit_altitude':'max(line_max_z+50, hover altitude, O01 ground altitude); same audited mission geometry',
        'transport_model_partition_seconds_conserved':True,'relay_drone_partition_seconds_conserved':True,
        'png':{'file':png.name,'size_px':size,'dpi':a.dpi,'decode':'PASS'},
        'other_figures_preserved_byte_identical':True,'new_optimization_performed':False}
    Path(str(out)+'_图03_核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    Path(str(out)+'_图03_说明.md').write_text(f"""六面板依次展示全部航线、A型运输、B型运输、C型运输、R01中继、R02中继。前三种机型按最终方案分类；最后两幅仅绘制精确通信审计判定需由该实体中继无人机保障的运输航段，并叠加其实际中继往返路径、悬停位置和回传链路。

全部面板使用最终工作DEM的原始海拔及真实飞行海拔。DEM仅每两像元取样用于显示，不进行额外插值平滑；航线保留爬升、水平巡航和下降。中继去返巡航海拔取沿线最高点加50米、实际悬停海拔与O01地面高程中的最大值，符合既定中继模型。重复往返的几何位置重合，不人为平移以分开线条。

三维坐标采用O01相对UTM49N距离（千米）及实际海拔（米）；纵向显示统一夸张{Z_EXAGGERATION:g}倍以便辨认山脉，未改变数据高程。所有面板范围、视角、色带和海拔色标相同。悬停点文字中的高度是实际离地高度，点的竖轴位置是实际海拔。

22个运输架次、4个中继架次、80个货箱及{len(t)}个解析区间沿用时间主方案。各机型分面时长之和等于全部航线时长；两架中继实体分面时长之和等于真实中继需求时长。
""",encoding='utf-8')
    print(json.dumps({'png':qa['png'],'panels':panels},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
