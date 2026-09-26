"""问题三绘图优化：九幅紧凑地图，使用V3最终方案及精确通信状态。"""
from pathlib import Path
import argparse,json,hashlib,os
os.environ.setdefault('Q2_CERTIFIED_PREFIX','问题三_V3_最终方案_运输')
from 问题三_V3_基础绘图 import load_v3,resolve,windows
from 问题三_V2_基础绘图 import (P,N,CM,INK,BLUE,RED,MapStyle,COORD,OFFSETS,
    plt,np,pd,Line2D,LineCollection,ScalarMappable,patheffects,clip_intervals)
from matplotlib.colors import LightSource
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

class CompactMap(MapStyle):
    def __init__(self,missions):
        super().__init__(missions)
        from 问题三_V2_基础绘图 import read_dem
        z,left,top,dx,dy=read_dem(P/'q1_flat/最终工作DEM.tif')
        x0,x1,y0,y1=self.ext
        r0=max(0,int((top-y1)/dy));r1=min(z.shape[0],int((top-y0)/dy)+2)
        c0=max(0,int((x0-left)/dx));c1=min(z.shape[1],int((x1-left)/dx)+2)
        self.zz=z[r0:r1,c0:c1]
        shade=LightSource(azdeg=315,altdeg=48).hillshade(self.zz,dx=30,dy=30,vert_exag=1.7)
        terrain=CM(self.norm(self.zz))[:,:,:3]*(.72+.28*shade[:,:,None])
        self.rgb=np.clip(.27+.73*terrain,0,1)
    def base(self,ax,i,title):
        ax.imshow(self.rgb,extent=self.zext,origin='upper',interpolation='nearest',zorder=0)
        ax.set_xlim(self.ext[:2]);ax.set_ylim(self.ext[2:]);ax.set_aspect(self.aspect)
        ax.set_xticks([109.16,109.20,109.24,109.28]);ax.set_yticks([23.00,23.04,23.08])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'));ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.tick_params(labelsize=15.5,width=1.6,length=4.5,pad=3,labelleft=i%3==0,labelbottom=i//3==2)
        for sp in ax.spines.values():sp.set_linewidth(1.65);sp.set_color('black')
        ax.set_title(f'{chr(97+i)}  {title}',loc='left',fontsize=21.5,pad=8,color='black')
        if i%3==0:ax.set_ylabel('纬度（°）',fontsize=18,labelpad=6)
        if i//3==2:ax.set_xlabel('经度（°）',fontsize=18,labelpad=5)

def halo(text,width=2.8):
    text.set_path_effects([patheffects.withStroke(linewidth=width,foreground='white')])

def nodes(ax,labels=True):
    for site,xy in COORD.items():
        if site=='O01':
            ax.scatter(*xy,s=280,marker='*',c='#111111',ec='white',lw=1.2,zorder=25)
            halo(ax.annotate('O01',xy,xytext=(0,-21),textcoords='offset points',ha='center',fontsize=16,zorder=28,fontweight='bold'))
        else:
            ax.scatter(*xy,s=66,c='white',ec='#202020',lw=1.1,zorder=20)
            if labels:
                ox,oy=OFFSETS[site]
                halo(ax.annotate(site,xy,xytext=(ox,oy),textcoords='offset points',ha='right' if ox<0 else 'left',fontsize=15.5,zorder=25))

def routes(ax,frame,color='#373c46',lw=1.65,alpha=.73):
    # 合并重复的平面边仅用于显示，保留真实节点间直线。
    seen=set()
    for rec in frame.to_dict('records'):
        seq=rec['访问顺序'].split('→')
        for a,b in zip(seq[:-1],seq[1:]):
            key=tuple(sorted([a,b]))
            if key in seen:continue
            seen.add(key);xy=np.array([COORD[a],COORD[b]])
            ax.plot(xy[:,0],xy[:,1],color='white',lw=lw+1,alpha=.6,zorder=2)
            ax.plot(xy[:,0],xy[:,1],color=color,lw=lw,alpha=alpha,zorder=3)

def segments(ax,df,color,lw=3.6):
    if df.empty:return
    a=df[['p0_lon','p0_lat']].to_numpy();b=df[['p1_lon','p1_lat']].to_numpy();moving=np.linalg.norm(a-b,axis=1)>1e-11
    if moving.any():
        seg=np.stack([a[moving],b[moving]],axis=1)
        ax.add_collection(LineCollection(seg,colors='white',linewidths=lw+1.65,zorder=5))
        ax.add_collection(LineCollection(seg,colors=[color],linewidths=lw,zorder=6))
    if (~moving).any():
        xy=np.unique(np.round(a[~moving],9),axis=0)
        ax.scatter(xy[:,0],xy[:,1],s=76,c=[color],edgecolors='white',lw=.8,zorder=8)

def relay(ax,m,backhaul=False,height=False):
    for _,r in m.iterrows():
        x,y=r.lon,r.lat
        if backhaul:
            o=COORD['O01'];ax.plot([o[0],x],[o[1],y],color='white',lw=3.8,zorder=9)
            ax.plot([o[0],x],[o[1],y],color='#222222',lw=2.3,ls=(0,(4,3)),zorder=10)
        ax.scatter(x,y,s=190,marker='D',c=[RED],ec='white',lw=1.5,zorder=30)
        right=x>ax.get_xlim()[0]+.73*np.diff(ax.get_xlim())[0]
        label=f'{r.relay_mission_id} · {r.agl:.0f}米' if height else r.relay_mission_id
        halo(ax.annotate(label,(x,y),xytext=(-10 if right else 10,8),textcoords='offset points',
            ha='right' if right else 'left',fontsize=16,fontweight='bold',zorder=31),3.1)

def draw(prefix,geometry,stations,out,dpi=340):
    f,b,m,c,t,s,source=load_v3(prefix,geometry,stations);sty=CompactMap(m);t=sty.coordinates(t)
    covered=t[~t.direct_ok].copy();covered['relay_drone']=covered.relay_mission_id.map(m.set_index('relay_mission_id').relay_drone)
    plt.rcParams.update({'font.size':18,'axes.labelsize':18,'text.color':'black','axes.labelcolor':'black'})
    fig=plt.figure(figsize=(22.8,18.7));gs=fig.add_gridspec(3,3,left=.058,right=.988,bottom=.09,top=.965,wspace=.035,hspace=.105)
    periods=windows(f,m);titles=['运输航线','直连与中继','中继位置与高度','回传链路','R01 通信保障','R02 通信保障']+[f'{lo//60}—{hi//60} 分钟' for lo,hi in periods]
    axs=[fig.add_subplot(gs[i//3,i%3]) for i in range(9)]
    for i,ax in enumerate(axs):sty.base(ax,i,titles[i])
    routes(axs[0],f,lw=2.15,alpha=.95);nodes(axs[0]);sty.north(axs[0])
    segments(axs[1],t[t.direct_ok],BLUE);segments(axs[1],covered,RED);nodes(axs[1])
    routes(axs[2],f,lw=1.3,alpha=.55);nodes(axs[2],labels=False);relay(axs[2],m,height=True)
    routes(axs[3],f,lw=1.4,alpha=.55);relay(axs[3],m,backhaul=True);nodes(axs[3])
    for k,drone in enumerate(['R01','R02']):
        routes(axs[4+k],f,lw=1.3,alpha=.4);segments(axs[4+k],covered[covered.relay_drone.eq(drone)],RED,4.0)
        nodes(axs[4+k]);relay(axs[4+k],m[m.relay_drone.eq(drone)])
    for k,(lo,hi) in enumerate(periods):
        seg=clip_intervals(t,lo,hi);segments(axs[6+k],seg[seg.direct_ok],BLUE);segments(axs[6+k],seg[~seg.direct_ok],RED)
        nodes(axs[6+k]);active=m[(m.service_start_s<hi)&(m.service_end_s>lo)];relay(axs[6+k],active)
    handles=[Line2D([],[],color='#373c46',lw=2.2,label='运输航线'),Line2D([],[],color=BLUE,lw=3.6,label='直连'),
        Line2D([],[],color=RED,lw=3.6,label='中继'),Line2D([],[],color='#222222',lw=2.2,ls='--',label='回传'),
        Line2D([],[],ls='',marker='D',mfc=RED,mec='white',ms=10,label='中继悬停'),Line2D([],[],ls='',marker='*',mfc='#111111',mec='white',ms=14,label='调度中心')]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.065,.006),ncol=6,frameon=False,fontsize=17,handlelength=1.4,columnspacing=1.4)
    cb=fig.colorbar(ScalarMappable(norm=sty.norm,cmap=CM),cax=fig.add_axes([.705,.026,.235,.011]),orientation='horizontal')
    cb.set_ticks([200,600,1000]);cb.ax.tick_params(labelsize=15,width=1.2,length=4);cb.outline.set_linewidth(1.1)
    fig.text(.8225,.044,'高程（米）',ha='center',fontsize=16)
    path=resolve(str(out)+'_图01_九面板通信空间图.png');tmp=path.with_name(path.stem+'.tmp.png')
    fig.savefig(tmp,dpi=dpi,bbox_inches='tight',pad_inches=.05);plt.close(fig)
    with Image.open(tmp)as im:im.verify()
    tmp.replace(path)
    qa={'PASS':True,'source':source,'transport_sorties':len(f),'relay_sorties':len(m),'boxes':len(b),
        'direct_seconds':float(t.loc[t.direct_ok,'duration_s'].sum()),'relay_seconds':float(t.loc[~t.direct_ok,'duration_s'].sum()),
        'global_elevation_range_m':[sty.norm.vmin,sty.norm.vmax],'height_agl_m':m.agl.tolist(),
        'display':'真实DEM色阶加光照；未平滑或修改高程/坐标/通信状态；共用色标及地理范围','png_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    resolve(str(out)+'_地图绘图核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(path)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prefix',default='问题三_V3_最终方案');p.add_argument('--geometry',default=None)
    p.add_argument('--stations',default='问题三_V3_站点.csv');p.add_argument('--output-prefix',default='问题三_V4');p.add_argument('--dpi',type=int,default=340)
    a=p.parse_args();draw(a.prefix,a.geometry,a.stations,a.output_prefix,a.dpi)
