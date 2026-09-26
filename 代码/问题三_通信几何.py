"""问题三连续通信几何。基于分片常数DEM，精确检查射线扫掠三角形。
FSPL原题常数32.45；阻挡不直接断链，而增加10 dB。
轨迹按真实爬升/巡航/下降速度、每航段向上取整后静止等待构造。
"""
from pathlib import Path
import sys, json, math, ctypes, subprocess, shutil, os, warnings
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'q1_flat'))
from 精确算法 import read_dem,utm49,touched_cells
from 问题二_物理模型与输入 import read_inputs
FSPL_CONSTANT=32.45
LIMIT={'direct':122.,'access':116.,'backhaul':126.}
CPP=r'''
#include <cmath>
#include <algorithm>
#include <vector>
struct V {double x,y,z;};
static bool lineclear(const float*dem,int nr,int nc,V a,V b){
 double dx=b.x-a.x,dy=b.y-a.y,dz=b.z-a.z;
 if(fabs(dx)+fabs(dy)<1e-11){int x=floor(a.x),y=floor(a.y);return x>=0&&x<nc&&y>=0&&y<nr&&dem[y*nc+x]<std::min(a.z,b.z)-1e-7;}
 std::vector<double> ts{0.,1.};
 if(fabs(dx)>1e-12)for(int k=ceil(std::min(a.x,b.x));k<=floor(std::max(a.x,b.x));k++){double t=(k-a.x)/dx;if(t>0&&t<1)ts.push_back(t);}
 if(fabs(dy)>1e-12)for(int k=ceil(std::min(a.y,b.y));k<=floor(std::max(a.y,b.y));k++){double t=(k-a.y)/dy;if(t>0&&t<1)ts.push_back(t);}
 std::sort(ts.begin(),ts.end());
 for(int j=0;j+1<(int)ts.size();j++) {double t=(ts[j]+ts[j+1])/2;int x=floor(a.x+t*dx),y=floor(a.y+t*dy);if(x<0||x>=nc||y<0||y>=nr)return false;double low=std::min(a.z+ts[j]*dz,a.z+ts[j+1]*dz);if(dem[y*nc+x]>=low-1e-7)return false;}
 // Cell-boundary contacts: include every adjacent closed pixel.
 for(double t:ts){double xx=a.x+t*dx,yy=a.y+t*dy,zz=a.z+t*dz;int x=floor(xx),y=floor(yy);for(int h=-1;h<=1;h++)for(int k=-1;k<=1;k++){int c=x+k,r=y+h;if(c<0||c>=nc||r<0||r>=nr)continue;if(xx>=c-1e-9&&xx<=c+1+1e-9&&yy>=r-1e-9&&yy<=r+1+1e-9&&dem[r*nc+c]>=zz-1e-7)return false;}}
 return true;
}
static std::vector<V> clip(std::vector<V> p,int axis,double b,bool lower){
 std::vector<V>o;if(p.empty())return o;V prev=p.back();auto val=[&](V v){return axis?v.y:v.x;};
 bool pin=lower?val(prev)>=b-1e-10:val(prev)<=b+1e-10;
 for(V cur:p){bool in=lower?val(cur)>=b-1e-10:val(cur)<=b+1e-10;if(in!=pin){double den=val(cur)-val(prev);if(fabs(den)>1e-16){double t=(b-val(prev))/den;o.push_back({prev.x+t*(cur.x-prev.x),prev.y+t*(cur.y-prev.y),prev.z+t*(cur.z-prev.z)});}}if(in)o.push_back(cur);prev=cur;pin=in;}return o;
}
static bool triclear(const float*dem,int nr,int nc,V a,V b,V c){
 double det=(b.x-a.x)*(c.y-a.y)-(b.y-a.y)*(c.x-a.x);
 if(fabs(det)<1e-9)return lineclear(dem,nr,nc,a,b)&&lineclear(dem,nr,nc,a,c)&&lineclear(dem,nr,nc,b,c);
 std::vector<V>tri{a,b,c};double zmin=std::min({a.z,b.z,c.z});
 int ya=floor(std::min({a.y,b.y,c.y})-1e-9),yb=floor(std::max({a.y,b.y,c.y})+1e-9);
 for(int y=ya;y<=yb;y++){
  auto strip=clip(clip(tri,1,y,true),1,y+1,false);if(strip.empty())continue;
  double xmin=1e30,xmax=-1e30;for(V v:strip){xmin=std::min(xmin,v.x);xmax=std::max(xmax,v.x);}
  for(int x=floor(xmin-1e-9);x<=floor(xmax+1e-9);x++){
   if(x<0||x>=nc||y<0||y>=nr)return false;
   double z=dem[y*nc+x];if(!std::isfinite(z)||z<=-9999)return false;if(z<zmin-1e-7)continue;
   auto p=clip(clip(strip,0,x,true),0,x+1,false);if(p.empty())continue;
   double low=1e30;for(V v:p)low=std::min(low,v.z);if(z>=low-1e-7)return false;
  }
 }
 return true;
}
extern "C" {
 int rayclear(const float*dem,int nr,int nc,const double*a,const double*b){return lineclear(dem,nr,nc,{a[0],a[1],a[4]},{b[0],b[1],b[4]});}
 int sweptclear(const float*dem,int nr,int nc,const double*a,const double*b,const double*c){return triclear(dem,nr,nc,{a[0],a[1],a[4]},{b[0],b[1],b[4]},{c[0],c[1],c[4]});}
 void pointcover(const float*dem,int nr,int nc,const double*anchor,const double*points,int n,double limit,unsigned char*out){
 for(int i=0;i<n;i++){const double*p=points+5*i;double d=hypot(hypot(anchor[2]-p[2],anchor[3]-p[3]),anchor[4]-p[4]);double fs=32.45+20*log10(2400.)+20*log10(std::max(d,1e-5)/1000.);if(fs+10<=limit-1e-9)out[i]=1;else if(fs>limit-1e-9)out[i]=0;else out[i]=lineclear(dem,nr,nc,{anchor[0],anchor[1],anchor[4]},{p[0],p[1],p[4]});}}
}
'''

class PythonKernel:
 """无编译器时的纯Python核；算法与C++一致，可在Windows/Jupyter复核。"""
 backend='python'
 @staticmethod
 def rayclear(dem,nr,nc,a,b):
  a=np.asarray(a,float);b=np.asarray(b,float);delta=b-a
  if abs(delta[0])+abs(delta[1])<1e-11:
   x,y=int(math.floor(a[0])),int(math.floor(a[1]));return 0<=x<nc and 0<=y<nr and dem[y,x]<min(a[4],b[4])-1e-7
  ts=[0.,1.]
  for ax in [0,1]:
   if abs(delta[ax])>1e-12:
    for k in range(math.ceil(min(a[ax],b[ax])),math.floor(max(a[ax],b[ax]))+1):
     t=(k-a[ax])/delta[ax]
     if 0<t<1:ts.append(t)
  ts=sorted(ts)
  for l,r in zip(ts[:-1],ts[1:]):
   x,y=np.floor((a+delta*((l+r)/2))[:2]).astype(int)
   if x<0 or x>=nc or y<0 or y>=nr:return False
   if dem[y,x]>=min(a[4]+l*delta[4],a[4]+r*delta[4])-1e-7:return False
  for t in ts:
   p=a+delta*t;x,y=np.floor(p[:2]).astype(int)
   for r in range(y-1,y+2):
    for c in range(x-1,x+2):
     if 0<=c<nc and 0<=r<nr and c-1e-9<=p[0]<=c+1+1e-9 and r-1e-9<=p[1]<=r+1+1e-9 and dem[r,c]>=p[4]-1e-7:return False
  return True
 @staticmethod
 def _clip(poly,axis,bound,lower):
  if not poly:return []
  out=[];prev=poly[-1]
  def inside(v):return v[axis]>=bound-1e-10 if lower else v[axis]<=bound+1e-10
  pin=inside(prev)
  for cur in poly:
   inc=inside(cur)
   if inc!=pin:
    den=cur[axis]-prev[axis]
    if abs(den)>1e-16:out.append(prev+(cur-prev)*((bound-prev[axis])/den))
   if inc:out.append(cur)
   prev=cur;pin=inc
  return out
 @classmethod
 def sweptclear(cls,dem,nr,nc,a,b,c):
  det=(b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
  if abs(det)<1e-9:return cls.rayclear(dem,nr,nc,a,b) and cls.rayclear(dem,nr,nc,a,c) and cls.rayclear(dem,nr,nc,b,c)
  tri=[np.array([p[0],p[1],p[4]]) for p in (a,b,c)];zmin=min(p[2] for p in tri)
  for y in range(math.floor(min(p[1] for p in tri)-1e-9),math.floor(max(p[1] for p in tri)+1e-9)+1):
   strip=cls._clip(cls._clip(tri,1,y,True),1,y+1,False)
   if not strip:continue
   for x in range(math.floor(min(p[0] for p in strip)-1e-9),math.floor(max(p[0] for p in strip)+1e-9)+1):
    if x<0 or x>=nc or y<0 or y>=nr:return False
    z=dem[y,x]
    if not np.isfinite(z) or z<=-9999:return False
    if z<zmin-1e-7:continue
    poly=cls._clip(cls._clip(strip,0,x,True),0,x+1,False)
    if poly and z>=min(p[2] for p in poly)-1e-7:return False
  return True
 @classmethod
 def pointcover(cls,dem,nr,nc,anchor,points,n,limit,out):
  for i,p in enumerate(points[:n]):
   d=float(np.linalg.norm(anchor[2:]-p[2:]));fs=FSPL_CONSTANT+20*math.log10(2400.)+20*math.log10(max(d,1e-5)/1000.)
   out[i]=(fs+10<=limit-1e-9 or (fs<=limit-1e-9 and cls.rayclear(dem,nr,nc,anchor,p)))

def get_kernel():
 if os.environ.get('Q3_GEOMETRY_BACKEND','').lower()=='python' or os.name=='nt' or not shutil.which('g++'):return PythonKernel()
 cpp=ROOT/'q3_geometry_kernel.cpp';so=ROOT/'q3_geometry_kernel.so'
 if not cpp.exists() or cpp.read_text(encoding='utf-8')!=CPP or not so.exists():
  cpp.write_text(CPP, encoding='utf-8')
  try:subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(so)],check=True,capture_output=True)
  except (OSError,subprocess.CalledProcessError) as exc:
   warnings.warn('C++加速核不可用，使用等价纯Python连续几何核。');return PythonKernel()
 try:lib=ctypes.CDLL(str(so))
 except OSError:return PythonKernel()
 f32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
 lib.rayclear.argtypes=[f32,ctypes.c_int,ctypes.c_int,f64,f64];lib.sweptclear.argtypes=[f32,ctypes.c_int,ctypes.c_int,f64,f64,f64];lib.pointcover.argtypes=[f32,ctypes.c_int,ctypes.c_int,f64,f64,ctypes.c_int,ctypes.c_double,u8]
 return lib

class Scene:
 def __init__(self):
  self.dem,self.left,self.top,self.dx,self.dy=read_dem(ROOT/'q1_flat/最终工作DEM.tif');self.dem=np.ascontiguousarray(self.dem,dtype=np.float32);self.nr,self.nc=self.dem.shape
  self.nodes,self.boxes,self.models,_,_=read_inputs();self.kernel=get_kernel()
  self.nodepos={s:self.position(n.longitude_deg,n.latitude_deg,n.operating_altitude_m) for s,n in self.nodes.iterrows()}
  n=self.nodes.loc['O01'];self.gateway=self.position(n.longitude_deg,n.latitude_deg,n.ground_elevation_m+20.)
 def position(self,lon,lat,z):
  e,n=utm49(float(lon),float(lat));return np.array([(lon-self.left)/self.dx,(self.top-lat)/self.dy,e,n,z],dtype=float)
 def ground(self,lon,lat):
  x,y=int((lon-self.left)/self.dx),int((self.top-lat)/self.dy)
  if x<0 or x>=self.nc or y<0 or y>=self.nr:return float('nan')
  return float(self.dem[y,x])
 def point_cover(self,anchor,points,kind='access'):
  p=np.ascontiguousarray(points,dtype=float);o=np.zeros(len(p),np.uint8);self.kernel.pointcover(self.dem,self.nr,self.nc,np.ascontiguousarray(anchor),p,len(p),LIMIT.get(kind,kind) if isinstance(kind,str) else kind,o);return o.astype(bool)
 def certify_link(self,anchor,p0,p1,threshold_db):
  a,p0,p1=map(lambda x:np.ascontiguousarray(x,dtype=float),(anchor,p0,p1))
  d=max(np.linalg.norm(a[2:]-p0[2:]),np.linalg.norm(a[2:]-p1[2:]));fs=FSPL_CONSTANT+20*math.log10(2400)+20*math.log10(max(d,1e-5)/1000)
  if fs+10<=threshold_db-1e-9:return True,threshold_db-fs-10,'遮挡预算仍通过'
  if fs>threshold_db-1e-9:return False,threshold_db-fs,'距离超限'
  clear=bool(self.kernel.sweptclear(self.dem,self.nr,self.nc,a,p0,p1));return clear,threshold_db-fs-(0 if clear else 10),'全区间视距' if clear else '未获全区间证书'
 def station(self,sid,lon,lat,agl):
  ground=self.ground(lon,lat);p=self.position(lon,lat,ground+agl);ok,margin,why=self.certify_link(self.gateway,p,p,LIMIT['backhaul']);a=self.nodepos['O01'];cells=touched_cells(a[0],a[1],p[0],p[1],self.nc,self.nr);peak=max(float(self.dem[r,c]) for r,c in cells)
  return dict(station_id=sid,lon=lon,lat=lat,east=p[2],north=p[3],px=p[0],py=p[1],ground_z=ground,z=p[4],agl=agl,distance_m=float(np.linalg.norm(a[2:4]-p[2:4])),line_max_z=peak,backhaul_ok=ok,backhaul_margin_db=margin)

POS=['px','py','east','north','z']
def get_points(df,prefix):return np.ascontiguousarray(df[[prefix+'_'+c for c in POS]].to_numpy(float))

def build_trajectory(scene=None,prefix=None,dt=5.,frames=None):
 scene=scene or Scene();prefix=Path(prefix or ROOT/'全局认证_冻结最终十九架次');F,L=frames if frames is not None else (pd.read_csv(str(prefix)+'_架次.csv'),pd.read_csv(str(prefix)+'_航段.csv'));rows=[]
 for _,r in F.iterrows():
  g=scene.models.loc[r['机型']];t=float(r['开始秒'])+g.fixed_prep_s+g.load_per_box_s*len(r['货箱编号'].split(','));route=r['架次编号'];ids=r['货箱编号'].split(',')
  def phase(end,a,b,name,src,dst):
   nonlocal t
   if end<t-1e-8:raise ValueError('negative time')
   if end-t>1e-10:
    times=np.linspace(t,end,max(1,int(math.ceil((end-t)/dt)))+1)
    for u,v in zip(times[:-1],times[1:]):
     p0=a+(b-a)*(u-t)/(end-t);p1=a+(b-a)*(v-t)/(end-t)
     row=dict(id=len(rows),route=route,model=r['机型'],start_s=u,end_s=v,relative_start_s=u-r['开始秒'],relative_end_s=v-r['开始秒'],stage=name,from_node=src,to_node=dst)
     row.update({f'p0_{c}':p0[k] for k,c in enumerate(POS)});row.update({f'p1_{c}':p1[k] for k,c in enumerate(POS)});rows.append(row)
   t=float(end)
  for _,leg in L[L['架次编号'].eq(route)].iterrows():
   src,dst=leg['起点'],leg['终点'];a=scene.nodepos[src].copy();b=scene.nodepos[dst].copy();up=a.copy();up[4]=leg['巡航海拔m'];down=b.copy();down[4]=up[4];start=t
   phase(t+leg['爬升m']/g.climb_speed_m_s,a,up,'爬升',src,dst)
   phase(t+leg['距离m']/g.cruise_speed_m_s,up,down,'巡航',src,dst)
   phase(t+leg['下降m']/g.descent_speed_m_s,down,b,'下降',src,dst)
   phase(start+leg['飞行秒'],b,b,'整数秒等待',src,dst)
   if dst!='O01':
    count=sum(scene.boxes.loc[bid,'node_id']==dst for bid in ids);phase(t+g.handoff_base_s+g.handoff_per_box_s*count,b,b,'交接',src,dst)
  assert abs(t-r['返回秒'])<1e-5,(route,t,r['返回秒'])
 d=pd.DataFrame(rows);p0=get_points(d,'p0');p1=get_points(d,'p1');cert=[scene.certify_link(scene.gateway,a,b,LIMIT['direct']) for a,b in zip(p0,p1)];d['direct_ok']=[x[0] for x in cert];d['direct_margin_lower_db']=[x[1] for x in cert];d['direct_certificate']=[x[2] for x in cert]
 d['direct_sample_start_ok']=scene.point_cover(scene.gateway,p0,'direct');d['direct_sample_mid_ok']=scene.point_cover(scene.gateway,(p0+p1)/2,'direct');d['direct_sample_end_ok']=scene.point_cover(scene.gateway,p1,'direct')
 return d

def build_key_trajectory(scene,keys,context=None,dt=5.):
 """任意给定合法路线池的相对时间轨迹；route=P00000等，准备开始为0。"""
 from 问题二_地形ALNS_CPSAT统一实验 import Engine
 c=context or json.loads((ROOT/'问题二_统一实验输入.json').read_text(encoding='utf-8'));eng=Engine(c);flights=[];legs=[]
 for j,raw in enumerate(keys):
  key=(int(raw[0]),tuple(tuple(map(int,g)) for g in raw[1]));r=eng.evaluate(key)
  if r is None:raise ValueError(f'invalid route {j}')
  rid=f'P{j:05d}';flights.append({'架次编号':rid,'机型':'ABC'[r.g],'开始秒':0,'返回秒':r.duration,'货箱编号':','.join(c['boxids'][b] for b in r.boxes)})
  src=0
  for dst in (*r.sites,0):
   a=c['arc'][src][dst];legs.append({'架次编号':rid,'起点':c['nodes'][src],'终点':c['nodes'][dst],'距离m':a[0],'爬升m':a[1],'下降m':a[2],'巡航海拔m':a[3],'飞行秒':eng.ft[r.g,src,dst]});src=dst
 return build_trajectory(scene,dt=dt,frames=(pd.DataFrame(flights),pd.DataFrame(legs)))

def certify_pool(pool_json=None):
 """读取已认证25站，导出所有候选路线相对盲区和站点兼容列表。"""
 scene=Scene();data=json.loads(Path(pool_json or ROOT/'问题三_有限路线池.json').read_text(encoding='utf-8'));keys=data['keys'];df=build_key_trajectory(scene,keys);st=pd.read_csv(ROOT/'q3_geometry_stations.csv');p0=get_points(df,'p0');p1=get_points(df,'p1');blind=df.index[~df.direct_ok];mask=np.ones((len(st),len(df)),bool)
 for j,s in st.iterrows():
  a=s[POS].to_numpy(float)
  for k in blind:mask[j,k]=scene.certify_link(a,p0[k],p1[k],LIMIT['access'])[0]
  print('pool station',s.station_id,'certified',int(mask[j,blind].sum()),'/',len(blind),flush=True)
 out=[]
 for j in range(len(keys)):
  sub=df[df.route.eq(f'P{j:05d}')];intervals=[]
  for k,r in sub[~sub.direct_ok].iterrows():
   stations=st.loc[mask[:,k],'station_id'].tolist();lo=int(math.floor(r.start_s+1e-8));hi=int(math.ceil(r.end_s-1e-8))
   if intervals and intervals[-1]['stations']==stations and lo<=intervals[-1]['hi']:intervals[-1]['hi']=max(hi,intervals[-1]['hi'])
   else:intervals.append(dict(lo=lo,hi=hi,stations=stations))
  out.append(dict(pool_index=j,blind_intervals=intervals))
 result=dict(routes=out,station_ids=st.station_id.tolist(),continuous_certificate=True,dt_max_s=5.,uncovered_intervals=int((~mask[:,blind].any(axis=0)).sum()))
 (ROOT/'q3_geometry_pool_blind.json').write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding='utf-8');df.to_csv(ROOT/'q3_geometry_pool_intervals.csv',index=False);np.savez_compressed(ROOT/'q3_geometry_pool_coverage.npz',mask=mask,station_ids=st.station_id.to_numpy(str),interval_ids=df.id.to_numpy(int));print('POOL_DONE',len(keys),result['uncovered_intervals'],flush=True)
 return result

def candidates(scene,step_m=300.):
 n=scene.nodes;lon0,lon1=n.longitude_deg.min()-.006,n.longitude_deg.max()+.006;lat0,lat1=n.latitude_deg.min()-.006,n.latitude_deg.max()+.006
 dl=step_m/(111320*math.cos(math.radians(n.latitude_deg.mean())));da=step_m/110750
 xy=[(x,y) for x in np.arange(lon0,lon1,dl) for y in np.arange(lat0,lat1,da)]
 # Include DEM high points in 600m blocks, and task node sites.
 rows=[]
 for lon,lat in xy:
  for agl in [100.,200.,300.]:
   d=scene.station(f'J{len(rows):05d}',float(lon),float(lat),agl)
   if d['backhaul_ok']:rows.append(d)
 return pd.DataFrame(rows)

def main():
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--prefix');ap.add_argument('--step',type=float,default=300.);ap.add_argument('--dt',type=float,default=5.);ap.add_argument('--keep',type=int,default=24);args=ap.parse_args();scene=Scene()
 d=build_trajectory(scene,args.prefix,args.dt);d.to_csv(ROOT/'q3_geometry_intervals.csv',index=False);blind=d[~d.direct_ok];print('intervals',len(d),'needs relay',len(blind),'time',float((blind.end_s-blind.start_s).sum()),flush=True)
 st=candidates(scene,args.step);print('candidates',len(st),flush=True)
 # Representative physical endpoints, not artificial values. Deterministic stratification.
 endpoints=np.r_[get_points(blind,'p0'),get_points(blind,'p1')];endpoints=np.unique(np.round(endpoints,8),axis=0)
 if len(endpoints)>500:endpoints=endpoints[np.linspace(0,len(endpoints)-1,500).astype(int)]
 pointmask=[]
 for j,(_,s) in enumerate(st.iterrows()):
  p=s[POS].to_numpy(float);pointmask.append(scene.point_cover(p,endpoints))
 pm=np.asarray(pointmask);st['sample_cover_count']=pm.sum(axis=1);print('best sampled',st.sample_cover_count.max(),'of',len(endpoints),flush=True)
 # Retain greedy residual-cover stations plus top alternatives by coverage and low altitude.
 chosen=[];uncovered=np.ones(len(endpoints),bool)
 for _ in range(min(6,len(st))):
  gain=(pm[:,uncovered]).sum(axis=1);j=int(gain.argmax());
  if gain[j]==0:break
  chosen.append(j);uncovered &=~pm[j]
  if not uncovered.any():break
 order=st.sort_values(['sample_cover_count','agl','distance_m'],ascending=[False,True,True]).index.tolist()
 # Keep geographically diverse alternatives to avoid one tight cluster.
 for j in order:
  if j in chosen:continue
  p=st.loc[j];
  if not chosen or min(math.hypot(p.east-st.loc[k].east,p.north-st.loc[k].north)+abs(p.agl-st.loc[k].agl)*2 for k in chosen)>280:chosen.append(j)
  if len(chosen)>=args.keep:break
 st['selected_for_certificate']=False;st.loc[chosen,'selected_for_certificate']=True;st.to_csv(ROOT/'q3_geometry_candidates_all.csv',index=False)
 selected=st.loc[chosen].copy().reset_index(drop=True);p0=get_points(d,'p0');p1=get_points(d,'p1');coverage=np.zeros((len(selected),len(d)),bool)
 for j,s in selected.iterrows():
  anchor=s[POS].to_numpy(float);coverage[j,d.direct_ok.to_numpy()]=True
  for k in blind.index:coverage[j,k]=scene.certify_link(anchor,p0[k],p1[k],LIMIT['access'])[0]
  print('certified',s.station_id,int(coverage[j,blind.index].sum()),'/',len(blind),flush=True)
 # Certify residual-driven additional candidates; sparse screening must not omit stationary service demands.
 for extra in range(12):
  rem=blind.index[~coverage[:,blind.index].any(axis=0)]
  if not len(rem):break
  ep=np.unique(np.round(np.r_[p0[rem],p1[rem]],8),axis=0)
  score=[]
  for _,s in st.iterrows():score.append(int(scene.point_cover(s[POS].to_numpy(float),ep).sum()))
  ranked=np.argsort(score)[::-1];added=False
  for k in ranked[:40]:
   s=st.iloc[k]
   if s.station_id in set(selected.station_id):continue
   anchor=s[POS].to_numpy(float);cov=np.array([scene.certify_link(anchor,a,b,LIMIT['access'])[0] if not direct else True for a,b,direct in zip(p0,p1,d.direct_ok)],bool)
   if not cov[rem].any():continue
   selected=pd.concat([selected,s.to_frame().T],ignore_index=True);coverage=np.r_[coverage,cov[None,:]];added=True
   print('residual candidate',s.station_id,'covers',int(cov[rem].sum()),'of',len(rem),flush=True);break
  if not added:break
 selected['certified_blind_cover_count']=coverage[:,blind.index].sum(axis=1);selected.to_csv(ROOT/'q3_geometry_stations.csv',index=False)
 np.savez_compressed(ROOT/'q3_geometry_coverage.npz',mask=coverage,station_ids=selected.station_id.to_numpy(str),interval_ids=d.id.to_numpy(int))
 uncovered=blind.index[~coverage[:,blind.index].any(axis=0)];report=dict(intervals=len(d),blind_intervals=len(blind),blind_transport_seconds=float((blind.end_s-blind.start_s).sum()),candidate_count=len(st),certified_stations=len(selected),uncovered_intervals=uncovered.tolist(),max_dt_s=float((d.end_s-d.start_s).max()),fspl_constant=FSPL_CONSTANT,link_limits=LIMIT,scope='continuous conservative sufficient certificate on each swept ray triangle; unproved intervals are not automatically physically disconnected')
 (ROOT/'q3_geometry_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding='utf-8');print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
