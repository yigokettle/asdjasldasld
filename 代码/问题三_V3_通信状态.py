"""问题三V3：由解析地形遮挡事件和距离根恢复实际直连优先通信状态。
不修改原调度的保守通信约束。每个线性运动阶段内，令u=s*t，射线扫掠
是参数三角形0<=u<=s<=1。各DEM像元柱体与其相交后，顶点u/s极值给出
遮挡时段；取并集后和LOS/NLOS距离阈值根共同划分状态。切换瞬时另表记录。
"""
from pathlib import Path
import argparse, ctypes, hashlib, json, math, os, shutil, subprocess, sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from 问题三_通信几何 import Scene,build_trajectory,get_points,POS,LIMIT,FSPL_CONSTANT
# 与旧核一致，触地以及距地形1e-7m以内视作遮挡；不是平滑地形。
HEIGHT_EPS=1e-7
EVENT_EPS=1e-12
CPP=r'''
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
using P=std::array<double,2>;
static std::vector<P> clip(const std::vector<P>& p,double a,double b,double c){
 std::vector<P>o;if(p.empty())return o;P prev=p.back();double vp=a*prev[0]+b*prev[1]+c;bool pin=vp>=0;
 for(P cur:p){double vc=a*cur[0]+b*cur[1]+c;bool in=vc>=0;if(in!=pin){double den=vp-vc;if(fabs(den)>1e-30){double w=vp/den;o.push_back({prev[0]+w*(cur[0]-prev[0]),prev[1]+w*(cur[1]-prev[1])});}}if(in)o.push_back(cur);prev=cur;vp=vc;pin=in;}return o;
}
extern "C" int blockevents(const float*dem,int nr,int nc,const double*A,const double*B,const double*C,double*out,int capacity){
 double ax=A[0],ay=A[1],az=A[4],bx=B[0]-ax,by=B[1]-ay,bz=B[4]-az,dx=C[0]-B[0],dy=C[1]-B[1],dz=C[4]-B[4];
 double minz=std::min({az,B[4],C[4]});std::vector<P>times;
 // Raster rows are pruned with the projected swept triangle (possibly degenerate).
 std::vector<P>xy{{A[0],A[1]},{B[0],B[1]},{C[0],C[1]}};
 int ya=floor(std::min({A[1],B[1],C[1]})-1e-9),yb=floor(std::max({A[1],B[1],C[1]})+1e-9);
 for(int y=ya;y<=yb;y++){
  auto strip=clip(clip(xy,0,1,-y+1e-10),0,-1,y+1+1e-10);if(strip.empty())continue;
  double xmin=1e30,xmax=-1e30;for(P p:strip){xmin=std::min(xmin,p[0]);xmax=std::max(xmax,p[0]);}
  for(int x=floor(xmin-1e-9);x<=floor(xmax+1e-9);x++){
   if(x<0||x>=nc||y<0||y>=nr)return -2;
   double h=dem[y*nc+x];if(!std::isfinite(h)||h<=-9999)return -3;if(h+1e-7<minz)continue;
   if(ax>=x-1e-10&&ax<=x+1+1e-10&&ay>=y-1e-10&&ay<=y+1+1e-10&&az<=h+1e-7){out[0]=0;out[1]=1;return 1;}
   std::vector<P>p{{0,0},{1,0},{1,1}};
   p=clip(p,bx,dx,ax-x+1e-10);p=clip(p,-bx,-dx,x+1-ax+1e-10);
   p=clip(p,by,dy,ay-y+1e-10);p=clip(p,-by,-dy,y+1-ay+1e-10);
   p=clip(p,-bz,-dz,h+1e-7-az);if(p.empty())continue;
   double lo=1e30,hi=-1e30;
   for(P v:p)if(v[0]>1e-14){double t=v[1]/v[0];lo=std::min(lo,t);hi=std::max(hi,t);}
   if(hi>=lo&&hi>=-1e-12&&lo<=1+1e-12)times.push_back({std::max(0.,lo),std::min(1.,hi)});
  }
 }
 std::sort(times.begin(),times.end(),[](P a,P b){return a[0]<b[0]||(a[0]==b[0]&&a[1]<b[1]);});
 std::vector<P>u;for(P t:times){if(!u.empty()&&t[0]<=u.back()[1]+1e-13)u.back()[1]=std::max(u.back()[1],t[1]);else u.push_back(t);}
 if((int)u.size()>capacity)return -(10000+(int)u.size());for(int i=0;i<(int)u.size();i++){out[2*i]=u[i][0];out[2*i+1]=u[i][1];}return u.size();
}
'''

def _clip(poly,a,b,c):
 if not poly:return []
 out=[];prev=poly[-1];vp=a*prev[0]+b*prev[1]+c;pin=vp>=0
 for cur in poly:
  vc=a*cur[0]+b*cur[1]+c;inc=vc>=0
  if inc!=pin:
   den=vp-vc
   if abs(den)>1e-30:out.append(prev+(cur-prev)*(vp/den))
  if inc:out.append(cur)
  prev=cur;vp=vc;pin=inc
 return out

def python_blocks(dem,a,b,c):
 nr,nc=dem.shape;ab=b-a;bc=c-b;minz=min(a[4],b[4],c[4]);out=[];xy=[a[:2],b[:2],c[:2]]
 for y in range(math.floor(min(p[1] for p in xy)-1e-9),math.floor(max(p[1] for p in xy)+1e-9)+1):
  strip=_clip(_clip(xy,0,1,-y+1e-10),0,-1,y+1+1e-10)
  if not strip:continue
  for x in range(math.floor(min(p[0] for p in strip)-1e-9),math.floor(max(p[0] for p in strip)+1e-9)+1):
   if not (0<=x<nc and 0<=y<nr):raise ValueError('communication ray outside DEM')
   h=float(dem[y,x])
   if not np.isfinite(h) or h<=-9999:raise ValueError('DEM NoData on ray')
   if h+HEIGHT_EPS<minz:continue
   if x-1e-10<=a[0]<=x+1+1e-10 and y-1e-10<=a[1]<=y+1+1e-10 and a[4]<=h+HEIGHT_EPS:return [(0.,1.)]
   poly=[np.array([0.,0.]),np.array([1.,0.]),np.array([1.,1.])]
   for ca,cb,cc in [(ab[0],bc[0],a[0]-x+1e-10),(-ab[0],-bc[0],x+1-a[0]+1e-10),(ab[1],bc[1],a[1]-y+1e-10),(-ab[1],-bc[1],y+1-a[1]+1e-10),(-ab[4],-bc[4],h+HEIGHT_EPS-a[4])]:poly=_clip(poly,ca,cb,cc)
   ts=[v[1]/v[0] for v in poly if v[0]>1e-14]
   if ts and max(ts)>=-EVENT_EPS and min(ts)<=1+EVENT_EPS:out.append((max(0.,min(ts)),min(1.,max(ts))))
 merged=[]
 for lo,hi in sorted(out):
  if merged and lo<=merged[-1][1]+1e-13:merged[-1]=(merged[-1][0],max(hi,merged[-1][1]))
  else:merged.append((lo,hi))
 return merged

class EventKernel:
 def __init__(self,scene,backend='auto'):
  self.scene=scene;self.backend='python';self.lib=None
  if backend=='python' or os.name=='nt' or not shutil.which('g++'):return
  cpp=ROOT/'q3_v3_communication_events.cpp';so=ROOT/'q3_v3_communication_events.so'
  try:
   if not cpp.exists() or cpp.read_text(encoding='utf-8')!=CPP or not so.exists():
    cpp.write_text(CPP, encoding='utf-8');subprocess.run(['g++','-O3','-std=c++17','-shared','-fPIC',str(cpp),'-o',str(so)],check=True,capture_output=True)
   self.lib=ctypes.CDLL(str(so));arr32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS');arr64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS');self.lib.blockevents.argtypes=[arr32,ctypes.c_int,ctypes.c_int,arr64,arr64,arr64,arr64,ctypes.c_int];self.backend='cpp'
  except (OSError,subprocess.CalledProcessError):self.lib=None
 def blocks(self,a,b,c,dem=None):
  dem=self.scene.dem if dem is None else np.ascontiguousarray(dem,np.float32);a,b,c=(np.ascontiguousarray(v,float) for v in (a,b,c))
  if self.lib is None:return python_blocks(dem,a,b,c)
  out=np.empty((4096,2),float);n=self.lib.blockevents(dem,*dem.shape,a,b,c,out,len(out))
  if n<0:raise ValueError(f'analytic geometry kernel error {n}')
  return [(float(lo),float(hi)) for lo,hi in out[:n]]

class Link:
 def __init__(self,kernel,anchor,p0,p1,limit):
  self.a=anchor;self.p=p0;self.v=p1-p0;self.limit=float(limit);self.blocks=kernel.blocks(anchor,p0,p1)
  self.events=[0.,1.]+[x for pair in self.blocks for x in pair]
  w=p0[2:]-anchor[2:];v=self.v[2:];aa=float(v@v);bb=float(2*w@v)
  self.distance_roots=[]
  for penalty in [0.,10.]:
   radius=1000*10**((limit-penalty-FSPL_CONSTANT-20*math.log10(2400.))/20)
   cc=float(w@w-radius*radius)
   if aa>1e-20:
    disc=bb*bb-4*aa*cc
    if disc>=-1e-7:
     sd=math.sqrt(max(0.,disc))
     for t in [(-bb-sd)/(2*aa),(-bb+sd)/(2*aa)]:
      if 0<t<1:self.events.append(t);self.distance_roots.append((float(t),penalty))
 def blocked(self,t):return any(lo<=t<=hi for lo,hi in self.blocks)
 def margin(self,t):
  d=float(np.linalg.norm(self.p[2:]+t*self.v[2:]-self.a[2:]));loss=FSPL_CONSTANT+20*math.log10(2400.)+20*math.log10(max(d,1e-5)/1000.)+(10. if self.blocked(t) else 0.)
  return self.limit-loss
 def interval_margin(self,lo,hi):
  penalty=10. if self.blocked((lo+hi)/2) else 0.;ds=[np.linalg.norm(self.p[2:]+t*self.v[2:]-self.a[2:]) for t in [lo,hi]]
  return self.limit-(FSPL_CONSTANT+20*math.log10(2400.)+20*math.log10(max(max(ds),1e-5)/1000.)+penalty)

def point_link(scene,a,p,limit):
 blocked=not bool(scene.kernel.rayclear(scene.dem,scene.nr,scene.nc,np.ascontiguousarray(a),np.ascontiguousarray(p)))
 d=float(np.linalg.norm(a[2:]-p[2:]));margin=limit-(FSPL_CONSTANT+20*math.log10(2400.)+20*math.log10(max(d,1e-5)/1000.)+(10. if blocked else 0.))
 return margin>=-1e-8,margin,blocked

FIELDS=['运输架次编号','通信阶段','开始时刻（s）','结束时刻（s）','保障方式','中继架次编号']
def submission_boundaries(formal,events):
 """正时长采用[start,end)，仅将不符合该默认选边的瞬时状态补为零时长行。"""
 supplements=[];seen={};isolated_modes=0;left_mode_exceptions=0;right_mode_exceptions=0
 for _,e in events.sort_values(['运输架次编号','时刻（s）']).iterrows():
  rid=e['运输架次编号'];t=float(e['时刻（s）']);key=(rid,round(t,8));v=(str(e['保障方式']),str(e['中继架次编号']) if pd.notna(e['中继架次编号']) else '')
  if key in seen:
   if seen[key]!=v:raise AssertionError(('同一物理时刻边界状态不一致',key,seen[key],v))
   continue
  seen[key]=v;rows=formal[formal['运输架次编号'].eq(rid)];tol=1e-8
  right=rows[(rows['开始时刻（s）']<=t+tol)&(rows['结束时刻（s）']>t+tol)]
  left=rows[(rows['开始时刻（s）']<t-tol)&(rows['结束时刻（s）']>=t-tol)]
  def state(row):return (str(row['保障方式']),str(row['中继架次编号']) if pd.notna(row['中继架次编号']) else '')
  vr=state(right.iloc[0]) if len(right) else None;vl=state(left.iloc[-1]) if len(left) else None
  # 每条路线最终返回瞬时没有右侧区间，沿最后一区间；若实际事件不同同样补零行。
  vr=vr if vr is not None else vl;vl=vl if vl is not None else vr
  if vr is None:raise AssertionError(('事件没有相邻通信区间',rid,t))
  isolated_modes+=int(v[0]!=vr[0] and v[0]!=vl[0]);right_mode_exceptions+=int(v[0]!=vr[0]);left_mode_exceptions+=int(v[0]!=vl[0])
  if v!=vr:supplements.append(dict(zip(FIELDS,[rid,e['通信阶段'],t,t,*v])))
 points=pd.DataFrame(supplements,columns=FIELDS);combined=pd.concat([formal,points],ignore_index=True).sort_values(['运输架次编号','开始时刻（s）','结束时刻（s）'],kind='stable').reset_index(drop=True)
 submission_mismatches=0
 for (rid,t),v in seen.items():
  point=points[(points['运输架次编号']==rid)&(np.abs(points['开始时刻（s）']-t)<2e-8)]
  if len(point):got=state(point.iloc[0])
  else:
   rows=formal[formal['运输架次编号']==rid];right=rows[(rows['开始时刻（s）']<=t+1e-8)&(rows['结束时刻（s）']>t+1e-8)]
   got=state(right.iloc[0]) if len(right) else state(rows.sort_values('结束时刻（s）').iloc[-1])
  submission_mismatches+=int(got!=v)
 if submission_mismatches:raise AssertionError(('六列提交表未复现精确边界',submission_mismatches))
 return points,combined,dict(submission_boundary_mismatches=submission_mismatches,unique_physical_boundary_events=len(seen),right_half_open_mode_exceptions=right_mode_exceptions,left_half_open_mode_exceptions=left_mode_exceptions,isolated_mode_events=isolated_modes,supplementary_zero_duration_records=len(points),submission_records=len(combined),boundary_dedup_tolerance_seconds=1e-8)

def run(prefix,output_prefix=None,backend='auto'):
 base=Path(prefix);base=base if base.is_absolute() else ROOT/base;out=Path(output_prefix) if output_prefix else Path(str(base)+'_V3');out=out if out.is_absolute() else ROOT/out
 scene=Scene();kernel=EventKernel(scene,backend);transport=str(base)+'_运输';flights=pd.read_csv(transport+'_架次.csv');relays=pd.read_csv(str(base)+'_中继架次.csv')
 anchors={};backhaul={}
 for rr in relays.to_dict('records'):
  a=scene.position(rr['经度'],rr['纬度'],rr['悬停海拔m']);sid=rr['中继架次'];anchors[sid]=a;ok,margin,_=point_link(scene,scene.gateway,a,LIMIT['backhaul']);assert ok,(sid,'backhaul');backhaul[sid]=margin
 # Whole physical phases, not fixed-time sampling, are the domains of analytic events.
 phases=build_trajectory(scene,transport,dt=1e9);p0=get_points(phases,'p0');p1=get_points(phases,'p1');raw=[];boundaries=[];failures=[];mismatch=[];minmargin=1e20;events_count=0
 for k,r in phases.iterrows():
  start,end=float(r.start_s),float(r.end_s);duration=end-start;direct=Link(kernel,scene.gateway,p0[k],p1[k],LIMIT['direct']);links={};active=[];cuts=list(direct.events)
  for rr in relays.to_dict('records'):
   if rr['建链完成秒']<=end+1e-10 and rr['服务结束秒']>=start-1e-10:
    sid=rr['中继架次'];links[sid]=Link(kernel,anchors[sid],p0[k],p1[k],LIMIT['access']);active.append(rr);cuts.extend(links[sid].events)
    cuts.extend((t-start)/duration for t in [rr['建链完成秒'],rr['服务结束秒']] if start<t<end)
  cuts=sorted(set(float(np.clip(t,0,1)) for t in cuts));events_count+=len(cuts)
  for lo,hi in zip(cuts[:-1],cuts[1:]):
   if hi-lo<=EVENT_EPS:continue
   mid=(lo+hi)/2;t=start+duration*mid;mode='直连';sid='';link=direct;mar=direct.margin(mid)
   independently_ok,_,_=point_link(scene,scene.gateway,p0[k]+mid*(p1[k]-p0[k]),LIMIT['direct'])
   if (mar>=-1e-8)!=independently_ok:mismatch.append(dict(route=r.route,phase=k,t=t,analytic_margin=mar,point_ok=independently_ok))
   if mar< -1e-8:
    possible=[(min(links[q['中继架次']].margin(mid),backhaul[q['中继架次']]),q) for q in active if q['建链完成秒']<=t<=q['服务结束秒'] and links[q['中继架次']].margin(mid)>=-1e-8]
    if possible:
     mar,rr=max(possible,key=lambda v:(v[0],v[1]['中继架次']));sid=rr['中继架次'];mode='中继';link=links[sid]
    else:mode='中断';failures.append(dict(route=r.route,start=start+lo*duration,end=start+hi*duration,stage=r.stage))
   lower=link.interval_margin(lo,hi)
   if sid:lower=min(lower,backhaul[sid])
   minmargin=min(minmargin,lower)
   v0=p0[k]+lo*(p1[k]-p0[k]);v1=p0[k]+hi*(p1[k]-p0[k])
   geometry={'id':len(raw),'route':r.route,'model':r.model,'stage':r.stage,'start_s':start+lo*duration,'end_s':start+hi*duration,'direct_ok':mode=='直连','relay_id':sid,'from_node':r.from_node,'to_node':r.to_node}
   geometry.update({f'p0_{c}':float(v0[j]) for j,c in enumerate(POS)});geometry.update({f'p1_{c}':float(v1[j]) for j,c in enumerate(POS)})
   for name,v in [('p0',v0),('p1',v1)]:geometry[name+'_lon']=scene.left+v[0]*scene.dx;geometry[name+'_lat']=scene.top-v[1]*scene.dy
   raw.append(dict(zip(FIELDS,[r.route,r.stage,start+lo*duration,start+hi*duration,mode,sid]))|geometry|{'解析阶段索引':int(k),'区间链路裕量下界dB':lower,'直连中点裕量dB':direct.margin(mid),'状态区间依据':'解析遮挡集合+两类距离阈值根','区间边界约定':'开区间内状态恒定，边界瞬时见事件表'})
  # Point-only equality/contact states are explicitly classified independently at each cut.
  for u in cuts:
   t=start+duration*u;p=p0[k]+u*(p1[k]-p0[k]);ok,mar,blocked=point_link(scene,scene.gateway,p,LIMIT['direct']);mode='直连';sid=''
   if not ok:
    choices=[]
    for rr in active:
     if rr['建链完成秒']-1e-8<=t<=rr['服务结束秒']+1e-8:
      good,m,_=point_link(scene,anchors[rr['中继架次']],p,LIMIT['access'])
      if good:choices.append((min(m,backhaul[rr['中继架次']]),rr['中继架次']))
    if choices:mar,sid=max(choices);mode='中继'
    else:mode='中断';failures.append(dict(route=r.route,time=t,stage=r.stage,point_only=True))
   boundaries.append({'运输架次编号':r.route,'通信阶段':r.stage,'时刻（s）':t,'保障方式':mode,'中继架次编号':sid,'直连遮挡':blocked,'所选链路裕量dB':mar,'解析阶段索引':int(k)})
 detailed=pd.DataFrame(raw);events=pd.DataFrame(boundaries).drop_duplicates(['运输架次编号','通信阶段','时刻（s）']).sort_values(['运输架次编号','时刻（s）'])
 merged=[]
 for rec in raw:
  cur={key:rec[key] for key in FIELDS}
  if merged and all(merged[-1][f]==cur[f] for f in ['运输架次编号','通信阶段','保障方式','中继架次编号']) and abs(merged[-1]['结束时刻（s）']-cur['开始时刻（s）'])<1e-8:merged[-1]['结束时刻（s）']=cur['结束时刻（s）']
  else:merged.append(cur)
 formal=pd.DataFrame(merged,columns=FIELDS)
 points,submission,boundary_summary=submission_boundaries(formal,events)
 for f in flights.to_dict('records'):
  c=formal[formal['运输架次编号'].eq(f['架次编号'])].sort_values('开始时刻（s）');begin=f['开始秒']+300+30*len(f['货箱编号'].split(','));assert abs(c.iloc[0]['开始时刻（s）']-begin)<1e-6;assert abs(c.iloc[-1]['结束时刻（s）']-f['返回秒'])<1e-6;assert np.max(np.abs(c['结束时刻（s）'].to_numpy()[:-1]-c['开始时刻（s）'].to_numpy()[1:]),initial=0)<1e-6
 # Audit the old conservative record to quantify exact-priority correction, if available.
 corrected=0.;oldfile=Path(str(base)+'_通信保障.csv')
 if oldfile.exists():
  old=pd.read_csv(oldfile)
  for _,o in old[old['保障方式'].eq('中继')].iterrows():
   exact=formal[(formal['运输架次编号']==o['运输架次编号'])&(formal['保障方式']=='直连')]
   corrected+=sum(max(0.,min(o['结束时刻（s）'],e['结束时刻（s）'])-max(o['开始时刻（s）'],e['开始时刻（s）'])) for _,e in exact.iterrows())
 seconds={mode:float((formal.loc[formal['保障方式']==mode,'结束时刻（s）']-formal.loc[formal['保障方式']==mode,'开始时刻（s）']).sum()) for mode in ['直连','中继','中断']}
 report=dict(status='通过' if not failures and not mismatch else '失败',source_prefix=str(base),output_prefix=str(out),method='分段线性轨迹下逐DEM像元柱体的(s,u)凸多边形裁切，解析投影遮挡时间集合；LOS/NLOS距离阈值二次根；状态边界瞬时独立分类；实际直连优先',physical_phases=len(phases),analytic_atomic_intervals=len(detailed),formal_records=len(formal),boundary_events=len(events),communication_seconds=seconds,old_relay_seconds_corrected_to_direct=corrected,direct_priority_mismatches=len(mismatch),point_midpoint_crosschecks=len(detailed),failed_coverage_count=len(failures),failures=failures[:30],classification_mismatches=mismatch[:30],minimum_open_interval_link_margin_db=minmargin,minimum_boundary_selected_margin_db=float(events['所选链路裕量dB'].min()),backend=kernel.backend,numerical_tolerance=dict(terrain_contact_m=HEIGHT_EPS,parameter_event_merge=EVENT_EPS,link_margin_db=1e-8),boundary_convention='通信提交记录为六列正时长区间加必要零时长瞬时记录。正时长采用[start,end)，零时长行在对应t优先；各运输架次最后返回端点沿左侧状态，若实际状态不同亦有零时长覆盖。零行不计累计时长。绘图使用通信保障正时长表。',boundary_summary=boundary_summary,dem_sha256=hashlib.sha256((ROOT/'q1_flat/最终工作DEM.tif').read_bytes()).hexdigest(),source_hashes={suffix:hashlib.sha256(Path(str(base)+suffix).read_bytes()).hexdigest() for suffix in ['_运输_架次.csv','_运输_航段.csv','_中继架次.csv']})
 for suffix,frame in [('通信保障',formal),('通信提交记录',submission),('通信瞬时补充记录',points),('通信解析区间',detailed),('通信边界事件',events),('通信物理阶段',phases)]:frame.to_csv(str(out)+'_'+suffix+'.csv',index=False,encoding='utf-8-sig')
 Path(str(out)+'_连续通信复核.json').write_text(json.dumps(report,ensure_ascii=False,indent=2), encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2));return report

def selftest(interval_file=None,backend='auto'):
 scene=Scene();kernel=EventKernel(scene,backend);rng=np.random.default_rng(20260924)
 ridge=np.zeros((10,10),np.float32);ridge[4,6]=20.
 a=np.array([1.5,1.5,45.,45.,10.]);b=np.array([8.5,3.5,255.,105.,10.]);c=np.array([8.5,7.5,255.,225.,10.])
 intervals=kernel.blocks(a,b,c,ridge);assert intervals and all(not(lo<=0<=hi or lo<=1<=hi) for lo,hi in intervals) and any(lo<=.5<=hi for lo,hi in intervals)
 synthetic_mismatch=0
 for t in np.linspace(0,1,1001):
  blocked=any(lo<=t<=hi for lo,hi in intervals);actual=not bool(scene.kernel.rayclear(ridge,10,10,a,np.ascontiguousarray(b+t*(c-b))))
  synthetic_mismatch+=blocked!=actual
 assert synthetic_mismatch==0
 # Packaged actual trajectory intervals: analytic blocking versus independent ray kernel.
 interval_file=Path(interval_file) if interval_file else ROOT/'问题三_V3_最终方案_精确_通信物理阶段.csv'
 interval_file=interval_file if interval_file.is_absolute() else ROOT/interval_file
 old=pd.read_csv(interval_file);p0=get_points(old,'p0');p1=get_points(old,'p1');checks=0;mismatches=[]
 for j,(b,c) in enumerate(zip(p0,p1)):
  blocks=kernel.blocks(scene.gateway,b,c)
  for t in [0.,.25,.5,.75,1.]:
   want=any(lo<=t<=hi for lo,hi in blocks);actual=not bool(scene.kernel.rayclear(scene.dem,scene.nr,scene.nc,scene.gateway,np.ascontiguousarray(b+t*(c-b))))
   checks+=1
   if want!=actual:mismatches.append(dict(interval=j,t=t,analytic_blocked=want,ray_blocked=actual))
 assert not mismatches,mismatches[:5]
 # Longer changing rays plus independent pure-Python polygon clipping.
 random_checks=0;fallback_cases=0;events_checked=0
 for j in range(90):
  k=int(rng.integers(len(old)));b=p0[k].copy();c=p1[k].copy();c[:2]+=rng.uniform(-20,20,2);c[4]+=rng.uniform(-20,20);a=scene.gateway.copy();a[:2]+=rng.uniform(-40,40,2);a[4]=max(a[4],float(scene.dem[int(a[1]),int(a[0])])+100)
  blocks=kernel.blocks(a,b,c)
  for t in np.linspace(0,1,101):
   want=any(lo<=t<=hi for lo,hi in blocks);actual=not bool(scene.kernel.rayclear(scene.dem,scene.nr,scene.nc,np.ascontiguousarray(a),np.ascontiguousarray(b+t*(c-b))));assert want==actual;(random_checks:=random_checks+1)
  for lo,hi in blocks:
   for edge in (lo,hi):
    for t in [max(0.,edge-1e-7),min(1.,edge+1e-7)]:
     want=any(l<=t<=h for l,h in blocks);actual=not bool(scene.kernel.rayclear(scene.dem,scene.nr,scene.nc,np.ascontiguousarray(a),np.ascontiguousarray(b+t*(c-b))));assert want==actual;events_checked+=1
  if j<20:
   py=python_blocks(scene.dem,a,b,c);assert np.asarray(py).shape==np.asarray(blocks).shape and np.allclose(py,blocks,rtol=0,atol=2e-12);fallback_cases+=1
 result=dict(passed=True,seed=20260924,narrow_ridge_blocked_intervals=intervals,narrow_ridge_dense_ray_checks=1001,narrow_ridge_mismatches=synthetic_mismatch,source_interval_file=str(interval_file),source_interval_count=len(old),actual_interval_ray_checks=checks,actual_interval_mismatches=len(mismatches),longer_random_ray_checks=random_checks,near_transition_ray_checks=events_checked,cpp_python_parameter_polygon_cases=fallback_cases,proof='Visibility transition completeness follows from convex parameter-polygon projection t=u/s; dense checks are independent regression tests, not the continuous proof.',terrain_contact_tolerance_m=HEIGHT_EPS,backend=kernel.backend)
 (ROOT/'q3_v3_communication_QA.json').write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding='utf-8');print(json.dumps(result,ensure_ascii=False,indent=2));return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',default='问题三_冻结推荐方案');ap.add_argument('--output-prefix');ap.add_argument('--backend',default='auto',choices=['auto','python']);ap.add_argument('--self-test',action='store_true');ap.add_argument('--test-intervals');args=ap.parse_args();selftest(args.test_intervals,args.backend) if args.self_test else run(args.prefix,args.output_prefix,args.backend)
