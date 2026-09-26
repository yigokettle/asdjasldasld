"""问题三V4：联合资源、逐箱交付、同域实际方案权衡（仅真实解）。
用法：python 问题三_V4_统计绘图.py --prefix 问题三_V3_推荐方案 \
  --comparison 问题三_V3_方案比较.csv --output-prefix 问题三_V3
不要求零延误、不固定方案数量/坐标范围、不插值虚构前沿。
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from 问题二_物理模型与输入 import read_inputs

P=Path(__file__).resolve().parent
CM=plt.colormaps['RdYlBu_r']
FP=P/'q1_flat'/'中文字体_绘图临时.otf'
font_manager.fontManager.addfont(str(FP));CN=font_manager.FontProperties(fname=str(FP)).get_name()
plt.rcParams.update({'font.family':['STIXGeneral',CN],'font.size':18.5,'axes.titlesize':22,
 'axes.labelsize':19,'xtick.labelsize':17,'ytick.labelsize':17,'legend.fontsize':17,
 'axes.linewidth':1.5,'text.color':'black','axes.labelcolor':'black','xtick.color':'black',
 'ytick.color':'black','figure.facecolor':'white','savefig.facecolor':'white','axes.unicode_minus':False})

def path(v):
    q=Path(v);return q if q.is_absolute() else P/q

def panel(ax,label,title):
    ax.set_title(f'{label}  {title}',loc='left',pad=10,fontweight='bold')
    ax.spines[['top','right']].set_visible(False);ax.tick_params(width=1.3,length=4.5)

def save(fig,name,dpi):
    out=Path(str(name)+'.png');tmp=out.with_name(out.stem+'.tmp.png')
    fig.savefig(tmp,dpi=dpi,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)
    from PIL import Image
    with Image.open(tmp) as im:im.load()
    tmp.replace(out)
    print(str(out),flush=True);return out

def gradbar(ax,x,y,w,h,lo=.07,hi=.94,z=2,vertical=False):
    if w<=0 or h<=0:return
    r=Rectangle((x,y),w,h,fc='none',ec='black',lw=.65,zorder=z+.2);ax.add_patch(r)
    gr=np.linspace(lo,hi,128)
    im=ax.imshow(gr[:,None] if vertical else gr[None,:],origin='lower',
       extent=[x,x+w,y,y+h],aspect='auto',cmap=CM,vmin=0,vmax=1,zorder=z)
    im.set_clip_path(r);return r

def gradline(ax,x,y,lo=.07,hi=.94,lw=3,step=False):
    x=np.asarray(x,float);y=np.asarray(y,float)
    if step:x=np.repeat(x,2)[1:];y=np.repeat(y,2)[:-1]
    if len(x)<2:return
    seg=np.stack([np.c_[x[:-1],y[:-1]],np.c_[x[1:],y[1:]]],axis=1)
    ax.add_collection(LineCollection(seg,colors=CM(np.linspace(lo,hi,len(seg))),lw=lw,zorder=3))

def bounds(v,frac=.15,nonnegative=False):
    v=np.asarray(v,float);lo,hi=float(v.min()),float(v.max());span=hi-lo
    pad=max(span*frac,abs(hi)*.015,0.01)
    return (max(0,lo-pad) if nonnegative else lo-pad,hi+pad)

def concurrency(frame,start,end):
    ev={}
    for a,b in frame[[start,end]].itertuples(index=False,name=None):
        assert b>=a
        ev[a]=ev.get(a,0)+1;ev[b]=ev.get(b,0)-1
    cur=0;rows=[]
    for t,delta in sorted(ev.items()):cur+=delta;rows.append((t,cur))
    assert cur==0
    return pd.DataFrame(rows,columns=['时刻秒','占用数量'])

def load(prefix):
    f=pd.read_csv(str(prefix)+'_运输_架次.csv');d=pd.read_csv(str(prefix)+'_运输_逐箱.csv')
    legs=pd.read_csv(str(prefix)+'_运输_航段.csv');relay=pd.read_csv(str(prefix)+'_中继架次.csv')
    nodes,boxes,models,drones,batteries=read_inputs()
    d=d.merge(boxes.reset_index(),left_on='货箱编号',right_on='box_id',validate='one_to_one')
    assert len(d)==80 and d['货箱编号'].nunique()==80
    hard=np.isfinite(d.hard_deadline_s)
    assert (d.loc[hard,'送达秒']<=d.loc[hard,'hard_deadline_s']+1e-7).all()
    d['实际交付分钟']=d['送达秒']/60
    d['期望余量分钟']=(d['期望送达秒']-d['送达秒'])/60
    d['硬时限余量分钟']=np.where(hard,(d.hard_deadline_s-d['送达秒'])/60,np.nan)
    resource=[];events=[]
    specs=[]
    for g,capu,capb in [('A',4,6),('B',2,4),('C',2,4)]:
        sub=f[f['机型'].eq(g)]
        specs.extend([(f'{g}型运输机',sub,'开始秒','返回秒',capu),(f'{g}型运输电池',sub,'开始秒','电池充满秒',capb)])
    specs.extend([('中继无人机',relay,'开始秒','无人机可用秒',2),('中继能源组件',relay,'开始秒','组件充满秒',6)])
    for label,fr,start,end,cap in specs:
        curve=concurrency(fr,start,end);peak=int(curve['占用数量'].max()) if len(curve) else 0
        assert peak<=cap
        events.append(curve.assign(资源=label,库存=cap))
        resource.append(dict(资源=label,峰值=peak,库存=cap))
    # 检查实体编号分配，不能只用总库存曲线替代。
    for fr,entity,end in [(f,'无人机','返回秒'),(f,'电池','电池充满秒'),
                           (relay,'中继无人机','无人机可用秒'),(relay,'能源组件','组件充满秒')]:
        for key,rows in fr.groupby(entity):
            rows=rows.sort_values('开始秒')
            assert np.all(rows['开始秒'].to_numpy()[1:]>=rows[end].to_numpy()[:-1]-1e-9),(entity,key)
    return f,d,legs,relay,models,drones,batteries,pd.DataFrame(resource),pd.concat(events,ignore_index=True)

STYLES={'准备':(.06,.22),'飞行':(.24,.46),'交付':(.49,.65),'建链':(.49,.65),
        '中继服务':(.56,.74),'任务':(.13,.58),'充电':(.78,.97)}

def phasebar(ax,y,a,b,stage):
    if b<=a:return
    if stage=='周转':ax.add_patch(Rectangle((a/60,y-.34),(b-a)/60,.68,fc='white',ec='black',hatch='///',lw=.65,zorder=2))
    else:gradbar(ax,a/60,y-.34,(b-a)/60,.68,*STYLES[stage])

def figure_resources(f,d,legs,relay,models,drones,batteries,out,dpi):
    entities=[sorted(drones),sorted(batteries),['R01','R02'],[f'H{i:02d}' for i in range(1,7)]]
    assert list(map(len,entities))==[8,14,2,6]
    # 横向时间轴四组面板，行高按实体数分配，避免2架中继机被拉成巨大条带。
    fig,axs=plt.subplots(4,1,figsize=(19.6,13.6),sharex=True,
              gridspec_kw={'height_ratios':[8.25,14.25,2.8,6.25]})
    fig.subplots_adjust(left=.092,right=.987,top=.93,bottom=.06,hspace=.17)
    all_segments=[]
    for panelno,(ax,ids) in enumerate(zip(axs,entities)):
        panel(ax,'abcd'[panelno],['运输机','运输电池','中继机','能源组件'][panelno])
        ax.set_yticks(range(len(ids)),ids);ax.set_ylim(len(ids)-.4,-.7)
        ax.tick_params(axis='y',labelsize=16)
        ax.grid(axis='x',color='#d8d8d8',lw=.6,alpha=.6,zorder=0)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        for ii in range(len(ids)):ax.axhline(ii,color='#dddddd',lw=.45,zorder=0)
    for _,r in f.iterrows():
        id=r['架次编号'];g=r['机型'];model=models.loc[g];seq=r['访问顺序'].split('→')
        boxes=r['货箱编号'].split(',');t=float(r['开始秒']);prep=float(model.fixed_prep_s+model.load_per_box_s*len(boxes))
        stages=[(t,t+prep,'准备')];t+=prep
        for _,leg in legs[legs['架次编号'].eq(id)].iterrows():
            ft=float(leg['飞行秒']);stages.append((t,t+ft,'飞行'));t+=ft
            if leg['终点']!='O01':
                n=len(d[(d['架次编号']==id)&(d['服务区']==leg['终点'])])
                dur=float(model.handoff_base_s+model.handoff_per_box_s*n)
                stages.append((t,t+dur,'交付'));t+=dur
        assert abs(t-r['返回秒'])<1e-7
        y=entities[0].index(r['无人机'])
        for a,b,stage in stages:
            phasebar(axs[0],y,a,b,stage);all_segments.append(dict(资源类别='运输机',实体=r['无人机'],架次=id,阶段=stage,开始秒=a,结束秒=b))
        axs[0].text((r['开始秒']+r['返回秒'])/120,y,id,ha='center',va='center',fontsize=13.8,fontweight='bold')
        y=entities[1].index(r['电池'])
        for a,b,stage in [(r['开始秒'],r['返回秒'],'任务'),(r['返回秒'],r['电池充满秒'],'充电')]:
            phasebar(axs[1],y,a,b,stage);all_segments.append(dict(资源类别='运输电池',实体=r['电池'],架次=id,阶段=stage,开始秒=a,结束秒=b))
        axs[1].text((r['开始秒']+r['返回秒'])/120,y,id,ha='center',va='center',fontsize=13.8,fontweight='bold')
    for _,r in relay.iterrows():
        a=r['开始秒'];ready=r['建链完成秒'];se=r['服务结束秒'];ret=r['返回秒'];id=r['中继架次']
        stages=[(a,a+180,'准备'),(a+180,ready-30,'飞行'),(ready-30,ready,'建链'),
                (ready,se,'中继服务'),(se,ret,'飞行'),(ret,r['无人机可用秒'],'周转')]
        y=entities[2].index(r['中继无人机'])
        for aa,bb,stage in stages:
            phasebar(axs[2],y,aa,bb,stage);all_segments.append(dict(资源类别='中继机',实体=r['中继无人机'],架次=id,阶段=stage,开始秒=aa,结束秒=bb))
        axs[2].text((ready+se)/120,y,id,ha='center',va='center',fontsize=13.8,fontweight='bold')
        y=entities[3].index(r['能源组件'])
        for aa,bb,stage in [(a,ret,'任务'),(ret,r['组件充满秒'],'充电')]:
            phasebar(axs[3],y,aa,bb,stage);all_segments.append(dict(资源类别='能源组件',实体=r['能源组件'],架次=id,阶段=stage,开始秒=aa,结束秒=bb))
        axs[3].text((a+ret)/120,y,id,ha='center',va='center',fontsize=13.8,fontweight='bold')
    handles=[Rectangle((0,0),1,1,fc=CM(.14),ec='black',label='准备'),Rectangle((0,0),1,1,fc=CM(.34),ec='black',label='飞行 / 任务占用'),
             Rectangle((0,0),1,1,fc=CM(.62),ec='black',label='交付 / 建链 / 中继服务'),Rectangle((0,0),1,1,fc=CM(.88),ec='black',label='充电'),
             Rectangle((0,0),1,1,fc='white',ec='black',hatch='///',label='地面周转')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.55,.985),ncol=5,frameon=False,fontsize=15.7)
    hi=max(float(f['电池充满秒'].max()),float(relay['组件充满秒'].max()),float(relay['无人机可用秒'].max()))/60
    for ax in axs:
        ax.set_xlim(0,hi*1.02)
        for t in [60,120,180]:
            if t<hi:ax.axvline(t,color='black',ls=':',lw=.8,alpha=.65,zorder=1)
    axs[-1].set_xlabel('任务开始后时间（分钟）')
    save(fig,str(out)+'_图02_运输与中继联合资源时序',dpi)
    return pd.DataFrame(all_segments)

def figure_deliveries(d,res,out,dpi):
    fig,axs=plt.subplots(2,3,figsize=(21.2,11.6));fig.subplots_adjust(left=.068,right=.988,bottom=.087,top=.94,wspace=.34,hspace=.36)
    ax=axs[0,0];panel(ax,'a','逐箱交付')
    dd=d.sort_values(['期望送达秒','送达秒','货箱编号']).reset_index(drop=True);x=np.arange(1,len(dd)+1)
    ax.step(x,dd['期望送达秒']/60,where='mid',color='black',ls='--',lw=2.2,label='期望时限')
    hard=np.isfinite(dd.hard_deadline_s);late=dd['送达秒']>dd['期望送达秒']+1e-8
    ax.scatter(x[hard],dd.loc[hard,'hard_deadline_s']/60,s=70,marker='D',fc='white',ec='black',lw=.9,zorder=3,label='硬时限')
    ax.scatter(x,dd['实际交付分钟'],s=86,c=dd['实际交付分钟'],cmap=CM,ec='black',lw=.8,zorder=4,label='实际交付')
    if late.any():ax.scatter(x[late],dd.loc[late,'实际交付分钟'],s=105,fc='none',ec=CM(.98),lw=1.8,zorder=5,label='普通物资延误')
    ax.set_xlim(-1,len(dd)+2);ax.set_ylim(0,max(dd['期望送达秒'].max()/60,dd['实际交付分钟'].max())*1.08)
    ax.set_xticks([1,20,40,60,80]);ax.set_xlabel('货箱序号');ax.set_ylabel('时间（分钟）')
    ax.legend(frameon=False,loc='upper left',fontsize=16,handlelength=1.5)
    ax=axs[0,1];panel(ax,'b','服务区交付')
    sites=sorted(d['服务区'].unique());mx=float(d['实际交付分钟'].max())
    rows=[]
    for i,s in enumerate(sites):
        a=d.loc[d['服务区']==s,'实际交付分钟'].to_numpy();lo,md,hi=a.min(),np.median(a),a.max()
        gradline(ax,np.linspace(lo,hi,24),np.full(24,i),lw=5.2)
        ax.scatter([lo,hi],[i,i],s=50,fc='white',ec='black',lw=.7,zorder=4)
        ax.scatter(md,i,s=110,c=[CM(.07+.88*i/max(1,len(sites)-1))],ec='black',lw=.6,zorder=5)
        rows.append(dict(服务区=s,箱数=len(a),首箱分钟=lo,中位数分钟=md,末箱分钟=hi))
    ax.set_yticks(range(len(sites)),sites);ax.set_ylim(len(sites)-.4,-.6);ax.set_xlim(0,mx*1.07)
    ax.set_xlabel('实际交付时间（分钟）');ax.xaxis.set_major_locator(MaxNLocator(5))
    ax=axs[0,2];panel(ax,'c','物资进度')
    materials=['医疗物资','饮用水','应急食品','生活卫生用品'];spans=[(.03,.18),(.22,.39),(.65,.81),(.85,.99)];dash=['-','--','-.',':'];handles=[]
    for m,(lo,hi),ls in zip(materials,spans,dash):
        a=np.sort(d.loc[d.material_type==m,'实际交付分钟'].to_numpy());assert len(a)>0
        tx=np.r_[0,a,mx*1.06];ty=np.r_[0,np.arange(1,len(a)+1)/len(a)*100,100]
        gradline(ax,tx,ty,lo,hi,4.2,True);ax.step(tx,ty,where='post',c=CM((lo+hi)/2),lw=1.5,ls=ls)
        handles.append(Line2D([],[],c=CM((lo+hi)/2),lw=3,ls=ls,label=f'{m}（{len(a)}箱）'))
    ax.set_xlim(0,mx*1.06);ax.set_ylim(0,106);ax.set_yticks([0,25,50,75,100]);ax.set_xlabel('时间（分钟）');ax.set_ylabel('累计交付比例（%）')
    ax.legend(handles=handles,loc='lower right',frameon=False,fontsize=15.5)
    ax=axs[1,0];panel(ax,'d','期望余量')
    vals=[d.loc[d.material_type==m,'期望余量分钟'].to_numpy() for m in materials]
    bp=ax.boxplot(vals,positions=range(4),widths=.52,patch_artist=True,showfliers=False,
         medianprops={'color':'black','lw':2},boxprops={'facecolor':'none','edgecolor':'black','lw':1.2},
         whiskerprops={'color':'black','lw':1.2},capprops={'color':'black','lw':1.2})
    vmin=float(d['期望余量分钟'].min());vmax=float(d['期望余量分钟'].max())
    for i,a in enumerate(vals):
        q1,q3=np.quantile(a,[.25,.75]);gradbar(ax,i-.26,q1,.52,q3-q1,.07,.96,z=1,vertical=True)
        rng=np.random.default_rng(20260923+i)
        ax.scatter(i+rng.uniform(-.16,.16,len(a)),a,s=58,c=a,cmap=CM,vmin=vmin,vmax=vmax,ec='black',lw=.35,zorder=4)
    ax.axhline(0,c='black',ls=':',lw=1.2);ax.set_xlim(-.6,3.6);ax.set_ylim(*bounds(np.r_[0,d['期望余量分钟']],.12))
    ax.set_xticks(range(4),['医疗物资','饮用水','应急食品','生活卫生\n用品']);ax.set_ylabel('期望时限 − 实际交付（分钟）')
    ax=axs[1,1];panel(ax,'e','硬时限余量')
    hd=d[np.isfinite(d.hard_deadline_s)];h=hd.groupby('服务区')['硬时限余量分钟'].min().sort_index()
    for i,(s,v) in enumerate(h.items()):
        gradline(ax,np.linspace(0,v,32),np.full(32,i),lw=4.7)
        ax.scatter(v,i,s=108,c=[CM(.07+.88*i/max(1,len(h)-1))],ec='black',lw=.6,zorder=4)
    ax.axvline(0,c='black',ls=':',lw=1);ax.set_yticks(range(len(h)),h.index);ax.set_ylim(len(h)-.4,-.6)
    ax.set_xlim(-max(1,h.max()*.02),h.max()*1.15);ax.set_xlabel('各服务区最小余量（分钟）')
    ax=axs[1,2];panel(ax,'f','资源峰值')
    for i,r in res.iterrows():
        ax.plot([0,r['库存']],[i,i],c='#c6c6c6',lw=3,zorder=1)
        gradline(ax,np.linspace(0,r['峰值'],24),np.full(24,i),lw=5.5)
        ax.scatter(r['库存'],i,s=135,marker='|',c='black',lw=2,zorder=4)
        ax.scatter(r['峰值'],i,s=118,c=[CM(.93)],ec='black',lw=.7,zorder=5)
        ax.text(res['库存'].max()+.45,i,f"{r['峰值']} / {r['库存']}",va='center',fontsize=16)
    ax.set_yticks(range(len(res)),res['资源']);ax.set_ylim(len(res)-.4,-.6);ax.set_xlim(0,res['库存'].max()+1.55)
    ax.set_xticks([0,2,4,6]);ax.set_xlabel('同时占用数量');ax.text(.99,1.01,'峰值 / 库存',transform=ax.transAxes,ha='right',fontsize=15)
    save(fig,str(out)+'_图07_逐箱交付与资源核验',dpi)
    return pd.DataFrame(rows)

def load_comparison(file):
    c=pd.read_csv(file)
    aliases={'完成时间秒':'联合完成秒','运输数':'运输架次','中继数':'中继架次','总能耗':'总能耗kWh','前缀':'源前缀','加权逾期':'加权延误'}
    c=c.rename(columns={k:v for k,v in aliases.items() if k in c and v not in c})
    assert all(k in c for k in ['源前缀','运输架次','中继架次','联合完成秒','运输能耗kWh','中继能耗kWh','总能耗kWh','推荐'])
    c['推荐']=c['推荐'].astype(str).str.lower().isin(['true','1','是']);assert c['推荐'].sum()==1
    if '加权延误' not in c:c['加权延误']=np.nan
    for i,r in c.iterrows():
        src=path(r['源前缀']);f=pd.read_csv(str(src)+'_运输_架次.csv');rl=pd.read_csv(str(src)+'_中继架次.csv');dd=pd.read_csv(str(src)+'_运输_逐箱.csv')
        assert len(f)==r['运输架次'] and len(rl)==r['中继架次']
        assert abs(f['能耗kWh'].sum()-r['运输能耗kWh'])<1e-7 and abs(rl['能耗kWh'].sum()-r['中继能耗kWh'])<1e-7
        assert abs(max(f['返回秒'].max(),rl['返回秒'].max())-r['联合完成秒'])<1e-6
        actual=float(dd['归一化加权延误'].sum())
        if np.isfinite(r['加权延误']):assert abs(actual-r['加权延误'])<1e-6
        c.loc[i,'加权延误']=actual
    c=c.sort_values(['联合完成秒','总能耗kWh']).reset_index(drop=True)
    c['联合完成分钟']=c['联合完成秒']/60;c['图中名称']=[f'方案{i+1}' for i in range(len(c))]
    c.loc[c['推荐'],'图中名称']='推荐方案'
    return c

def representative_rows(c):
    """完整档案都画散点；条形面板只抽取可解释且真实存在的代表。"""
    selections=[]
    def add(i,label):
        if i not in [j for j,_ in selections]:selections.append((i,label))
    zero=c[c['加权延误']<1e-10]
    add(int(c.index[c['推荐']][0]),'推荐方案')
    if len(zero):add(int(zero.sort_values(['联合完成秒','总能耗kWh']).index[0]),'零延误快送')
    add(int(c.sort_values(['总能耗kWh','联合完成秒']).index[0]),'低能耗方案')
    tmp=c.assign(_count=c['运输架次']+c['中继架次'])
    add(int(tmp.sort_values(['_count','总能耗kWh','联合完成秒']).index[0]),'较少架次')
    late=c[c['加权延误']>1e-10]
    if len(late):add(int(late.sort_values(['总能耗kWh','联合完成秒']).index[0]),'允许延误')
    if '来源配置' in c:
        legacy=c[c['来源配置'].astype(str).str.contains('旧完整|旧推荐')]
        if len(legacy):add(int(legacy.index[0]),'原推荐方案')
    r=c.loc[[i for i,_ in selections]].copy().reset_index(drop=True)
    r['图中名称']=[label for _,label in selections]
    return r,selections

def figure_tradeoffs(c,out,dpi):
    representative,selections=representative_rows(c)
    representative.to_csv(str(out)+'_条形代表方案.csv',index=False,encoding='utf-8-sig')
    r=representative.copy();n=len(r)
    short={'推荐方案':'推荐方案','零延误快送':'零延误快送','低能耗方案':'低能耗方案','较少架次':'较少架次','允许延误':'允许延误','原推荐方案':'原方案'}
    labels=[short.get(v,v) for v in r['图中名称']]
    cols=CM(np.array([.96,.10,.30,.55,.73,.86])[:n]);styles=['-','--','-.',':',(0,(5,2)),(0,(3,1,1,1))]
    fig,axs=plt.subplots(2,3,figsize=(21.5,12.0))
    fig.subplots_adjust(left=.067,right=.988,bottom=.084,top=.93,wspace=.33,hspace=.37)
    # 所有点均为完整实际方案，二维重合不作坐标抖动。
    ax=axs[0,0];panel(ax,'a','时间与能耗')
    cnorm=matplotlib.colors.Normalize(c['联合完成分钟'].min(),c['联合完成分钟'].max())
    for i,row in c.iterrows():
        sel=row['推荐'];late=row['加权延误']>1e-10
        ax.scatter(row['联合完成分钟'],row['总能耗kWh'],s=480 if sel else 150,
            marker='*' if sel else ('^' if late else 'o'),
            c=[CM(.97) if sel else CM(.08+.84*cnorm(row['联合完成分钟']))],ec='black',lw=.95,zorder=6 if sel else 4)
    rec=c[c['推荐']].iloc[0]
    ax.annotate('推荐',(rec['联合完成分钟'],rec['总能耗kWh']),xytext=(-10,13),textcoords='offset points',ha='right',fontsize=18,fontweight='bold')
    ax.set_xlim(*bounds(c['联合完成分钟'],.18));ax.set_ylim(*bounds(c['总能耗kWh'],.18))
    ax.set_xlabel('联合完成时间（分钟）');ax.set_ylabel('总能耗（千瓦时）')
    ax.xaxis.set_major_locator(MaxNLocator(5));ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.legend(handles=[Line2D([],[],ls='',marker='o',mfc=CM(.25),mec='black',label='零延误'),
              Line2D([],[],ls='',marker='^',mfc=CM(.7),mec='black',label='允许延误')],
              loc='upper right',frameon=False,fontsize=16,handletextpad=.3)
    ax=axs[0,1];panel(ax,'b','运输与中继架次')
    combinations=c.groupby(['运输架次','中继架次']).size()
    cmin=float((c['运输架次']+c['中继架次']).min());cmax=float((c['运输架次']+c['中继架次']).max())
    for (nt,nr),count in combinations.items():
        if nt==rec['运输架次'] and nr==rec['中继架次'] and count==1:
            continue  # 唯一推荐配置只画星号，避免重复圆形符号。
        color=CM(.08+.84*((nt+nr-cmin)/max(1,cmax-cmin)))
        ax.scatter(nt,nr,s=130+90*int(count),c=[color],ec='black',lw=1,zorder=3)
        if count>1:ax.text(nt,nr,str(int(count)),ha='center',va='center',fontsize=17,fontweight='bold',zorder=4)
    ax.scatter(rec['运输架次'],rec['中继架次'],s=500,marker='*',c=[CM(.97)],ec='black',lw=1,zorder=6)
    ax.set_xlim(c['运输架次'].min()-.7,c['运输架次'].max()+.7);ax.set_ylim(c['中继架次'].min()-.6,c['中继架次'].max()+.6)
    ax.set_xticks(np.arange(c['运输架次'].min(),c['运输架次'].max()+1));ax.set_yticks(np.arange(c['中继架次'].min(),c['中继架次'].max()+1))
    ax.set_xlabel('运输架次数');ax.set_ylabel('中继架次数')
    ax.text(.99,.04,'圆内数字：同配置方案数',transform=ax.transAxes,ha='right',fontsize=15.5)
    ax=axs[0,2];panel(ax,'c','目标取舍')
    fields=['加权延误','联合完成秒','总能耗kWh'];allobj=np.c_[c[fields],c['运输架次']+c['中继架次']]
    repobj=np.c_[r[fields],r['运输架次']+r['中继架次']];lo=allobj.min(axis=0);span=allobj.max(axis=0)-lo
    norm=np.divide(repobj-lo,span,out=np.zeros_like(repobj,dtype=float),where=span>1e-12)
    for i in range(n):
        isrec=bool(r.iloc[i]['推荐'])
        ax.plot(range(4),norm[i],c=cols[i],ls=styles[i],lw=3 if isrec else 2.2,
                marker='*' if isrec else 'o',ms=13 if isrec else 6,mec='black',mew=.6,zorder=5 if isrec else 3,label=labels[i])
    ax.set_xticks(range(4),['延误','时间','能耗','总架次']);ax.set_xlim(-.15,3.15);ax.set_ylim(-.07,1.42)
    ax.set_yticks([0,.25,.5,.75,1]);ax.set_ylabel('档案内归一化值')
    ax.legend(loc='upper center',bbox_to_anchor=(.50,1.015),ncol=2,fontsize=15.5,frameon=False,handlelength=2,columnspacing=.8)
    ax=axs[1,0];panel(ax,'d','能耗构成')
    for i,row in r.iterrows():
        te=float(row['运输能耗kWh']);re=float(row['中继能耗kWh'])
        gradbar(ax,0,i-.30,te,.60,.07,.57);gradbar(ax,te,i-.30,re,.60,.74,.98)
        ax.text(te/2,i,f'{te:.2f}',ha='center',va='center',fontsize=17)
        ax.text(te+re+1.1,i,f'{re:.2f}',va='center',fontsize=17)
    ax.set_yticks(range(n),labels);ax.set_ylim(n-.35,-1.02);ax.set_xlim(0,r['总能耗kWh'].max()*1.13);ax.set_xlabel('能耗（千瓦时）')
    handles=[Rectangle((0,0),1,1,fc=CM(.32),ec='black',label='运输'),Rectangle((0,0),1,1,fc=CM(.88),ec='black',label='中继')]
    ax.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=16,handlelength=1,columnspacing=1)
    ax=axs[1,1];panel(ax,'e','架次构成')
    for i,row in r.iterrows():
        nt=int(row['运输架次']);nr=int(row['中继架次']);gradbar(ax,0,i-.30,nt,.60,.07,.57);gradbar(ax,nt,i-.30,nr,.60,.74,.98)
        ax.text(nt/2,i,str(nt),ha='center',va='center',fontsize=18);ax.text(nt+nr/2,i,str(nr),ha='center',va='center',fontsize=18)
        if row['推荐']:ax.scatter(nt+nr+1,i,s=230,marker='*',c=[CM(.97)],ec='black',lw=.7,zorder=4)
    ax.set_yticks(range(n),labels);ax.set_ylim(n-.35,-1.02);ax.set_xlim(0,float((r['运输架次']+r['中继架次']).max())+2.5);ax.set_xlabel('架次数');ax.xaxis.set_major_locator(MaxNLocator(5,integer=True))
    ax.legend(handles=handles,loc='upper center',ncol=2,frameon=False,fontsize=16,handlelength=1,columnspacing=1)
    ax=axs[1,2];panel(ax,'f','逐箱送达余量')
    values=[];marginrows=[]
    for i,row in r.iterrows():
        deliveries=pd.read_csv(str(path(row['源前缀']))+'_运输_逐箱.csv')
        val=(deliveries['期望送达秒']-deliveries['送达秒']).to_numpy()/60;values.append(val)
        marginrows.extend([{'源前缀':row['源前缀'],'图中名称':labels[i],'货箱编号':b,'余量分钟':float(v)} for b,v in zip(deliveries['货箱编号'],val)])
    bp=ax.boxplot(values,positions=range(n),widths=.50,patch_artist=True,showfliers=False,
        medianprops={'color':'black','lw':2.2},boxprops={'facecolor':'none','edgecolor':'black','lw':1.3},
        whiskerprops={'color':'black','lw':1.2},capprops={'color':'black','lw':1.2})
    for i,v in enumerate(values):
        q1,q3=np.quantile(v,[.25,.75]);gradbar(ax,i-.25,q1,.50,q3-q1,.08,.94,z=1,vertical=True)
        jitter=np.random.default_rng(20260923+i).uniform(-.18,.18,len(v))
        ax.scatter(i+jitter,v,s=17,c=[cols[i]],ec='black',lw=.25,alpha=.70,zorder=3)
    ax.axhline(0,c='black',ls=':',lw=1.5);ax.set_xlim(-.6,n-.4);ax.set_ylim(*bounds(np.concatenate(values),.10))
    ax.set_xticks(range(n),['推荐','快送','低能耗','少架次','可延误','原方案'][:n]);ax.tick_params(axis='x',labelsize=15.5)
    ax.set_ylabel('期望时限 − 交付（分钟）');ax.yaxis.set_major_locator(MaxNLocator(5))
    pd.DataFrame(marginrows).to_csv(str(out)+'_代表方案逐箱余量.csv',index=False,encoding='utf-8-sig')
    save(fig,str(out)+'_图06_实际方案权衡',dpi)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prefix',required=True);parser.add_argument('--comparison')
    parser.add_argument('--output-prefix',default='问题三_V4');parser.add_argument('--dpi',type=int,default=320)
    a=parser.parse_args();prefix=path(a.prefix);out=path(a.output_prefix)
    f,d,legs,relay,models,drones,batteries,res,events=load(prefix)
    seg=figure_resources(f,d,legs,relay,models,drones,batteries,out,a.dpi)
    sites=figure_deliveries(d,res,out,a.dpi)
    comp=load_comparison(path(a.comparison)) if a.comparison else None
    if comp is not None:
        figure_tradeoffs(comp,out,a.dpi);comp.to_csv(str(out)+'_方案权衡绘图数据.csv',index=False,encoding='utf-8-sig')
    for df,suffix in [(seg,'资源阶段'),(d,'逐箱交付绘图数据'),(sites,'服务区交付统计'),(res,'资源峰值'),(events,'资源并发事件')]:
        df.to_csv(str(out)+'_'+suffix+'.csv',index=False,encoding='utf-8-sig')
    qa={'PASS':True,'货箱数':len(d),'硬时限箱数':int(np.isfinite(d.hard_deadline_s).sum()),'普通物资延误箱数':int((d['期望余量分钟']<-1e-8).sum()),
        '加权延误':float(d['归一化加权延误'].sum()),'比较方案数':len(comp) if comp is not None else 0,
        '资源实体数':{'运输无人机':8,'运输电池':14,'中继机':2,'能源组件':6},'资源峰值':res.to_dict('records'),
        '口径':'普通期望余量可为负；硬时限必须满足。阶段/充电/周转来自实际记录。比较点是完整实际调度，不构造插值帕累托曲面；星号表示选定推荐。',
        '输入前缀':str(prefix),'输入哈希':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(str(prefix)+'_运输_架次.csv'),Path(str(prefix)+'_运输_逐箱.csv'),Path(str(prefix)+'_中继架次.csv')]}}
    qa['脚本sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    qa['输出PNG哈希']={q.name:hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(str(out)+'_图02_运输与中继联合资源时序.png'),Path(str(out)+'_图06_实际方案权衡.png'),Path(str(out)+'_图07_逐箱交付与资源核验.png')] if q.exists()}
    Path(str(out)+'_统计绘图核验.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in qa.items() if k not in ['输入哈希','资源峰值','口径']},ensure_ascii=False),flush=True)
if __name__=='__main__':main()
