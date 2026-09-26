import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

root=Path(r'D:/zxq/数模26/D题'); out=root/'问题三_图件'; out.mkdir(exist_ok=True)
res=root/'问题三结果'
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimSun']; plt.rcParams['axes.unicode_minus']=False
st=pd.read_csv(res/'问题三_时间主方案_精确_通信物理阶段.csv')
box=pd.read_csv(res/'问题三_时间主方案_运输_逐箱.csv')
mission=pd.read_csv(res/'问题三_时间主方案_运输_架次.csv')

fig,ax=plt.subplots(figsize=(13,7),dpi=220)
ids=list(st['route'].drop_duplicates()); ymap={v:i for i,v in enumerate(ids)}
for _,r in st.iterrows():
    y=ymap[r['route']]; color='#2a9d8f' if bool(r['direct_ok']) else '#e76f51'
    ax.plot([r['start_s'],r['end_s']],[y,y],lw=7,color=color,solid_capstyle='butt')
ax.set_yticks(range(len(ids))); ax.set_yticklabels(ids); ax.set_xlabel('时间 / s'); ax.set_title('问题三原创图A：连续通信状态带'); ax.grid(axis='x',alpha=.25)
fig.tight_layout(); fig.savefig(out/'问题三_原创图A_通信状态带.png'); plt.close(fig)

pivot=box.pivot_table(index='架次编号',columns='服务区',values='送达秒',aggfunc='min')
fig,ax=plt.subplots(figsize=(13,8),dpi=220); im=ax.imshow(pivot.fillna(np.nan),aspect='auto',cmap=LinearSegmentedColormap.from_list('delivery',['#edf8fb','#8bd3dd','#006d77']))
ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index); ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns,rotation=45,ha='right'); ax.set_title('问题三原创图B：架次—服务区逐箱送达热力图'); ax.set_xlabel('服务区'); ax.set_ylabel('运输架次'); fig.colorbar(im,ax=ax,label='送达时刻 / s'); fig.tight_layout(); fig.savefig(out/'问题三_原创图B_送达热力图.png'); plt.close(fig)

mission['开始秒']=pd.to_numeric(mission['开始秒']); mission['返回秒']=pd.to_numeric(mission['返回秒'])
t=np.arange(0,int(mission['返回秒'].max())+60,60); load=[((mission['开始秒']<=q)&(mission['返回秒']>q)).sum() for q in t]
fig,ax=plt.subplots(figsize=(12,6),dpi=220); ax.step(t,load,where='post',color='#6a4c93',lw=2.5); ax.fill_between(t,load,step='post',alpha=.18,color='#6a4c93'); ax.set_title('问题三原创图C：运输无人机同时占用数量'); ax.set_xlabel('时间 / s'); ax.set_ylabel('占用架次'); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig(out/'问题三_原创图C_资源负载.png'); plt.close(fig)
print('done')
