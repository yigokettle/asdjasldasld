"""认证更新独立绘图入口。沿用用户已选渐变样式，只读取环境变量指定并核验的方案。"""
from pathlib import Path
import sys,json,os
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager,patheffects
from matplotlib.ticker import MaxNLocator
P=Path(__file__).resolve().parent
OUT=P
EX=P/'q1_flat'  # 用户原始绘图示例随数据一同提供
sys.path.insert(0,str(P))
from 问题二_物理模型与输入 import read_inputs
from q1_flat.精确算法 import read_dem,touched_cells
N,B,M,U,BAT=read_inputs()
PREFIX=os.environ.get('Q2_CERTIFIED_PREFIX','全局认证_冻结最终十九架次')
if not PREFIX: raise RuntimeError('请设置 Q2_CERTIFIED_PREFIX 为已经冻结并独立核验的方案前缀')
SOURCE=Path(PREFIX) if Path(PREFIX).is_absolute() else P/PREFIX
F=pd.read_csv(str(SOURCE)+'_架次.csv')
D=pd.read_csv(str(SOURCE)+'_逐箱.csv')
L=pd.read_csv(str(SOURCE)+'_航段.csv')
assert len(D)==80 and D['货箱编号'].nunique()==80
assert set(F['架次编号'])==set(D['架次编号'])==set(L['架次编号'])
D=D.merge(B[['material_type','mass_kg','volume_m3','first_batch']].rename_axis('货箱编号').reset_index(),on='货箱编号',validate='one_to_one')
FONT=P/'q1_flat'/'中文字体_绘图临时.otf';font_manager.fontManager.addfont(str(FONT));CN=font_manager.FontProperties(fname=str(FONT)).get_name()
plt.rcParams.update({'font.family':['STIXGeneral',CN],'font.size':16,'axes.labelsize':17,'xtick.labelsize':13,'ytick.labelsize':13,'axes.titlesize':19,'legend.fontsize':13,'axes.unicode_minus':False,'axes.linewidth':1.4,'figure.facecolor':'white','savefig.facecolor':'white','text.color':'black','axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','pdf.fonttype':42,'ps.fonttype':42})
CM=plt.colormaps['RdYlBu_r'];MC={'A':CM(.08),'B':CM(.77),'C':CM(.98)}
ALGS=['地形ALNS–CP-SAT','普通ALNS–CP-SAT','遗传算法–CP-SAT','蚁群算法–CP-SAT','NSGA-II–CP-SAT','地形ALNS（无重组）']
LABELS=['地形ALNS','普通ALNS','遗传算法','蚁群算法','NSGA-II','地形ALNS对照']
AC={a:CM(v) for a,v in zip(ALGS,[.07,.20,.71,.84,.98,.34])}

def panel(ax,letter,title=''):
    ax.set_title(f'{letter}  {title}',loc='left',fontsize=19,pad=10,color='black')
    ax.tick_params(width=1.25,length=4)
    for sp in ax.spines.values():sp.set_linewidth(1.4)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)
    ax.tick_params(labelsize=13,width=1.2,length=4)

def save(fig,name):
    path=OUT/name;temp=path.with_suffix('.tmp.png')
    fig.savefig(temp,dpi=320,bbox_inches='tight',pad_inches=.10)
    plt.close(fig);os.replace(temp,path);print(name,flush=True)

def gradient_box(ax,patch,lo=.05,hi=.95):
    # 用户渐变配色方式：对一份RdYlBu_r连续取值，再裁切到真实四分位箱体。
    vertices=patch.get_path().vertices
    xx=vertices[:,0];yy=vertices[:,1]
    if np.ptp(xx)<1e-12 or np.ptp(yy)<1e-12:return
    im=ax.imshow(np.linspace(lo,hi,256).reshape(256,1),origin='lower',aspect='auto',
      extent=[xx.min(),xx.max(),yy.min(),yy.max()],cmap=CM,vmin=0,vmax=1,zorder=patch.get_zorder()-.1,alpha=.68)
    im.set_clip_path(patch);patch.set_facecolor('none');patch.set_edgecolor('black');patch.set_linewidth(1.3)

def gradient_patch(ax,patch,lo=.05,hi=.95,orientation='vertical',alpha=1,zorder=None):
    """数据坐标内裁切渐变；不更改柱长/区间，不改变已有坐标范围。"""
    xy=patch.get_path().transformed(patch.get_patch_transform()).vertices
    x0,y0=np.min(xy,axis=0);x1,y1=np.max(xy,axis=0)
    if x1-x0<=1e-14 or y1-y0<=1e-14:return None
    grad=np.linspace(lo,hi,256)
    grad=grad[:,None] if orientation=='vertical' else grad[None,:]
    limits=(ax.get_xlim(),ax.get_ylim());aspect=ax.get_aspect()
    z=patch.get_zorder()-.05 if zorder is None else zorder
    im=ax.imshow(grad,extent=[x0,x1,y0,y1],origin='lower',aspect='auto',
                 cmap=CM,vmin=0,vmax=1,alpha=alpha,zorder=z,interpolation='bilinear')
    im.set_clip_path(patch);patch.set_facecolor('none')
    ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);ax.set_aspect(aspect)
    return im

def gradient_fill(ax,collection,xmin,xmax,ymin,ymax,lo=.05,hi=.95,
                  orientation='vertical',alpha=1,zorder=None):
    from matplotlib.patches import PathPatch
    if xmax<=xmin or ymax<=ymin:return None
    grad=np.linspace(lo,hi,256)
    grad=grad[:,None] if orientation=='vertical' else grad[None,:]
    limits=(ax.get_xlim(),ax.get_ylim());aspect=ax.get_aspect()
    im=ax.imshow(grad,extent=[xmin,xmax,ymin,ymax],origin='lower',aspect='auto',
        cmap=CM,vmin=0,vmax=1,alpha=alpha,
        zorder=collection.get_zorder()-.05 if zorder is None else zorder,
        interpolation='bilinear')
    im.set_clip_path(PathPatch(collection.get_paths()[0],transform=ax.transData))
    collection.set_facecolor('none')
    ax.set_xlim(limits[0]);ax.set_ylim(limits[1]);ax.set_aspect(aspect)
    return im

def gradient_line(ax,x,y,lo=.05,hi=.95,lw=2.5,step=None,alpha=1,zorder=3):
    """沿已有折线着色，post阶梯在原事件位置跳变，不插值数据值。"""
    from matplotlib.collections import LineCollection
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=float)
    if len(x)<2:return None
    if step=='post':
        x=np.repeat(x,2)[1:];y=np.repeat(y,2)[:-1]
    path=np.c_[x,y];pieces=[];positions=[]
    # 对直线作绘图细分，仅改变渲染色阶；数据折点原位保留。
    for i,(a,b) in enumerate(zip(path[:-1],path[1:])):
        if np.allclose(a,b,rtol=0,atol=1e-14):continue
        t=np.linspace(0,1,20);fine=a+(b-a)*t[:,None]
        pieces.extend(np.stack([fine[:-1],fine[1:]],axis=1))
        positions.extend((i+(t[:-1]+t[1:])/2)/(len(path)-1))
    lc=LineCollection(pieces,colors=CM(lo+(hi-lo)*np.asarray(positions)),
                      linewidths=lw,alpha=alpha,zorder=zorder,capstyle='butt')
    ax.add_collection(lc)
    return lc

from matplotlib.patches import Patch as _Patch,Rectangle as _Rectangle
from matplotlib.legend_handler import HandlerBase as _HandlerBase
from matplotlib.legend import Legend as _Legend
class GradientKey(_Patch):
    def __init__(self,label,lo,hi,hatch=None):
        super().__init__(facecolor='none',edgecolor='black',label=label,hatch=hatch)
        self.lo=lo;self.hi=hi
class GradientKeyHandler(_HandlerBase):
    def create_artists(self,legend,handle,xd,yd,w,h,fs,trans):
        artists=[_Rectangle((-xd+j*w/32,-yd),w/32+.02,h,
                    fc=CM(value),ec='none',transform=trans)
                  for j,value in enumerate(np.linspace(handle.lo,handle.hi,32))]
        artists.append(_Rectangle((-xd,-yd),w,h,fc='none',ec='black',lw=.45,
                                  hatch=handle.get_hatch(),transform=trans))
        return artists
_Legend.update_default_handler_map({GradientKey:GradientKeyHandler()})
