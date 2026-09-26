from pathlib import Path
import os
os.environ['MPLBACKEND']='Agg'
import numpy as np, pandas as pd, rasterio
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LightSource, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
ROOT=Path(r'D:/zxq/数模26/D题'); out=ROOT/'问题三_图件'; out.mkdir(exist_ok=True)
dem=ROOT/'数据/镇龙乡地理空间数据/镇龙乡及周边地理数据/数字高程模型数据（DEM）/镇龙乡及周边30米DEM.tif'
sites=pd.read_csv(Path(r'D:/zxq/数模26/github_export_q2_q4/结果/site_groups.csv')); sites=sites[sites.scenario_id.eq('R2_不设单站组')].drop_duplicates('site_id')
o_lon,o_lat=109.225,23.005
plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimSun']; plt.rcParams['axes.unicode_minus']=False
cmap=LinearSegmentedColormap.from_list('q4terrain',['#d9edf5','#b9d9e8','#e8edcf','#f1d49b','#c98755'],N=256)
with rasterio.open(dem) as ds:
 z=ds.read(1).astype(float); z[z<=-9990]=np.nan
 rows=np.linspace(0,z.shape[0]-1,180).astype(int); cols=np.linspace(0,z.shape[1]-1,220).astype(int); zz=z[np.ix_(rows,cols)]; zz=np.nan_to_num(zz,nan=np.nanmedian(zz))
 X,Y=np.meshgrid(np.linspace(109.15,109.30,zz.shape[1]),np.linspace(23.00,23.095,zz.shape[0]))
fig=plt.figure(figsize=(16,8),dpi=220); ax=fig.add_subplot(121,projection='3d'); ax.plot_surface(X,Y,zz,cmap=cmap,rstride=2,cstride=2,linewidth=0,antialiased=True,shade=True,alpha=.96)
for _,r in sites.iterrows():
 col='#d8892f' if r.group_id=='G1' else '#3d6f9e'; ax.plot([o_lon,r.longitude],[o_lat,r.latitude],[120,r.elevation_m+80],color=col,lw=1.4,alpha=.8); ax.scatter(r.longitude,r.latitude,r.elevation_m+80,c=col,s=42,edgecolors='white',linewidth=.6); ax.text(r.longitude,r.latitude,r.elevation_m+100,r.site_id,fontsize=7)
ax.scatter([o_lon],[o_lat],[120],marker='*',s=180,c='#ffd166',edgecolors='#222',label='O01 调度中心'); ax.legend(handles=[Line2D([],[],color='#d8892f',lw=2,label='G1 分组连接'),Line2D([],[],color='#3d6f9e',lw=2,label='G2 分组连接'),Line2D([],[],marker='*',color='w',markerfacecolor='#ffd166',markeredgecolor='#222',label='O01 调度中心')],loc='upper left',fontsize=8); ax.view_init(elev=32,azim=-62); ax.set_xlabel('经度'); ax.set_ylabel('纬度'); ax.set_zlabel('海拔 / m'); ax.set_title('(a) 三维 DEM 与分组航线'); 
ax2=fig.add_subplot(122); rgb=LightSource(azdeg=315,altdeg=38).shade(zz,cmap=cmap,vert_exag=.65); ax2.imshow(rgb,extent=[X.min(),X.max(),Y.min(),Y.max()],origin='lower',aspect='auto'); levels=np.linspace(np.nanpercentile(zz,10),np.nanpercentile(zz,90),12); ax2.contour(X,Y,zz,levels=levels,colors='#6e8da3',linewidths=.55,alpha=.55)
for _,r in sites.iterrows():
 col='#d8892f' if r.group_id=='G1' else '#3d6f9e'; ax2.plot([o_lon,r.longitude],[o_lat,r.latitude],color=col,lw=1.8,alpha=.85); ax2.scatter(r.longitude,r.latitude,c=col,s=48,edgecolors='white',linewidth=.7); ax2.text(r.longitude+.001,r.latitude+.001,r.site_id,fontsize=8,color='black',bbox=dict(fc='white',alpha=.55,ec='none',pad=.5))
ax2.scatter(o_lon,o_lat,marker='*',s=210,c='#ffd166',edgecolors='#222',zorder=5); ax2.set_xlabel('经度'); ax2.set_ylabel('纬度'); ax2.set_title('(b) 阴影地形、服务区与分组连接'); ax2.grid(alpha=.18)
fig.colorbar(ScalarMappable(norm=plt.Normalize(np.nanpercentile(zz,2),np.nanpercentile(zz,98)),cmap=cmap),ax=[ax,ax2],shrink=.65,pad=.04,label='DEM 海拔 / m'); fig.suptitle('问题四原创图：R2 独立分组的地形与航线关系',fontsize=17,fontweight='bold'); fig.tight_layout(); fig.savefig(out/'问题四_原创_三维地形与分组航线.png',bbox_inches='tight'); plt.close(fig); print(out/'问题四_原创_三维地形与分组航线.png')






