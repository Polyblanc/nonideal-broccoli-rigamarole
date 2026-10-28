# Cover residual component envelope (above PCB, outside heatsink/caps) with stacked cuboid slabs.
import numpy as np, json, sys
from scipy import ndimage as ndi
P=json.load(open(sys.argv[1])); hm=np.load(sys.argv[2]); h=hm['hmap']; mn=hm['mn']; res=float(hm['res'])
A=P['auto_blocks']; g=A['grid']; step=A['level_step']
pc=P['pcb']; zt=pc['z1']
x0,x1,y0,y1=pc['main']
# sample heightmap onto grid cells (median), rows = y
nx=int(round((x1-x0)/g)); ny=int(round((y1-y0)/g))
Hs=ndi.median_filter(np.nan_to_num(h,nan=0),size=int(A['smooth_mm']/res)|1)
G=np.zeros((ny,nx))
for j in range(ny):
    for i in range(nx):
        cx0=int((x0+i*g-mn[0])/res); cy0=int((y0+j*g-mn[1])/res); k=int(g/res)
        G[j,i]=np.median(Hs[cy0:cy0+k,cx0:cx0+k])
cxs=x0+(np.arange(nx)+0.5)*g; cys=y0+(np.arange(ny)+0.5)*g
CX,CY=np.meshgrid(cxs,cys)
hs=P['heatsink']; r=hs['rect']; nt=hs['notch']
mg=1.2
inHS=(CX>r[0]-mg)&(CX<r[1]+mg)&(CY>r[2]-mg)&(CY<r[3]+mg)&~((CX<nt[1]-mg)&(CY>nt[2]+mg))
excl=inHS.copy()
for cx,cy in P['capacitors']['xy']: excl|=(CX-cx)**2+(CY-cy)**2<(P['capacitors']['r']+0.3)**2
for cut in pc['cuts']: excl|=(CX>cut[0])&(CX<cut[1])&(CY>cut[2])&(CY<cut[3])
G=np.where(excl,zt,G); G=np.minimum(G,15.0)
mode=sys.argv[4] if len(sys.argv)>4 else 'slab'
MINA=int(sys.argv[5]) if len(sys.argv)>5 else 4
def largest_rect(M):
    ny,nx=M.shape; hgt=np.zeros(nx,int); best=(0,0,0,0,0)
    for j in range(ny):
        hgt=np.where(M[j],hgt+1,0); st=[]
        for i in range(nx+1):
            cur=hgt[i] if i<nx else 0; start=i
            while st and st[-1][1]>=cur:
                s0,hh=st.pop(); a=hh*(i-s0)
                if a>best[4]: best=(j-hh+1,j+1,s0,i,a)
                start=s0
            st.append((start,cur))
    return best
boxes=[]
if mode=='tower':
    Q=np.round((G-zt)/step).astype(int); Q[Q<1]=0
    Q=ndi.median_filter(Q,size=3)
    used=np.zeros(Q.shape,bool)
    for j in range(ny):
        i=0
        while i<nx:
            q=Q[j,i]
            if q>0 and not used[j,i]:
                i2=i
                while i2+1<nx and Q[j,i2+1]==q and not used[j,i2+1]: i2+=1
                j2=j
                while j2+1<ny and (Q[j2+1,i:i2+1]==q).all() and not used[j2+1,i:i2+1].any(): j2+=1
                used[j:j2+1,i:i2+1]=True
                boxes.append([x0+i*g,x0+(i2+1)*g,y0+j*g,y0+(j2+1)*g,zt,zt+q*step])
                i=i2+1
            else: i+=1
    json.dump(boxes,open(sys.argv[3],'w')); print('tower blocks',len(boxes)); sys.exit()
L=zt+step
while L<=G.max()+step/2:
    m=G>=L-step/2+A['min_excess']*(L==zt+step)
    m=ndi.binary_opening(m,np.ones((2,2)))
    rem=m.copy()
    while rem.sum()>=MINA:
        j0,j1,i0,i1,area=largest_rect(rem)
        if area<MINA: break
        rem[j0:j1,i0:i1]=False
        boxes.append([x0+i0*g,x0+i1*g,y0+j0*g,y0+j1*g,L-step,L])
    L+=step
json.dump(boxes,open(sys.argv[3],'w'))
print('blocks',len(boxes),'levels up to',max(b[5] for b in boxes))
