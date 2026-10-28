# Watertight, de-noised 2.5D "shrink-wrap" of the part from the scan height field (board removed, mm units).
import numpy as np, sys, trimesh, fast_simplification
from scipy import ndimage as ndi
hm=np.load(sys.argv[1]); h=hm['hmap']; mn=hm['mn']; res=float(hm['res']); out=sys.argv[2]; target=int(sys.argv[3])
X0,X1,Y0,Y1=28,206,-120,-42           # part window (mm)
c0=int((X0-mn[0])/res); c1=int((X1-mn[0])/res); r0=int((Y0-mn[1])/res); r1=int((Y1-mn[1])/res)
H=np.nan_to_num(h[r0:r1,c0:c1],nan=0.0)
H=ndi.median_filter(H,size=int(1.0/res)|1)          # kill spikes / film specks
H=ndi.gaussian_filter(H,0.3/res)                    # light smoothing (0.3 mm)
# downsample to 0.2 mm
f=2; H=H[:H.shape[0]//f*f,:H.shape[1]//f*f].reshape(H.shape[0]//f,f,H.shape[1]//f,f).mean((1,3)); r=res*f
M=H>0.8
lab,n=ndi.label(M); M=lab==(1+np.argmax(ndi.sum(M,lab,range(1,n+1))))
M=ndi.binary_closing(M,np.ones((5,5))); M=ndi.binary_opening(M,np.ones((5,5))); M=ndi.binary_fill_holes(M)
H=np.where(M,np.maximum(H,0.8),0)
ny,nx=M.shape
# cells = quads between pixel centres; a cell exists if all 4 corners are in M
Q=M[:-1,:-1]&M[1:,:-1]&M[:-1,1:]&M[1:,1:]
# remove corner-only contacts (non-manifold) by iterated opening on Q
Q=ndi.binary_opening(Q,np.ones((2,2)))
vid=-np.ones((ny,nx),int); used=np.zeros((ny,nx),bool)
jj,ii=np.where(Q)
for dj,di in [(0,0),(1,0),(0,1),(1,1)]: used[jj+dj,ii+di]=True
vj,vi=np.where(used); nv=len(vj); vid[vj,vi]=np.arange(nv)
xs=mn[0]+(c0+0.5)*res+vi*r; ys=mn[1]+(r0+0.5)*res+vj*r
top=np.c_[xs,ys,H[vj,vi]]; bot=np.c_[xs,ys,np.zeros(nv)]
a=vid[jj,ii]; b=vid[jj,ii+1]; c=vid[jj+1,ii+1]; d=vid[jj+1,ii]
Ft=np.r_[np.c_[a,b,c],np.c_[a,c,d]]
Fb=Ft[:,::-1]+nv
# boundary edges of the top surface -> vertical walls
E=np.r_[Ft[:,[0,1]],Ft[:,[1,2]],Ft[:,[2,0]]]
key=np.sort(E,1); u,inv,cnt=np.unique(key,axis=0,return_inverse=True,return_counts=True)
bd=E[cnt[inv.ravel()]==1]
Fw=np.r_[np.c_[bd[:,1],bd[:,0],bd[:,0]+nv],np.c_[bd[:,1],bd[:,0]+nv,bd[:,1]+nv]]
mesh=trimesh.Trimesh(np.r_[top,bot],np.r_[Ft,Fb,Fw],process=True)
print('raw faces',len(mesh.faces),'watertight',mesh.is_watertight)
v,fc=fast_simplification.simplify(mesh.vertices.astype(np.float32),mesh.faces.astype(np.int32),target_count=target)
ms=trimesh.Trimesh(v,fc,process=True); ms.merge_vertices(); trimesh.repair.fix_normals(ms)
print('decimated faces',len(ms.faces),'watertight',ms.is_watertight,'volume %.0f'%ms.volume,'bounds',ms.bounds.round(2).tolist())
ms.export(out+'.stl'); ms.export(out+'.3mf'); ms.export(out+'.ply')
