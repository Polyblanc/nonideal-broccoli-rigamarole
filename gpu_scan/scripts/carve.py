# Visual-hull / space carving from orthographic depth views. Board frame, mm.
import numpy as np, sys, trimesh, fast_simplification, json
from scipy import ndimage as ndi
from skimage import measure
S=sys.argv[1]; res=float(sys.argv[2]) if len(sys.argv)>2 else 0.25
X0,X1,Y0,Y1,Z0,Z1=28,206,-120,-42,0,36
xs=np.arange(X0,X1,res)+res/2; ys=np.arange(Y0,Y1,res)+res/2; zs=np.arange(Z0,Z1,res)+res/2
def load(name):
    d=np.load(f'{S}/work/view_{name}_d.npz'); h=d['hmap'].astype(np.float32); mn=d['mn']; r=float(d['res'])
    hit=~np.isnan(h)
    hit=ndi.binary_fill_holes(ndi.binary_closing(hit,np.ones((3,3)),iterations=2))
    # fill holes inside the silhouette with the nearest valid depth
    idx=ndi.distance_transform_edt(np.isnan(h),return_distances=False,return_indices=True)
    h=h[tuple(idx)]; h[~hit]=np.nan
    h=ndi.median_filter(np.nan_to_num(h,nan=-1e9),3); h[~hit]=np.nan
    return h,mn,r
def sample(h,mn,r,u,v):   # u,v arrays (broadcastable) in the view's (u,v) coords
    i=np.floor((u-mn[0])/r).astype(int); j=np.floor((v-mn[1])/r).astype(int)
    ok=(i>=0)&(j>=0)&(i<h.shape[1])&(j<h.shape[0])
    out=np.full(np.broadcast(i,j).shape,np.nan,np.float32); out[ok]=h[j[ok] if j.shape==ok.shape else np.broadcast_to(j,ok.shape)[ok], i[ok] if i.shape==ok.shape else np.broadcast_to(i,ok.shape)[ok]]
    return out
X,Y=np.meshgrid(xs,ys,indexing='ij')                      # (nx,ny)
Xz,Zx=np.meshgrid(xs,zs,indexing='ij')                    # (nx,nz)
Yz,Zy=np.meshgrid(ys,zs,indexing='ij')                    # (ny,nz)
top=sample(*load('top'),X,Y)                              # z_top(x,y)
fr=-sample(*load('front'),Xz,Zx)                          # y_min(x,z)
bk=sample(*load('back'),Xz,Zx)                            # y_max(x,z)
lf=-sample(*load('left'),Yz,Zy)                           # x_min(y,z)
rt=sample(*load('right'),Yz,Zy)                           # x_max(y,z)
tol=res
def carve(use):
    occ=np.ones((len(xs),len(ys),len(zs)),bool)
    if 'top' in use: occ&=(zs[None,None,:]<=np.nan_to_num(top,nan=-1e9)[:,:,None]+tol)
    if 'front' in use: occ&=(ys[None,:,None]>=np.nan_to_num(fr,nan=1e9)[:,None,:]-tol)
    if 'back' in use: occ&=(ys[None,:,None]<=np.nan_to_num(bk,nan=-1e9)[:,None,:]+tol)
    if 'left' in use: occ&=(xs[:,None,None]>=np.nan_to_num(lf,nan=1e9)[None,:,:]-tol)
    if 'right' in use: occ&=(xs[:,None,None]<=np.nan_to_num(rt,nan=-1e9)[None,:,:]+tol)
    lab,n=ndi.label(occ); 
    if n>1: occ=lab==(1+np.argmax(ndi.sum(occ,lab,range(1,n+1))))
    return occ
out={}
for tag,use in [('3view',['top','front','left']),('5view',['top','front','back','left','right'])]:
    occ=carve(use); vol=occ.sum()*res**3
    pad=np.pad(occ,1).astype(np.float32); pad=ndi.gaussian_filter(pad,0.6)
    v,f,_,_=measure.marching_cubes(pad,0.5,spacing=(res,res,res))
    v+=np.array([X0,Y0,Z0])-res/2
    m=trimesh.Trimesh(v,f[:,::-1],process=True); print('  raw faces',len(m.faces),'watertight',m.is_watertight)
    for target in (250000,400000,None):
        if target is None: mm=m; break
        vv,ff=fast_simplification.simplify(m.vertices.astype(np.float32),m.faces.astype(np.int32),target_count=target)
        mm=trimesh.Trimesh(vv,ff,process=True); mm.merge_vertices()
        if mm.is_watertight: break
    m=mm; trimesh.repair.fix_normals(m)
    m.export(f'{S}/out/gpu_hull_{tag}.stl'); m.export(f'{S}/out/gpu_hull_{tag}.3mf')
    out[tag]=dict(voxel_volume_mm3=round(float(vol)),mesh_volume_mm3=round(float(m.volume)),watertight=bool(m.is_watertight),faces=len(m.faces),bounds=m.bounds.round(2).tolist())
    np.save(f'{S}/work/occ_{tag}.npy',np.packbits(occ))
    print(tag,out[tag])
sw=trimesh.load(f'{S}/out/gpu_shrinkwrap.stl'); out['top_only_shrinkwrap_volume_mm3']=round(float(sw.volume)); print('shrinkwrap volume',sw.volume)
json.dump(out,open(f'{S}/out/hull_stats.json','w'),indent=1)
