# Orthographic top-down render of texture (z-buffer max height) in plane frame
import numpy as np, sys
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
d=np.load(sys.argv[1]); fr=np.load(sys.argv[2]); res=float(sys.argv[4])  # m per px
Vp=fr['Vp']; VT=d['VT']; F=d['F']; FT=d['FT']
tex=np.asarray(Image.open(sys.argv[3]).convert('L'),dtype=np.float32); th,tw=tex.shape
mn=Vp[:,:2].min(0); W,H=(np.ptp(Vp[:,:2],0)/res).astype(int)+2
img=np.zeros((H,W),np.float32); zb=np.full((H,W),-1e9,np.float32); hmap=np.full((H,W),np.nan,np.float32)
P=(Vp[:,:2]-mn)/res
for s in range(0,len(F),20000):
    f=F[s:s+20000]; ft=FT[s:s+20000]
    a,b,c=P[f[:,0]],P[f[:,1]],P[f[:,2]]
    lo=np.floor(np.minimum(np.minimum(a,b),c)).astype(int); hi=np.ceil(np.maximum(np.maximum(a,b),c)).astype(int)
    span=(hi-lo).max(1); 
    for k in np.unique(span):
      allidx=np.where(span==k)[0]
      per=max(1,int(2e7//((k+1)**2)))
      for q in range(0,len(allidx),per):
        idx=allidx[q:q+per]
        g=np.arange(k+1); gx,gy=np.meshgrid(g,g); gx=gx.ravel(); gy=gy.ravel()
        px=lo[idx,0,None]+gx; py=lo[idx,1,None]+gy
        A,B,C=a[idx],b[idx],c[idx]
        den=(B[:,1]-C[:,1])*(A[:,0]-C[:,0])+(C[:,0]-B[:,0])*(A[:,1]-C[:,1])
        ok=np.abs(den)>1e-12; den=np.where(ok,den,1)
        qx=px+0.5; qy=py+0.5
        w0=((B[:,1,None]-C[:,1,None])*(qx-C[:,0,None])+(C[:,0,None]-B[:,0,None])*(qy-C[:,1,None]))/den[:,None]
        w1=((C[:,1,None]-A[:,1,None])*(qx-C[:,0,None])+(A[:,0,None]-C[:,0,None])*(qy-C[:,1,None]))/den[:,None]
        w2=1-w0-w1
        m=(w0>=-1e-6)&(w1>=-1e-6)&(w2>=-1e-6)&ok[:,None]&(px>=0)&(py>=0)&(px<W)&(py<H)

        z=w0*Vp[f[idx,0],2,None]+w1*Vp[f[idx,1],2,None]+w2*Vp[f[idx,2],2,None]
        uv=w0[...,None]*VT[ft[idx,0]][:,None]+w1[...,None]*VT[ft[idx,1]][:,None]+w2[...,None]*VT[ft[idx,2]][:,None]
        px,py,z,uv=px[m],py[m],z[m],uv[m]
        order=np.argsort(z); px,py,z,uv=px[order],py[order],z[order],uv[order]
        tu=np.clip((uv[:,0]*tw).astype(int),0,tw-1); tv=np.clip(((1-uv[:,1])*th).astype(int),0,th-1)
        val=tex[tv,tu]
        upd=z>zb[py,px]
        px,py,z,val=px[upd],py[upd],z[upd],val[upd]
        # later (higher z) writes win due to sort
        zb[py,px]=z; img[py,px]=val; hmap[py,px]=z
Image.fromarray(np.clip(img,0,255).astype(np.uint8)).save(sys.argv[5])
np.savez(sys.argv[6],hmap=hmap,mn=mn,res=res)
print(W,H)
