import numpy as np, sys
d=np.load(sys.argv[1]); V=d['V']
rng=np.random.default_rng(0)
best=(0,None)
for i in range(3000):
    p=V[rng.choice(len(V),3,replace=False)]
    n=np.cross(p[1]-p[0],p[2]-p[0]); nn=np.linalg.norm(n)
    if nn<1e-12: continue
    n/=nn; dist=np.abs((V-p[0])@n)
    c=(dist<0.0003).sum()
    if c>best[0]: best=(c,(n,p[0]))
n,p0=best[1]
# refine with SVD on inliers, iterate
for tol in [0.0005,0.0003,0.0002]:
    inl=np.abs((V-p0)@n)<tol
    P=V[inl]; c=P.mean(0); u,s,vt=np.linalg.svd(P-c,full_matrices=False); n=vt[2]; p0=c
if n[1]<0: n=-n
h=(V-p0)@n
inl=np.abs(h)<0.0003
print('normal',n,'inliers',inl.sum(),len(V),'rms mm',np.sqrt((h[inl]**2).mean())*1000)
print('height pct mm', np.percentile(h*1000,[0,1,5,25,50,75,95,99,100]))
# build frame: z=n, x along largest in-plane extent
a=np.array([1,0,0.]); x=a-(a@n)*n; x/=np.linalg.norm(x); y=np.cross(n,x)
R=np.stack([x,y,n]); Vp=(V-p0)@R.T
np.savez(sys.argv[2],R=R,p0=p0,Vp=Vp)
print('in-plane ext mm', np.ptp(Vp[:,:2],0)*1000)
hist,edges=np.histogram(Vp[:,2]*1000,bins=80,range=(-6,36)); 
for hc,e in zip(hist,edges): print(f'{e:6.1f} {hc}')
