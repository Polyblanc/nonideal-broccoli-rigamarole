import numpy as np, cv2, sys
img=cv2.imread(sys.argv[1],0)[::-1].copy(); H=img.shape[0]
hm=np.load(sys.argv[2]); res=float(hm['res']); mn=hm['mn']
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
p=cv2.aruco.DetectorParameters(); p.adaptiveThreshWinSizeMax=153; p.adaptiveThreshWinSizeStep=6; p.cornerRefinementMethod=cv2.aruco.CORNER_REFINE_SUBPIX; p.maxErroneousBitsInBorderRate=0.5; p.errorCorrectionRate=0.8
mc,mi,rej=cv2.aruco.ArucoDetector(d,p).detectMarkers(img)
print('raw markers',len(mi),sorted(mi.ravel().tolist()),'rejected',len(rej))
b=cv2.aruco.CharucoBoard((12,8),20.0,15.0,d); b.setLegacyPattern(True)
cp=cv2.aruco.CharucoParameters(); cp.minMarkers=1
cc,ci,_,_=cv2.aruco.CharucoDetector(b,cp,p).detectBoard(img)
obj=b.getChessboardCorners()[ci.ravel()][:,:2]  # mm
pix=cc.reshape(-1,2)
# pixel (flipped img) -> scan plane units: x=mn0+px*res ; y = mn1 + (H-1-py)*res  (+0.5 px center)
sx=mn[0]+(pix[:,0])*res; sy=mn[1]+(H-1-pix[:,1]+1)*res
S_=np.stack([sx,sy],1)
def umeyama(A,B,refl=False):
    ma,mb=A.mean(0),B.mean(0); A0,B0=A-ma,B-mb
    U,s,Vt=np.linalg.svd(B0.T@A0); D=np.eye(2)
    if (np.linalg.det(U@Vt)<0)!=refl: D[1,1]=-1
    R=U@D@Vt; c=(s*np.diag(D)).sum()/(A0**2).sum(); t=mb-c*R@ma; return c,R,t
best=None
for refl in [False,True]:
    c,R,t=umeyama(S_,obj,refl); e=np.linalg.norm((c*(R@S_.T)).T+t-obj,axis=1)
    print('refl',refl,'rms',np.sqrt((e**2).mean()))
    if best is None or e.mean()<best[0]: best=(e.mean(),c,R,t)
_,c,R,t=best
pred=(c*(R@S_.T)).T+t; err=np.linalg.norm(pred-obj,axis=1)
print('n corners',len(obj),'scale mm per scan unit',c,'=> scan unit is',c/1000,'m; det R',np.linalg.det(R))
print('residual mm: rms %.3f max %.3f'%(np.sqrt((err**2).mean()),err.max()))
# also affine check for anisotropy
A=np.c_[S_,np.ones(len(S_))]; M,_,_,_=np.linalg.lstsq(A,obj,rcond=None); sv=np.linalg.svd(M[:2])[1]; print('affine singular values',sv, 'anisotropy %.4f%%'%((sv[0]/sv[1]-1)*100))
np.savez(sys.argv[3],c=c,R=R,t=t,err=err)
