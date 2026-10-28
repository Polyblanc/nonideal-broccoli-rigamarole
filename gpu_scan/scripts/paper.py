import numpy as np, sys
hm=np.load(sys.argv[1]); h=hm['hmap']; res=float(hm['res'])
sim=np.load(sys.argv[2]); c=float(sim['c'])
paper=np.abs(h)<0.0008  # near plane
H,W=h.shape
def edges(mask,axis):
    lo=[];hi=[];idx=[]
    for i in range(0,mask.shape[0],10):
        r=np.where(mask[i])[0]
        if len(r)>mask.shape[1]*0.5: lo.append(r.min()); hi.append(r.max()); idx.append(i)
    return np.array(idx),np.array(lo),np.array(hi)
def robust_line(i,v):
    m=np.ones(len(i),bool)
    for _ in range(5):
        p=np.polyfit(i[m],v[m],1); r=v-np.polyval(p,i); s=np.median(np.abs(r[m]))*1.5+0.5; m=np.abs(r)<3*s
    return p,m.mean()
i,lo,hi=edges(paper,0); pl,fl=robust_line(i,lo); ph,fh=robust_line(i,hi)
mid=H/2; wpx=np.polyval(ph,mid)-np.polyval(pl,mid)
i2,lo2,hi2=edges(paper.T,1); pl2,fl2=robust_line(i2,lo2); ph2,fh2=robust_line(i2,hi2)
hpx=np.polyval(ph2,W/2)-np.polyval(pl2,W/2)
print('inlier frac',fl,fh,fl2,fh2,'slopes',pl[0],ph[0],pl2[0],ph2[0])
w=wpx*res*1000; hh=hpx*res*1000
print('paper in scan-mm: %.2f x %.2f  ratio %.4f (A4 %.4f)'%(w,hh,hh/w,297/210))
print('paper via charuco scale: %.1f x %.1f mm'%(w*c/1000,hh*c/1000))
print('A4 scale factors: %.4f %.4f ; charuco factor %.4f'%(210/w,297/hh,c/1000))
print('implied checker size if paper is A4: %.3f / %.3f mm'%(20*(210/w)/(c/1000),20*(297/hh)/(c/1000)))
