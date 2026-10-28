# transform mesh into board frame, millimetres. scale corrected by measured checker size.
import numpy as np, sys
d=np.load(sys.argv[1]); fr=np.load(sys.argv[2]); sim=np.load(sys.argv[3]); checker=float(sys.argv[4])
c=float(sim['c'])*checker/20.0; R=sim['R']; t=sim['t']*checker/20.0
Vp=fr['Vp']
xy=(c*(R@Vp[:,:2].T)).T+t
z=Vp[:,2]*c
# R has det -1 (reflection) relative to scan frame; flip z?  board frame (x,y) right-handed with z up out of paper:
# scan frame had right-handed (x,y,n); mapping via reflection in xy => need z -> keep up, frame becomes left-handed -> negate y instead
V=np.c_[xy[:,0],-xy[:,1],z]  # charuco y points down the image; flip so (x,y,z-up) is right-handed (no mirroring)
np.savez(sys.argv[5],V=V,F=d['F'],VT=d['VT'],FT=d['FT'],k=c)
print('k mm/unit',c,'bounds',V.min(0),V.max(0))
