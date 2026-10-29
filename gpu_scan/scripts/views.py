# Orthographic texture + depth views of the part from each side (board-frame mm). Crops away the board.
import numpy as np, sys, subprocess, os
S=sys.argv[1]; d=np.load(S+'/work/mesh_mm.npz'); V=d['V']; F=d['F']
X0,X1,Y0,Y1=28,206,-120,-42
inb=(V[:,0]>X0)&(V[:,0]<X1)&(V[:,1]>Y0)&(V[:,1]<Y1)&(V[:,2]>0.3)
keep=inb[F].all(1)
np.savez(S+'/work/part_mesh.npz',V=V,F=F[keep],VT=d['VT'],FT=d['FT'][keep])
views={'top':(0,1,2,1),'front':(0,2,1,-1),'back':(0,2,1,1),'left':(1,2,0,-1),'right':(1,2,0,1)}
for name,(u,v,w,sgn) in views.items():
    Vp=np.c_[V[:,u],V[:,v],sgn*V[:,w]]
    # anchor the raster frame to fixed bounds so depth maps share a grid
    lo={0:X0,1:Y0,2:0.0}; hi={0:X1,1:Y1,2:36.0}
    Vp=np.vstack([Vp,[[lo[u],lo[v],-1e6],[hi[u],hi[v],-1e6]]])
    np.savez(S+f'/work/view_{name}_frame.npz',Vp=Vp)
    np.savez(S+f'/work/view_{name}_mesh.npz',V=Vp,F=F[keep],VT=d['VT'],FT=d['FT'][keep])
print('triangles kept',keep.sum())
