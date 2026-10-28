import numpy as np, sys
def load_obj(p):
    v=[];vt=[];fv=[];ft=[]
    with open(p) as f:
        for line in f:
            if line.startswith('v '): v.append(line[2:].split())
            elif line.startswith('vt '): vt.append(line[3:].split()[:2])
            elif line.startswith('f '):
                a=[x.split('/') for x in line[2:].split()]
                fv.append([int(x[0]) for x in a]); ft.append([int(x[1]) if len(x)>1 and x[1] else 0 for x in a])
    return (np.array(v,float), np.array(vt,float), np.array(fv,np.int64)-1, np.array(ft,np.int64)-1)
if __name__=='__main__':
    V,VT,F,FT=load_obj(sys.argv[1])
    np.savez(sys.argv[2],V=V,VT=VT,F=F,FT=FT)
    print(V.shape,VT.shape,F.shape, V.min(0),V.max(0),np.ptp(V,0))
