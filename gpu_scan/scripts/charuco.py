import numpy as np, cv2, sys
img=cv2.imread(sys.argv[1],0)
d=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_100)
for flip in [False,True]:
  im=img[::-1] if flip else img
  for size in [(12,8),(8,12)]:
    for legacy in [False,True]:
      b=cv2.aruco.CharucoBoard(size,20.0,15.0,d); b.setLegacyPattern(legacy)
      det=cv2.aruco.CharucoDetector(b)
      cc,ci,mc,mi=det.detectBoard(im)
      n=0 if ci is None else len(ci); nm=0 if mi is None else len(mi)
      print(flip,size,legacy,'markers',nm,'corners',n)
