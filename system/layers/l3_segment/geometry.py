"""Geometry helpers for Layer 3 (from Pipeline v9)."""
import numpy as np

EDGE, FRAC = 8, 0.45

def inter(a,b):
    ix=max(0,min(a[2],b[2])-max(a[0],b[0])); iy=max(0,min(a[3],b[3])-max(a[1],b[1]))
    return ix*iy

def iou(a,b):
    i=inter(a,b); u=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-i
    return i/u if u>0 else 0.0

def ctr(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)

def border_filter(bx,W,H):
    if not bx: return []
    med=np.median([(x2-x1)*(y2-y1) for x1,y1,x2,y2 in bx]); keep=[]
    for b in bx:
        x1,y1,x2,y2=b; a=(x2-x1)*(y2-y1)
        if (x1<=EDGE or y1<=EDGE or x2>=W-EDGE or y2>=H-EDGE) and a<FRAC*med: continue
        keep.append(b)
    return keep

def deoverlap(A):
    A=[list(b) for b in A]
    for i in range(len(A)):
        for j in range(len(A)):
            if i==j or inter(A[i],A[j])<=0: continue
            a,b=A[i],A[j]
            if a[0]<b[0]:
                oxw=min(a[2],b[2])-max(a[0],b[0]); oyh=min(a[3],b[3])-max(a[1],b[1])
                if oxw<oyh:
                    m=(a[2]+b[0])/2; a[2]=min(a[2],m); b[0]=max(b[0],m)
    return A

def page_reading_order(bx):
    if not bx: return []
    idx=sorted(range(len(bx)),key=lambda i:bx[i][1])
    band=0.12*max(b[3]-b[1] for b in bx)+0.03*max(b[3] for b in bx); rows=[]
    for i in idx:
        y=bx[i][1]
        if rows and y-rows[-1][0]<=band: rows[-1][1].append(i)
        else: rows.append([y,[i]])
    o=[]
    for _,r in rows: o+=sorted(r,key=lambda i:bx[i][0])
    return o

def assign_by_containment(articles,regs,thr=0.15):
    out={i:[] for i in range(len(articles))}
    for r in regs:
        ra=(r[2]-r[0])*(r[3]-r[1]); best=-1; bs=0.0
        for i,art in enumerate(articles):
            s=inter(art,r)/ra if ra>0 else 0
            if s>bs: bs,best=s,i
        if best>=0 and bs>thr: out[best].append(r)
        else:
            cx,cy=ctr(r); d=1e18; bi=0
            for i,art in enumerate(articles):
                ax,ay=ctr(art); dd=(cx-ax)**2+(cy-ay)**2
                if dd<d: d,bi=dd,i
            out[bi].append(r)
    return out

def merge_overlapping(regs,thr=0.30):
    regs=[list(r) for r in regs]; changed=True
    while changed:
        changed=False; out=[]; used=[False]*len(regs)
        for i in range(len(regs)):
            if used[i]: continue
            cur=regs[i][:]
            for j in range(i+1,len(regs)):
                if used[j]: continue
                if iou(cur,regs[j])>thr:
                    cur=[min(cur[0],regs[j][0]),min(cur[1],regs[j][1]),
                         max(cur[2],regs[j][2]),max(cur[3],regs[j][3])]
                    used[j]=True; changed=True
            out.append(cur); used[i]=True
        regs=out
    return regs

def order_columns(regs,wtol=0.5):
    if not regs: return []
    regs=sorted(merge_overlapping(regs),key=lambda r:r[0]); cols=[]
    for r in regs:
        placed=False
        for col in cols:
            lo=min(c[0] for c in col); hi=max(c[2] for c in col)
            if min(r[2],hi)-max(r[0],lo) > wtol*min(r[2]-r[0],hi-lo):
                col.append(r); placed=True; break
        if not placed: cols.append([r])
    cols.sort(key=lambda c:min(x[0] for x in c)); out=[]
    for col in cols: col.sort(key=lambda r:r[1]); out+=col
    return out

