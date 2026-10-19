#!/usr/bin/env python3
"""
jcope_leadtime.py — how forecast skill decays with LEAD TIME (1, 3, 5, 7, 10 days ahead).
Features are fixed (the ocean state at day t); only the target horizon changes
(phytoplankton class at t+LEAD). Reads the JCOPE files once, builds one target column per
lead, trains one calibrated model per lead, and reports skill vs lead + a figure.

    python src/jcope_leadtime.py
Reads t/phy/din off HD-PGF-A (once); caches JCOPE Data/leadtime_table.npz.
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, brier_score_loss, average_precision_score

NLAT, NLON, KSURF, K50 = 356, 362, 9, 5
HALFWIN, EDGES = 5, [-0.75, 0.75]; LAGS = (7, 14, 28); N_CELLS = 8000   # sample ocean cells; read each file ONCE
# STORE_LEADS: targets cached in ONE file-read (all daily leads 1-10).
# LEADS: which to actually evaluate — edit freely and re-run; it loads the cache (seconds),
# so you can do lead 7 first, then add 1/10, or the full curve, with no re-read.
STORE_LEADS = list(range(1, 11)); LEADS = [1, 3, 5, 7, 10]   # full sweep from cache (uncalibrated, fast)
MAXLEAD = max(STORE_LEADS); TARGET_ROWS = 1_200_000  # match main forecast so lead-7 reproduces headline
CACHE = "JCOPE Data/leadtime_table.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
SEASN = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found.")

def build_cache():
    DATA = find_data()
    tf = sorted(glob.glob(f"{DATA}/t/t.*.npy")); pf = sorted(glob.glob(f"{DATA}/phy/phy.*.npy"))
    df = sorted(glob.glob(f"{DATA}/din/din.*.npy"))
    dates = [dt.datetime.strptime(os.path.basename(f).split(".")[1], "%Y%m%d") for f in tf]
    D = len(dates); doy = np.array([min(d.timetuple().tm_yday,365) for d in dates])
    year = np.array([d.year for d in dates]); sea = np.array([SEASN[d.month] for d in dates])
    sinD, cosD = np.sin(2*np.pi*doy/365), np.cos(2*np.pi*doy/365)
    def anom(X):
        clim = np.full((366,)+X.shape[1:], np.nan, np.float32)
        for dd in range(1,366):
            w = np.abs(((doy-dd+182)%365)-182) <= HALFWIN
            clim[dd] = np.nanmean(X[w], axis=0)
        return X - clim[np.clip(doy,1,365)]
    def lag(A,L): o=np.full_like(A,np.nan); o[L:]=A[:-L]; return o
    rng=np.random.RandomState(0)
    # sample OCEAN cells (land is NaN in the .npy) and read EACH FILE ONCE (3 passes total)
    ref=np.load(pf[D//2],mmap_mode="r")[KSURF].reshape(-1)
    ocean=np.where(np.isfinite(ref))[0]
    N=min(N_CELLS,ocean.size); cells=np.sort(rng.choice(ocean,N,replace=False))
    log(f"{D} days; sampling {N} of {ocean.size} ocean cells; reading each file once")
    def readlv(files,levels):
        outs=[np.empty((D,N),np.float32) for _ in levels]
        for i,f in enumerate(files):
            a=np.load(f,mmap_mode="r")
            for j,k in enumerate(levels): outs[j][i]=a[k].reshape(-1)[cells]
        return outs
    tS,t50=readlv(tf,[KSURF,K50]); log("  temp read")
    dS,d50=readlv(df,[KSURF,K50]); log("  din read")
    (pS,)=readlv(pf,[KSURF]);      log("  phy read")
    aTS,aDS,aD50,aStr,aPhy=anom(tS),anom(dS),anom(d50),anom(tS-t50),anom(pS)
    feat={"tS":aTS,"dS":aDS,"d50":aD50,"strat":aStr}
    for L in LAGS: feat[f"tS_l{L}"]=lag(aTS,L); feat[f"dS_l{L}"]=lag(aDS,L)
    spatial=[f for f in FEATS if f not in ("sinD","cosD")]
    futmax=np.full_like(aPhy,np.nan); futmax[:-MAXLEAD]=aPhy[MAXLEAD:]   # validity uses the largest lead
    fin=np.isfinite(aPhy)&np.isfinite(futmax)
    for f in spatial: fin=fin&np.isfinite(feat[f])
    idx=np.where(fin.reshape(-1))[0]
    take=rng.choice(idx,min(TARGET_ROWS,idx.size),replace=False); day=take//N
    T={f:feat[f].reshape(-1)[take] for f in spatial}
    T["sinD"]=sinD[day]; T["cosD"]=cosD[day]
    T["phyNow"]=aPhy.reshape(-1)[take]; T["yr"]=year[day]; T["se"]=sea[day]
    for L in STORE_LEADS:
        fut=np.full_like(aPhy,np.nan); fut[:-L]=aPhy[L:]; T[f"L{L}"]=fut.reshape(-1)[take]
    T={k:np.asarray(v).astype(np.float32) for k,v in T.items()}
    np.savez_compressed(CACHE,**T); log(f"cached {take.size:,} rows -> {CACHE}")
    return T

T = dict(np.load(CACHE)) if os.path.exists(CACHE) else build_cache()
missing=[L for L in LEADS if f"L{L}" not in T]
if missing: raise SystemExit(f"cache lacks leads {missing}; delete {CACHE} and re-run (rebuilds with STORE_LEADS).")
zc=lambda a:a/np.nanstd(a)
X=np.column_stack([zc(T[f]) for f in FEATS]).astype(np.float32)
yr=T["yr"].astype(int); se=T["se"].astype(int); ynow=np.digitize(zc(T["phyNow"]),EDGES).astype(int)

def base_rf(): return RandomForestClassifier(n_estimators=120,min_samples_leaf=5,
        class_weight="balanced_subsample",n_jobs=-1,random_state=0)
rows=[]
for L in LEADS:
    y=np.digitize(zc(T[f"L{L}"]),EDGES).astype(int)
    proba=np.zeros((y.size,3))
    for tr,te in GroupKFold(5).split(X,y,yr):
        # uncalibrated RF for the lead sweep (fast; we use the curve's SHAPE, not absolute Brier).
        m=base_rf().fit(X[tr],y[tr]); p=m.predict_proba(X[te])
        for j,c in enumerate(m.classes_): proba[te,c]=p[:,j]
    proba=np.clip(proba,1e-9,None); proba/=proba.sum(1,keepdims=True)
    acc=accuracy_score(y,proba.argmax(1))
    ycol=(y==0).astype(int); bs=brier_score_loss(ycol,proba[:,0])
    clim=np.zeros(y.size)
    for s in range(4): clim[se==s]=ycol[se==s].mean()
    bss_clim=1-bs/brier_score_loss(ycol,clim)
    bss_pers=1-bs/brier_score_loss(ycol,(ynow==0).astype(float))
    THR=0.3; fl=proba[:,0]>THR
    recall=(proba[ycol==1,0]>THR).mean()                       # collapses caught at P>THR
    precision=(ycol[fl]==1).mean() if fl.any() else np.nan     # warnings correct at P>THR
    rows.append((L,acc,bss_clim,bss_pers,recall,precision))
    log(f"lead {L:2d}d: acc {acc:.3f}  BSS_clim {bss_clim:+.3f}  BSS_pers {bss_pers:+.3f}  recall@0.3 {recall:.3f}  precision@0.3 {precision:.3f}")

print("\n=== forecast skill vs lead time ===")
print(f"{'lead(d)':>7s} {'accuracy':>9s} {'BSS_clim':>9s} {'BSS_pers':>9s} {'recall@.3':>10s} {'prec@.3':>9s}")
for L,acc,bc,bp,rc,pr in rows:
    print(f"{L:7d} {acc:9.3f} {bc:+9.3f} {bp:+9.3f} {rc:10.3f} {pr:9.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size":12})
L_=[r[0] for r in rows]
bss_pers=[r[3] for r in rows]
recall=[r[4] for r in rows]
prec=[r[5] for r in rows]
fig,ax=plt.subplots(1,2,figsize=(13,5.2))
a=ax[0]; a.axhline(0,color='0.55',lw=1); a.axvline(5,color='0.7',ls=':',lw=1.2)
a.plot(L_,[r[2] for r in rows],'-o',color='#1a7f37',lw=2,label='vs climatology')
a.plot(L_,bss_pers,'-o',color='#1c4f80',lw=2,label='vs persistence')
a.set_ylim(-0.5,0.34); a.set_xticks(L_)
a.set(title='A. Skill vs. baselines (Brier Skill Score)',xlabel='lead time (days)',ylabel='Brier Skill Score')
a.legend(frameon=False,loc='lower right')
b=ax[1]
b.plot(L_,recall,'-o',color='#1c4f80',lw=2,label='Collapses caught (recall)')
b.plot(L_,prec,'-o',color='#c1121f',lw=2,label='Warnings correct (precision)')
b.set_ylim(0,0.9); b.set_xticks(L_)
b.set(title='B. Warning performance (at P>0.3)',xlabel='lead time (days)',ylabel='fraction')
b.legend(frameon=False,loc='lower right')
fig.suptitle('Forecast skill vs. lead time  (JCOPE, 1993–2018)',fontsize=13,y=0.99)
fig.tight_layout(rect=[0,0,1,0.95]); os.makedirs("figures",exist_ok=True); fig.savefig("figures/jcope_leadtime.png",dpi=150)
print("\nsaved -> figures/jcope_leadtime.png")
