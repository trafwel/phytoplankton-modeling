#!/usr/bin/env python3
"""
jcope_forecast.py — THE final probabilistic result: a 7-day-ahead early-warning forecast of
phytoplankton collapse/normal/bloom from the JCOPE reanalysis, with calibration, skill vs
reference baselines, and marine-heatwave conditioning.

Task framing (this is a FORECAST, not a nowcast):
    features at day t  ->  phytoplankton class at day t+LEAD (LEAD=7)
Features (all at t): surface & subsurface temp/nutrient anomalies, their -7/-14/-28 d lags,
stratification, and season. The lags give trajectory; the target is the FUTURE state.

Outputs / validation:
  * out-of-fold ISOTONIC-calibrated P(collapse/normal/bloom), year-blocked CV.
  * SKILL vs baselines — the number that makes it a result:
      - climatology  (seasonal base rate of collapse)
      - persistence  (assume t+7 class = current class)
      Brier Skill Score = 1 - BS_model/BS_ref  (>0 means the forecast beats the baseline).
  * marine-heatwave conditioning: P(collapse in 7 d | warm now) vs (calm now).
  * reliability diagram + Brier; saved calibrated model.

Reads JCOPE depth files off HD-PGF-A once, caches the forecast table. Runs on the Mac.
    python src/jcope_forecast.py
"""
from __future__ import annotations
import os, glob, time, datetime as dt, warnings, numpy as np, joblib
os.environ["PYTHONWARNINGS"] = "ignore"; warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupKFold
from sklearn.metrics import log_loss, accuracy_score, brier_score_loss

NLAT, NLON, KSURF, K50 = 356, 362, 9, 5
HALFWIN, NCHUNK, EDGES = 5, 8, [-0.75, 0.75]; LAGS = (7, 14, 28); LEAD = 7
TARGET_ROWS = 1_200_000
CACHE = f"JCOPE Data/forecast_table_L{LEAD}.npz"
FEATS = ["tS","tS_l7","tS_l14","tS_l28","dS","dS_l7","dS_l14","dS_l28","d50","strat","sinD","cosD"]
NAMES = {0:"collapse", 1:"normal", 2:"bloom"}
SEASN = {12:0,1:0,2:0,3:1,4:1,5:1,6:2,7:2,8:2,9:3,10:3,11:3}; SNAME=["DJF","MAM","JJA","SON"]
t0 = time.time()
def log(*a): print(f"[{time.time()-t0:7.1f}s]", *a, flush=True)

def find_data():
    for c in sorted(glob.glob("/Volumes/*/JCOPE_full")):
        if glob.glob(f"{c}/t/t.*.npy"): return c
    raise SystemExit("ERROR: JCOPE_full not found — is HD-PGF-A connected?")

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
    def band(files, rows, levels):
        outs=[np.empty((D,rows.stop-rows.start,NLON),np.float32) for _ in levels]
        for i,f in enumerate(files):
            a=np.load(f,mmap_mode="r")
            for j,k in enumerate(levels): outs[j][i]=a[k,rows,:]
        return outs
    cols={k:[] for k in FEATS+["phyF","phyNow","yr","se"]}
    rng=np.random.RandomState(0); bnd=np.linspace(0,NLAT,NCHUNK+1).astype(int)
    for ci in range(NCHUNK):
        rows=slice(bnd[ci],bnd[ci+1])
        tS,t50=band(tf,rows,[KSURF,K50]); dS,d50=band(df,rows,[KSURF,K50]); (pS,)=band(pf,rows,[KSURF])
        aTS,aDS,aD50=anom(tS),anom(dS),anom(d50); aStr,aPhy=anom(tS-t50),anom(pS)
        feat={"tS":aTS,"dS":aDS,"d50":aD50,"strat":aStr}
        for L in LAGS: feat[f"tS_l{L}"]=lag(aTS,L); feat[f"dS_l{L}"]=lag(aDS,L)
        feat["sinD"]=np.broadcast_to(sinD[:,None,None],aTS.shape); feat["cosD"]=np.broadcast_to(cosD[:,None,None],aTS.shape)
        aPhyF=np.full_like(aPhy,np.nan); aPhyF[:-LEAD]=aPhy[LEAD:]   # target = phy at t+LEAD
        fin=np.isfinite(aPhyF)&np.isfinite(aPhy)
        for f in FEATS: fin=fin&np.isfinite(feat[f])
        idx=np.where(fin.ravel())[0]
        take=rng.choice(idx,min(TARGET_ROWS//NCHUNK,idx.size),replace=False)
        for f in FEATS: cols[f].append(feat[f].ravel()[take])
        cols["phyF"].append(aPhyF.ravel()[take]); cols["phyNow"].append(aPhy.ravel()[take])
        cols["yr"].append(np.broadcast_to(year[:,None,None],aTS.shape).ravel()[take])
        cols["se"].append(np.broadcast_to(sea[:,None,None],aTS.shape).ravel()[take])
        log(f"chunk {ci+1}/{NCHUNK}: +{take.size:,}")
    T={k:np.concatenate(v).astype(np.float32) for k,v in cols.items()}
    np.savez_compressed(CACHE, **T); log(f"cached -> {CACHE}")
    return T

T = dict(np.load(CACHE)) if os.path.exists(CACHE) else build_cache()
zc=lambda a:a/np.nanstd(a)
X=np.column_stack([zc(T[f]) for f in FEATS]).astype(np.float32)
y=np.digitize(zc(T["phyF"]), EDGES).astype(int)          # FUTURE class (t+7) — the target
ynow=np.digitize(zc(T["phyNow"]), EDGES).astype(int)     # current class — for persistence
yr=T["yr"].astype(int); se=T["se"].astype(int)
log(f"{y.size:,} rows | future-class balance " + " ".join(f"{NAMES[n]} {(y==n).mean():.0%}" for n in (0,1,2)))

def base_rf(): return RandomForestClassifier(n_estimators=120, min_samples_leaf=5,
        class_weight="balanced_subsample", n_jobs=-1, random_state=0)

# ---- out-of-fold calibrated probabilities (year-blocked) ----
proba=np.zeros((y.size,3),np.float32)
for k,(tr,te) in enumerate(GroupKFold(5).split(X,y,yr),1):
    cal=CalibratedClassifierCV(base_rf(), method="isotonic", cv=3).fit(X[tr],y[tr])
    p=cal.predict_proba(X[te])
    for j,c in enumerate(cal.classes_): proba[te,c]=p[:,j]
    log(f"  fold {k} done")
proba=np.clip(proba,1e-6,None); proba/=proba.sum(1,keepdims=True)
pred=proba.argmax(1)
print(f"\n=== 7-DAY FORECAST — phytoplankton collapse/normal/bloom ===")
print(f"overall  log-loss {log_loss(y,proba,labels=[0,1,2]):.3f}  acc {accuracy_score(y,pred):.3f}")
for s in range(4):
    m=se==s; print(f"  {SNAME[s]:4s}  log-loss {log_loss(y[m],proba[m],labels=[0,1,2]):.3f}  acc {accuracy_score(y[m],pred[m]):.3f}")

# ---- SKILL vs baselines (collapse event) ----
ycol=(y==0).astype(int)
bs_model=brier_score_loss(ycol, proba[:,0])
clim=np.zeros(y.size)                                     # seasonal climatological collapse rate
for s in range(4): clim[se==s]=ycol[se==s].mean()
bs_clim=brier_score_loss(ycol, clim)
bs_pers=brier_score_loss(ycol, (ynow==0).astype(float))  # persistence: current collapse -> future
print("\n=== SKILL for P(collapse in 7 days) — Brier score (lower better) ===")
print(f"  model        {bs_model:.4f}")
print(f"  climatology  {bs_clim:.4f}   -> Brier Skill Score vs clim = {1-bs_model/bs_clim:+.3f}")
print(f"  persistence  {bs_pers:.4f}   -> Brier Skill Score vs pers = {1-bs_model/bs_pers:+.3f}")
print("  (BSS > 0 => the forecast beats that baseline)")

# ---- marine-heatwave conditioning ----
tS_now=zc(T["tS"]); warm=tS_now > np.nanpercentile(tS_now,90)
print("\n=== heatwave conditioning: observed collapse rate 7 days later ===")
print(f"  after a WARM day (top-10% SST anom): {ycol[warm].mean():.1%}")
print(f"  after a calm day:                    {ycol[~warm].mean():.1%}")
print(f"  model mean P(collapse) | warm now:   {proba[warm,0].mean():.1%}  | calm: {proba[~warm,0].mean():.1%}")

# ---- reliability + save ----
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig,ax=plt.subplots(figsize=(6.5,6)); bins=np.linspace(0,1,11)
pc=proba[:,0]; idx=np.digitize(pc,bins)-1; mp,mo=[],[]
for b in range(10):
    m=idx==b
    if m.sum()>50: mp.append(pc[m].mean()); mo.append(ycol[m].mean())
ax.plot([0,1],[0,1],"k--",label="perfect"); ax.plot(mp,mo,"-o",color="#c1121f",label="P(collapse)")
ax.set(title="7-day collapse forecast — reliability", xlabel="predicted P(collapse)",
       ylabel="observed collapse frequency", xlim=(0,1), ylim=(0,1)); ax.legend(frameon=False)
fig.tight_layout(); os.makedirs("figures",exist_ok=True); fig.savefig("figures/jcope_forecast_reliability.png",dpi=140)
print("\nsaved -> figures/jcope_forecast_reliability.png")
final=CalibratedClassifierCV(base_rf(),method="isotonic",cv=3).fit(X,y)
os.makedirs("models",exist_ok=True)
joblib.dump({"model":final,"features":FEATS,"classes":NAMES,"lead_days":LEAD,"edges":EDGES},
            "models/jcope_forecast_7d.joblib")
print("saved -> models/jcope_forecast_7d.joblib")
