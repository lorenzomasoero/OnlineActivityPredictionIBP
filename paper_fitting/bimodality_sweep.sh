#!/bin/bash
# One-off detached characterization of REES46-28day TG-SSP bimodality (50 seeds).
# Records per-seed 7-window TG accuracies -> bimodality_tg.csv, for an appendix note.
set -u
cd /local/home/masoerl/PredictNewCustomers/SubmissionAOAS/paper-code/paper_fitting || exit 1
PY=/tmp/pcvenv/bin/python
echo "START $(date -Is)"
for s in $(seq 0 49); do
  SEED=$s OUTDIR="$PWD/out/bimodal/rees28_s$s" $PY fit_rees46.py > /dev/null 2>&1 && echo "OK seed $s"
done
$PY - <<'PYEOF'
import numpy as np, glob, os, csv
rows=[]; per_seed=[]
for d in sorted(glob.glob("out/bimodal/rees28_s*"), key=lambda x:int(x.split("_s")[-1])):
    s=int(d.split("_s")[-1]); p=os.path.join(d,"rees46_all_results.npy")
    if not os.path.exists(p): continue
    a=np.load(p,allow_pickle=True); a=a.item() if a.dtype==object and a.shape==() else a
    vals=[]
    for r in a.values():
        if isinstance(r,dict) and "TG_SSP" in r and "accuracy_v" in r["TG_SSP"]:
            v=float(r["TG_SSP"]["accuracy_v"]); vals.append(v)
            rows.append({"seed":s,"exp_id":r.get("exp_id"),"tg_accuracy_v":round(v,4)})
    if vals: per_seed.append((s, float(np.median(vals))))
with open("bimodality_tg.csv","w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["seed","exp_id","tg_accuracy_v"]); w.writeheader(); w.writerows(rows)
meds=np.array([m for _,m in per_seed])
lo=meds[meds<0.52]; hi=meds[meds>=0.52]
print(f"\nSeeds={len(meds)}  per-seed TG median: overall median={np.median(meds):.3f} mean={np.mean(meds):.3f}")
print(f"  low mode (<0.52):  n={len(lo)}  mean={lo.mean():.3f}" if len(lo) else "  low mode: none")
print(f"  high mode (>=0.52): n={len(hi)} mean={hi.mean():.3f}" if len(hi) else "  high mode: none")
print("Wrote bimodality_tg.csv")
PYEOF
echo "DONE $(date -Is)"
