# Cross-dataset judge: probes trained on all RAVDESS, tested on EmoNet-Voice strong-consensus clips.
import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
rc, re_, ec, ee = (np.load(p) for p in sys.argv[1:5])
EMO = {1:"neutral",2:"neutral",3:"happy",4:"sad",5:"angry",6:"fearful",7:"disgusted",8:"surprised"}
ry = np.array([EMO[int(f.split("-")[2])] for f in rc["files"]])
HI = {"angry","happy","fearful","surprised","hi"}; LO = {"neutral","sad","lo"}
rar = np.array([1 if v in HI else 0 if v in LO else -1 for v in ry])
ey = np.array([str(f).split("/")[0] for f in ec["files"]]); ear = np.array([1 if v in HI else 0 if v in LO else -1 for v in ey])
cat = np.isin(ey, ["angry","fearful","happy","sad","surprised","disgusted"]); C6 = ["angry","disgusted","fearful","happy","sad","surprised"]
L = [str(x) for x in re_["labels"]]
def show(name, pred, ascore):
    m = ear >= 0
    print(f"{name:<36} 6-way bal.acc {balanced_accuracy_score(ey[cat], pred[cat]):.0%} (chance 17%) n={cat.sum()} | arousal bal.acc {balanced_accuracy_score(ear[m], ascore[m] > 0.5):.0%} (chance 50%) n={m.sum()}")
for tag in ("fp32", "int8"):
    P = ee[tag][:, [L.index(k) for k in C6]]; Pa = ee[tag]
    show(f"emotion2vec+ {tag} zero-shot", np.array(C6)[P.argmax(1)], Pa[:, [L.index(k) for k in ("angry","happy","fearful","surprised")]].sum(1) / Pa[:, [L.index(k) for k in ("angry","happy","fearful","surprised","neutral","sad")]].sum(1))
def probe(name, Xr, Xe):
    sel = np.isin(ry, C6)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")).fit(Xr[sel], ry[sel])
    ca = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")).fit(Xr[rar >= 0], rar[rar >= 0])
    show(name, clf.predict(Xe), ca.predict_proba(Xe)[:, 1])
keep = rc["ced"].max(0) > 0.01; lg = lambda a: np.log(a[:, keep] + 1e-4)
probe("probe(RAVDESS): prosody", rc["pros"], ec["pros"])
probe("probe(RAVDESS): CED scores", lg(rc["ced"]), lg(ec["ced"]))
probe("probe(RAVDESS): e2v int8 embedding", re_["int8_emb"], ee["int8_emb"])
