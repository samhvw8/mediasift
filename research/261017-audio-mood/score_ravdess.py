import sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import balanced_accuracy_score
c = np.load(sys.argv[1]); e = np.load(sys.argv[2]); assert (c["files"] == e["files"]).all()
EMO = {1:"neutral",2:"neutral",3:"happy",4:"sad",5:"angry",6:"fearful",7:"disgusted",8:"surprised"}
code = np.array([int(f.split("-")[2]) for f in c["files"]]); actor = np.array([int(f.split("-")[6][:2]) for f in c["files"]])
y = np.array([EMO[k] for k in code]); test = actor >= 17; train = ~test
HI = {"angry","happy","fearful","surprised"}; LO = {"neutral","sad"}
ar = np.array([1 if v in HI else 0 if v in LO else -1 for v in y]); arm = ar >= 0
L = [str(x) for x in e["labels"]]; C7 = ["angry","disgusted","fearful","happy","neutral","sad","surprised"]
def show(name, pred7, arousal_score=None):
    acc = balanced_accuracy_score(y[test], pred7[test])
    line = f"{name:<34} 7-way bal.acc {acc:.0%} (chance 14%)"
    if arousal_score is not None:
        m = test & arm; line += f" | arousal bal.acc {balanced_accuracy_score(ar[m], arousal_score[m] > 0.5):.0%} (chance 50%)"
    print(line)
for tag in ("fp32", "int8"):
    P = e[tag][:, [L.index(k) for k in C7]]
    show(f"emotion2vec+ {tag} zero-shot", np.array(C7)[P.argmax(1)], P[:, [C7.index(k) for k in HI]].sum(1) / P.sum(1))
def probe(name, X):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")).fit(X[train], y[train])
    ca = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced")).fit(X[train & arm], ar[train & arm])
    show(name, clf.predict(X), ca.predict_proba(X)[:, 1])
ced = c["ced"]; keep = ced[train].max(0) > 0.01
probe("probe: prosody (9 numbers)", c["pros"])
probe("probe: CED scores", np.log(ced[:, keep] + 1e-4))
probe("probe: CED + prosody", np.hstack([np.log(ced[:, keep] + 1e-4), c["pros"]]))
probe("probe: emotion2vec int8 embedding", e["int8_emb"])
