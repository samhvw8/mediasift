# Judge: AudioSet eval labels (held out from CED training). Maps fixed before results were seen.
import sys, json, numpy as np
from sklearn.metrics import roc_auc_score
ced = np.load(sys.argv[1]); eff = np.load(sys.argv[2])
MOODS = ["Happy","Funny","Sad","Tender","Exciting","Angry","Scary"]
GENRES = ["Pop","HipHop","Rock","Metal","Reggae","Country","Jazz","Classical","Electronic"]
CED_ID = dict(zip(MOODS, range(276, 283))) | {"Pop":216,"HipHop":217,"Rock":219,"Metal":220,"Reggae":228,"Country":229,"Jazz":235,"Classical":237,"Electronic":239}
MT = ['action','adventure','advertising','background','ballad','calm','children','christmas','commercial','cool','corporate','dark','deep','documentary','drama','dramatic','dream','emotional','energetic','epic','fast','film','fun','funny','game','groovy','happy','heavy','holiday','hopeful','inspiring','love','meditative','melancholic','melodic','motivational','movie','nature','party','positive','powerful','relaxing','retro','romantic','sad','sexy','slow','soft','soundscape','space','sport','summer','trailer','travel','upbeat','uplifting']
EFF_MOOD = {"Happy":["happy","upbeat","positive","fun"],"Funny":["funny","fun","children"],"Sad":["sad","melancholic","emotional"],
            "Tender":["soft","romantic","love","calm","relaxing"],"Exciting":["energetic","action","fast","powerful"],
            "Angry":["heavy","powerful"],"Scary":["dark","dramatic","deep"]}
G400 = json.load(open(sys.argv[3]))["classes"]
EFF_GEN = {"Pop":lambda c:c.startswith("Pop---"),"HipHop":lambda c:c.startswith("Hip Hop---"),"Rock":lambda c:c.startswith("Rock---") and "Metal" not in c,
           "Metal":lambda c:"Metal" in c,"Reggae":lambda c:c.startswith("Reggae---"),"Country":lambda c:c.endswith("---Country") or "Country" in c.split("---")[1],
           "Jazz":lambda c:c.startswith("Jazz---"),"Classical":lambda c:c.startswith("Classical---"),"Electronic":lambda c:c.startswith("Electronic---")}
names = [str(n) for n in ced["names"]]; idx = {n: i for i, n in enumerate(str(x) for x in eff["names"])}
y = [n.split("/")[0] for n in names]; E = {k: eff[k][[idx[n] for n in names]] for k in eff.files if k not in ("names","prompts")}
z = lambda a: (a - a.mean(0)) / (a.std(0) + 1e-9)
def report(title, classes, scores):
    sel = np.array([c in classes for c in y]); yy = np.array(y)[sel]; S = scores[sel]
    for norm in ("raw", "z"):
        X = z(S) if norm == "z" else S
        acc = (np.array(classes)[X.argmax(1)] == yy).mean()
        auc = np.mean([roc_auc_score(yy == c, X[:, k]) for k, c in enumerate(classes)])
        per = " ".join(f"{c}:{(np.array(classes)[X[yy==c].argmax(1)]==c).mean():.0%}" for c in classes)
        print(f"{title:<28} {norm:<3} n={sel.sum()} top1 {acc:.0%} (chance {1/len(classes):.0%})  macroAUC {auc:.2f} | {per}")
cedp = ced["probs"]
report("CED mood (free)", MOODS, cedp[:, [CED_ID[m] for m in MOODS]])
report("EffNet moodtheme", MOODS, np.stack([E["moodtheme"][:, [MT.index(t) for t in EFF_MOOD[m]]].mean(1) for m in MOODS], 1))
if "clap" in E: report("CLAP zero-shot", MOODS, E["clap"][:, :7])
report("CED genre (free)", GENRES, cedp[:, [CED_ID[g] for g in GENRES]])
report("EffNet Discogs400", GENRES, np.stack([E["genre400"][:, [i for i, c in enumerate(G400) if EFF_GEN[g](c)]].max(1) for g in GENRES], 1))
if "clap" in E: report("CLAP zero-shot", GENRES, E["clap"][:, 7:16])
