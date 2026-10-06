"""
Probability-theory falsification of zero-shot individual behavior prediction.

Setup: each subject i has a latent bias theta_i ~ N(0, s^2). On trial t with
stimulus value x_it, choice prob = sigmoid(beta*x_it + theta_i). A 'zero-shot'
predictor sees only the POPULATION (theta=0); an 'enrolled' Bayesian predictor
updates a posterior over theta_i from the subject's own past choices.

Claim (to falsify zero-shot individual prediction): the zero-shot Bayes-optimal
individual forecast equals the population forecast, so for the individual-specific
component it has AUC ~ 0.5 and zero mutual information. Enrollment is required and
its value grows with trials -> the 'price of individuation', empirically.
"""
import numpy as np
from pathlib import Path
import matplotlib as mpl, matplotlib.pyplot as plt

rng = np.random.default_rng(0)
OUT = Path.home()/"Desktop/Research/submissions/neuro-ai/insilico-neuroforecasting-validity/figures"

def sigmoid(z): return 1/(1+np.exp(-z))

def auc(y, p):
    y = np.asarray(y); p = np.asarray(p)
    pos = p[y==1]; neg = p[y==0]
    if len(pos)==0 or len(neg)==0: return np.nan
    # Mann-Whitney U / (n_pos*n_neg)
    order = np.argsort(np.concatenate([pos,neg]))
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1,len(order)+1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def run(n_subjects=300, s=1.3, beta=1.0, trials_grid=(0,2,5,10,20,50,100,200), reps=3):
    """Two clean metrics of INDIVIDUAL value vs enrollment budget k:
      recovery   = corr(theta_hat, theta)            (0 at zero-shot, rises with k)
      dAUC       = AUC_full - AUC_population          (incremental held-out choice AUC)
    Zero-shot (k=0) => posterior = prior => theta_hat = 0 for everyone => both ~0.
    """
    rec = {k: [] for k in trials_grid}; dauc = {k: [] for k in trials_grid}
    for _ in range(reps):
        theta = rng.normal(0, s, n_subjects)
        grid = np.linspace(-5*s, 5*s, 201); prior = np.exp(-grid**2/(2*s**2)); prior/=prior.sum()
        for k in trials_grid:
            th_hat = np.zeros(n_subjects)
            yall, pfull, ppop = [], [], []
            for i in range(n_subjects):
                if k > 0:
                    xe = rng.normal(0,1,k); ye = (rng.random(k) < sigmoid(beta*xe+theta[i])).astype(int)
                    logpost = np.log(prior+1e-12)
                    for t in range(k):
                        pt = sigmoid(beta*xe[t] + grid)
                        logpost += np.log(np.where(ye[t]==1, pt, 1-pt)+1e-12)
                    post = np.exp(logpost-logpost.max()); post/=post.sum()
                    th_hat[i] = (post*grid).sum()
                xt = rng.normal(0,1,40); yt = (rng.random(40) < sigmoid(beta*xt+theta[i])).astype(int)
                yall.append(yt); pfull.append(sigmoid(beta*xt+th_hat[i])); ppop.append(sigmoid(beta*xt))
            yall=np.concatenate(yall); pfull=np.concatenate(pfull); ppop=np.concatenate(ppop)
            r = 0.0 if k==0 else float(np.corrcoef(th_hat, theta)[0,1])
            rec[k].append(r)
            dauc[k].append(auc(yall, pfull) - auc(yall, ppop))
    return ({k: float(np.mean(v)) for k,v in rec.items()},
            {k: float(np.mean(v)) for k,v in dauc.items()})

rec, dauc = run()
print("k  recovery_r  dAUC(full-pop)")
for k in rec: print(f"  {k:4d}   r={rec[k]:+.3f}   dAUC={dauc[k]:+.3f}")
res = dauc  # for the figure below (incremental individual AUC)

# figure
mpl.rcParams.update({"font.family":"serif","font.size":9,"pdf.fonttype":42,
                     "axes.spines.top":False,"axes.spines.right":False})
C=dict(purple="#CC79A7",red="#D55E00",grey="#999999",ink="#222222")
ks=list(res.keys()); vs=[res[k] for k in ks]
fig,ax=plt.subplots(figsize=(4.7,3.0)); fig.subplots_adjust(left=0.17,right=0.95,top=0.86,bottom=0.22)
ax.set_title("Bayes-optimal individual prediction: the price of enrollment",
             loc="left",fontweight="bold",fontsize=9.3)
ax.axhline(0.0,color=C["grey"],ls=":",lw=1)
ax.plot(ks,vs,"-o",color=C["purple"],lw=2.2,ms=5,label="incremental AUC over population")
ax.plot(0,res[0],"o",color=C["red"],ms=9,zorder=3)
ax.annotate("zero-shot = population\n(no individual value)",xy=(0,res[0]),xytext=(22,0.012),
            fontsize=7.2,color=C["red"],arrowprops=dict(arrowstyle="->",color=C["red"],lw=1))
ax.set_xlabel("per-subject enrollment trials")
ax.set_ylabel(r"$\Delta$AUC (individual $-$ population)")
ax.text(0.0,-0.32,"A Bayes-optimal forecaster adds zero individual value at zero-shot; value "
        "appears only once\nper-subject data is paid for, and saturates well below certainty.",
        transform=ax.transAxes,fontsize=6.8,color=C["ink"])
fig.savefig(OUT/"fig5_bayes_enrollment.pdf",bbox_inches="tight")
fig.savefig(OUT/"fig5_bayes_enrollment.png",bbox_inches="tight")
print("wrote fig5 ->",OUT)
