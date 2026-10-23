"""Leave-one-source-out (grouped) cross-validation for the EM-VRF ΔV model.

Uses the repository's train_model.py for data loading, ΔX_Na recalculation, the model recipe and metrics,
so the only thing that changes relative to the published 20 x 5 CV is how records are split.
usage: python loso_cv.py REPO_DIR DOI_CSV OUT_DIR
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import KFold, LeaveOneGroupOut, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO, DOI_CSV, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sys.path.insert(0, str(REPO))
import train_model as tm  # noqa: E402

OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()

x, y, rows = tm.load_training_data()
n = len(y)
meta = json.loads(tm.MODEL_META.read_text(encoding='utf-8'))['DeltaV']

# ------------------------------------------------------------------ groups (source papers)
doi = pd.read_csv(DOI_CSV, encoding='utf-8-sig')
doi_by_id = dict(zip(doi['Sample ID'].astype(int), doi['DOI'].str.lower()))
ids = rows['Sample ID'].astype(int).to_numpy()
groups_listed = np.array([doi_by_id[i] for i in ids])
groups_fixed = groups_listed.copy()
groups_fixed[ids == 30] = doi_by_id[29]          # record 30 most likely comes from the record-29 paper
GROUPINGS = {'as_listed': groups_listed, 'record30_with_29': groups_fixed}

ALPHAS = np.logspace(-3, 3, 13)
SVR_GRID = tm.NESTED_GRID


def svr(params=None):
    return tm.delta_recipe(params)


def ridge(alpha):
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def linear():
    return make_pipeline(StandardScaler(), LinearRegression())


def splits_logo(idx, g):
    return list(LeaveOneGroupOut().split(idx, groups=g[idx]))


def pooled_rmse(make, X, Y, folds):
    pred = np.zeros(len(Y))
    for tr, te in folds:
        pred[te] = make().fit(X[tr], Y[tr]).predict(X[te])
    return float(np.sqrt(np.mean((Y - pred) ** 2)))


def fit_predict(kind, Xtr, Ytr, Xte, inner_folds):
    """kind: 'mean' | 'linear' | 'ridge_tuned' | 'svr_fixed' | 'svr_tuned'. Returns (pred, chosen)."""
    if kind == 'mean':
        return np.full(len(Xte), Ytr.mean()), None
    if kind == 'linear':
        return linear().fit(Xtr, Ytr).predict(Xte), None
    if kind == 'svr_fixed':
        return svr().fit(Xtr, Ytr).predict(Xte), None
    if kind == 'ridge_tuned':
        best = min(ALPHAS, key=lambda a: pooled_rmse(lambda: ridge(a), Xtr, Ytr, inner_folds))
        return ridge(best).fit(Xtr, Ytr).predict(Xte), float(best)
    if kind == 'svr_tuned':
        best = min(SVR_GRID, key=lambda p: pooled_rmse(lambda: svr(p), Xtr, Ytr, inner_folds))
        return svr(best).fit(Xtr, Ytr).predict(Xte), json.dumps(best, sort_keys=True)
    raise ValueError(kind)


def loso(kind, X, g):
    pred = np.zeros(n)
    chosen = []
    for tr, te in LeaveOneGroupOut().split(X, groups=g):
        inner = list(LeaveOneGroupOut().split(X[tr], groups=g[tr])) if kind.endswith('tuned') else None
        p, c = fit_predict(kind, X[tr], y[tr], X[te], inner)
        pred[te] = p
        chosen.append(c)
    return pred, chosen


def random_repeated(kind, X, repeats=20, seed=tm.SEED):
    """Mean OOF prediction over repeated random 5-fold CV (inner tuning by random 5-fold)."""
    sums = np.zeros(n)
    for r, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed).split(X)):
        inner = list(KFold(5, shuffle=True, random_state=909 + r).split(X[tr])) if kind.endswith('tuned') else None
        sums[te] += fit_predict(kind, X[tr], y[tr], X[te], inner)[0]
    return sums / repeats


def grouped_repeated(kind, X, g, repeats=20, seed=tm.SEED):
    """Mean OOF prediction over repeated grouped 5-fold CV: whole source papers are assigned to folds,
    balancing fold sizes (largest groups first into the currently smallest fold, ties broken at random)."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(g)
    sums = np.zeros(n)
    for r in range(repeats):
        order = rng.permutation(uniq)
        order = sorted(order, key=lambda u: -np.sum(g == u))  # stable sort keeps random order within sizes
        fold_of, sizes = {}, np.zeros(5, int)
        for u in order:
            k = int(np.argmin(sizes + rng.random(5) * 1e-3))
            fold_of[u] = k
            sizes[k] += int(np.sum(g == u))
        fid = np.array([fold_of[u] for u in g])
        for k in range(5):
            tr, te = np.where(fid != k)[0], np.where(fid == k)[0]
            inner = list(LeaveOneGroupOut().split(X[tr], groups=g[tr])) if kind.endswith('tuned') else None
            sums[te] += fit_predict(kind, X[tr], y[tr], X[te], inner)[0]
    return sums / repeats


def h_width(pred):
    return tm.conformal_half_width(y, pred)[0]


report = {'n_records': int(n), 'environment': {'scikit_learn': __import__('sklearn').__version__,
                                               'numpy': np.__version__}}

# 1. reproduce the published random 20x5 CV
rand_svr = tm.repeated_oof(x, y)
m = tm.metric_dict(y, rand_svr)
report['random_20x5_svr_fixed'] = m
report['reproduces_model_meta'] = max(tm.metric_differences(m, meta['fixed_model_repeated_20x5_cv_metrics']).values())
print('random 20x5 reproduced, max diff from model_meta =', report['reproduces_model_meta'], flush=True)

# 2. group structure
for name, g in GROUPINGS.items():
    u, c = np.unique(g, return_counts=True)
    report.setdefault('groups', {})[name] = {'n_sources': int(len(u)),
                                             'multi_record_sources': {k: int(v) for k, v in zip(u, c) if v > 1}}

# 3. main comparison: every model under random 20x5, LOSO and grouped 20x5
KINDS = ['mean', 'linear', 'ridge_tuned', 'svr_fixed', 'svr_tuned']
preds = {}
for kind in KINDS:
    preds[('random', kind)] = rand_svr if kind == 'svr_fixed' else random_repeated(kind, x)
    print('random', kind, round(time.time() - t0), 's', flush=True)
for gname, g in GROUPINGS.items():
    for kind in KINDS:
        p, chosen = loso(kind, x, g)
        preds[('loso_' + gname, kind)] = p
        if kind == 'svr_tuned':
            report.setdefault('loso_svr_tuned_selected', {})[gname] = pd.Series(chosen).value_counts().to_dict()
        print('loso', gname, kind, round(time.time() - t0), 's', flush=True)
    for kind in ['mean', 'linear', 'svr_fixed']:
        preds[('grouped20x5_' + gname, kind)] = grouped_repeated(kind, x, g)
        print('grouped20x5', gname, kind, round(time.time() - t0), 's', flush=True)

table = []
for (scheme, kind), p in preds.items():
    mm = tm.metric_dict(y, p)
    table.append({'scheme': scheme, 'model': kind, **{k: mm[k] for k in ('R2', 'RMSE', 'MAE', 'Pearson_r')},
                  'interval_half_width_44of45': h_width(p)})
report['metrics'] = table

# 4. descriptor ablation (fixed SVR) under random 20x5 and LOSO (record 30 grouped with 29)
combos = {'dXNa': [2], 'Ravg+PhiTM': [0, 1], 'dXNa+Ravg': [0, 2], 'dXNa+PhiTM': [1, 2], 'all three': [0, 1, 2]}
abl = []
for name, cols in combos.items():
    xr = x[:, cols]
    pr = tm.repeated_oof(xr, y) if True else None
    pl, _ = loso('svr_fixed', xr, GROUPINGS['record30_with_29'])
    abl.append({'descriptors': name, 'random_20x5_RMSE': tm.metric_dict(y, pr)['RMSE'],
                'loso_RMSE': tm.metric_dict(y, pl)['RMSE'], 'random_20x5_R2': tm.metric_dict(y, pr)['R2'],
                'loso_R2': tm.metric_dict(y, pl)['R2']})
report['ablation_svr_fixed'] = abl

# 5. per-record errors and subsets
g = GROUPINGS['record30_with_29']
size = pd.Series(g).map(pd.Series(g).value_counts()).to_numpy()
per = pd.DataFrame({'Sample ID': ids, 'Formula': rows['Formula'], 'DOI_group': g, 'records_in_source': size,
                    'DeltaV_measured': y, 'pred_random_20x5': rand_svr,
                    'pred_loso': preds[('loso_record30_with_29', 'svr_fixed')]})
per['abs_err_random'] = (per['DeltaV_measured'] - per['pred_random_20x5']).abs()
per['abs_err_loso'] = (per['DeltaV_measured'] - per['pred_loso']).abs()
per['change'] = per['abs_err_loso'] - per['abs_err_random']
per.sort_values('change', ascending=False).to_csv(OUT / 'per_record_errors.csv', index=False, encoding='utf-8-sig',
                                                  float_format='%.4f')
subsets = {}
for label, mask in {'records from multi-record sources': size > 1, 'single-record sources': size == 1}.items():
    subsets[label] = {'n': int(mask.sum()),
                      'random_20x5': tm.metric_dict(y[mask], rand_svr[mask]),
                      'loso': tm.metric_dict(y[mask], per['pred_loso'].to_numpy()[mask])}
report['subsets'] = subsets
report['runtime_s'] = round(time.time() - t0, 1)
(OUT / 'loso_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False, default=float), encoding='utf-8')
pd.DataFrame(table).to_csv(OUT / 'metrics_table.csv', index=False, float_format='%.4f')
print(pd.DataFrame(table).round(3).to_string(index=False))
print(pd.DataFrame(abl).round(3).to_string(index=False))
print(json.dumps(subsets, indent=1, default=float))
print('done in', report['runtime_s'], 's')
