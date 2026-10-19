"""Training-subject-only four-role data; normalization is fitted on inner fitting subjects."""
import io
import json
import sys
import zipfile
from pathlib import Path
import numpy as np

PACKAGE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PACKAGE/'pipeline_snapshot'))
from makd_har.data import _iter_wisdm_segments,extract_wisdm_features,UCI_SIGNAL_NAMES

ROLES=('fit','stop','value','select')


def require(ok,message):
    if not ok:raise RuntimeError(message)


def source_record(drive,dataset):
    relative='results/uci/production' if dataset=='uci' else 'results/wisdm/production-v2'
    return json.loads((drive/relative/'seed-11.json').read_text())


def read_training(drive,dataset,arrays=False):
    source=source_record(drive,dataset);saved=source['dataset_manifest']
    eligible=saved['train']['subjects'];raw=[];features=[];labels=[];subjects=[]
    if dataset=='uci':
        with zipfile.ZipFile(drive/'resources/UCI HAR Dataset.zip') as archive:
            prefixes=[name[:-len('train/y_train.txt')] for name in archive.namelist() if name.endswith('train/y_train.txt')]
            require(len(prefixes)==1,'Ambiguous UCI archive')
            prefix=prefixes[0]
            def load(name):return np.loadtxt(io.BytesIO(archive.read(prefix+'train/'+name)))
            y=load('y_train.txt').astype(np.int64)-1;s=load('subject_train.txt').astype(np.int64)
            take=np.isin(s,eligible);labels=y[take];subjects=s[take]
            if arrays:
                raw=np.stack([load('Inertial Signals/'+name+'_train.txt')[take] for name in UCI_SIGNAL_NAMES],axis=-1).astype(np.float32)
                features=load('X_train.txt')[take].astype(np.float32)
    else:
        for subject,label,segment in _iter_wisdm_segments(drive/'resources/WISDM_ar_v1.1_raw.txt'):
            if subject not in eligible:continue
            for start in range(0,len(segment)-128+1,64):
                labels.append(label);subjects.append(subject)
                if arrays:
                    window=segment[start:start+128];raw.append(window)
                    vector,_=extract_wisdm_features(window,20.0);features.append(vector)
        labels=np.asarray(labels,dtype=np.int64);subjects=np.asarray(subjects,dtype=np.int64)
        if arrays:raw=np.asarray(raw,dtype=np.float32);features=np.asarray(features,dtype=np.float32)
    observed=part_manifest(labels,subjects)
    require(observed==saved['train'],'Training subjects/counts differ from frozen production metadata')
    require(not set(eligible)&set(saved['validation']['subjects']+saved['test']['subjects']),'Original subject leakage')
    if arrays:return raw,features,labels,subjects
    return labels,subjects


def part_manifest(labels,subjects):
    return dict(subjects=np.unique(subjects).astype(int).tolist(),samples=len(labels),
        class_counts={str(i):int((labels==i).sum()) for i in range(6)})


def coverage(labels,subjects,groups):
    reports={}
    for role in ROLES:
        take=np.isin(subjects,groups[role]);y=labels[take];s=subjects[take]
        report=part_manifest(y,s)
        report['subjects_per_class']={str(i):len(np.unique(s[y==i])) for i in range(6)}
        reports[role]=report
    return reports


def choose_roles(labels,subjects):
    """First metadata-feasible permutation; never sees model performance."""
    unique=np.unique(subjects);held=max(2,int(round(len(unique)*.15)))
    require(len(unique)-3*held>=4,'Too few subjects for four development roles')
    for offset in range(10000):
        shuffled=np.random.default_rng(20261018+offset).permutation(unique).astype(int).tolist()
        groups=dict(stop=sorted(shuffled[:held]),value=sorted(shuffled[held:2*held]),
                    select=sorted(shuffled[2*held:3*held]),fit=sorted(shuffled[3*held:]))
        report=coverage(labels,subjects,groups)
        if all(min(r['class_counts'].values())>=20 and min(r['subjects_per_class'].values())>=2 for r in report.values()):
            return dict(groups=groups,coverage=report,permutation_seed=20261018+offset,
                eligibility='Original production training subjects only; original validation and test excluded',
                allocation='Each held-out role round(0.15*n_subjects), minimum two; fitting uses remainder',
                feasibility='Every role has >=20 windows and >=2 distinct subjects for each of six classes',
                choice='First feasible permutation among seeds 20261018 + offset, offset 0..9999; metadata only')
    raise RuntimeError('Four-role subject coverage infeasible under predeclared search; specify grouped alternative before training')


def validate_groups(labels,subjects,split):
    groups=split['groups'];sets=[set(groups[r]) for r in ROLES]
    require(all(not sets[i]&sets[j] for i in range(4) for j in range(i)),'Development subject leakage')
    require(set.union(*sets)==set(np.unique(subjects).astype(int)),'Missing/extra training subjects')
    observed=coverage(labels,subjects,groups)
    require(observed==split['coverage'],'Changed role coverage')
    require(all(min(r['class_counts'].values())>=20 and min(r['subjects_per_class'].values())>=2 for r in observed.values()),
            'Insufficient role/class coverage')


def load_roles(drive,dataset,split):
    raw,features,y,s=read_training(drive,dataset,True);validate_groups(y,s,split)
    fit=np.isin(s,split['groups']['fit'])
    # Double-precision accumulation avoids cancellation when centering nearly constant channels/features.
    raw_mean=raw[fit].mean(axis=(0,1),keepdims=True,dtype=np.float64)
    raw_std=np.maximum(raw[fit].std(axis=(0,1),keepdims=True,dtype=np.float64),1e-7)
    feat_mean=features[fit].mean(axis=0,keepdims=True,dtype=np.float64)
    feat_std=np.maximum(features[fit].std(axis=0,keepdims=True,dtype=np.float64),1e-7)
    result={}
    for role in ROLES:
        take=np.isin(s,split['groups'][role])
        result[role]=dict(raw=((raw[take]-raw_mean)/raw_std).astype(np.float32),
                          features=((features[take]-feat_mean)/feat_std).astype(np.float32),
                          y=y[take],subjects=s[take])
        require(np.isfinite(result[role]['raw']).all() and np.isfinite(result[role]['features']).all(),'Nonfinite normalized inputs')
    scalers=dict(raw_mean=raw_mean,raw_std=raw_std,feature_mean=feat_mean,feature_std=feat_std)
    return result,scalers
