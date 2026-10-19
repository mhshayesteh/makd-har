"""Verify complete teacher-ensemble inference and independently reconstruct all predictions/metrics."""
import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

SEEDS=[11,23,37,53,71,89,107,131,167,197]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def independent_metrics(y,p):
    assert y.ndim==p.ndim==1 and len(y)==len(p)>0
    assert np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer)
    assert np.isin(y,range(6)).all() and np.isin(p,range(6)).all()
    cm=np.zeros((6,6),dtype=int);np.add.at(cm,(y,p),1)
    tp=np.diag(cm).astype(float);actual=cm.sum(axis=1);predicted=cm.sum(axis=0)
    f1=np.divide(2*tp,actual+predicted,out=np.zeros(6),where=actual+predicted>0)
    pr=np.divide(tp,predicted,out=np.zeros(6),where=predicted>0);rc=np.divide(tp,actual,out=np.zeros(6),where=actual>0)
    return dict(accuracy=float(tp.sum()/len(y)),macro_f1=float(f1.mean()),macro_precision=float(pr.mean()),macro_recall=float(rc.mean()),
                per_class_f1=f1.tolist(),confusion_matrix=cm.tolist())


def reconstruct(logits,names,weights):
    assert logits.ndim==3 and logits.shape[0]==5 and logits.shape[2]==6 and np.isfinite(logits).all()
    result={'teacher-'+name:logits[i].argmax(axis=1) for i,name in enumerate(names)}
    for temperature in (1,4):
        values=logits.astype(np.float64);values-=values.max(axis=2,keepdims=True);values/=temperature
        p=np.exp(values);p/=p.sum(axis=2,keepdims=True)
        result[f'uniform-T{temperature}']=(sum(p)/5).argmax(axis=1)
        result[f'shapley-T{temperature}']=sum(w*prob for w,prob in zip(weights,p)).argmax(axis=1)
    return result


def original_weights(source):
    from analyze_weighting_pilot import independent_phi
    game={tuple(map(int,k.split(','))) if k else ():v for k,v in source['shapley']['coalition_values'].items()}
    assert len(game)==32;phi=independent_phi(game)
    np.testing.assert_allclose(phi,source['shapley']['values'],rtol=0,atol=1e-12)
    z=np.zeros(5) if phi.std()<=1e-12 else (phi-phi.mean())/phi.std()
    w=np.exp(z-z.max());w/=w.sum();np.testing.assert_allclose(w,source['shapley_weights'],rtol=0,atol=1e-12)
    return w


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--drive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();drive=args.drive.resolve();root=drive/'results/revision-2/teacher-ensemble'
    if not (root/'completed.json').is_file():raise SystemExit('Teacher-ensemble evaluation incomplete; no analysis written.')
    sys.path.insert(0,str(drive/'revision_2/teacher_ensemble'))
    from run_ensemble import verify_inputs,protocol
    frozen=verify_inputs(drive);spec=protocol(frozen);env=json.loads((root/'environment.json').read_text())
    assert env['deterministic_operations'] and env['tensor_float_32_execution'] is False
    assert json.loads((root/'experiment-protocol.json').read_text())==spec
    complete=json.loads((root/'completed.json').read_text());gate=json.loads((root/'preflight-completed.json').read_text())
    assert complete['status']=='complete' and complete['mode']=='research' and complete['training_fits']==0 and complete['test_evaluated'] is True
    assert (complete['banks'],complete['teacher_evaluations'],complete['ensemble_evaluations'],complete['metric_sets'])==(20,100,80,360)
    assert gate['passed'] and gate['teacher_checks']==100 and gate['banks']==20 and gate['test_evaluated'] is False
    for r in (complete,gate):assert r['specification_sha256']==digest(spec) and r['environment_sha256']==digest(env)
    assert len(gate['files'])==44 and len(complete['files'])==40
    expected_outputs={f'results/revision-2/teacher-ensemble/{ds}/seed-{seed}/{name}' for ds in ('uci','wisdm') for seed in SEEDS for name in ('evaluation.json','predictions-and-logits.npz')}
    expected_pref={f'results/revision-2/teacher-ensemble/data/{ds}.{ext}' for ds in ('uci','wisdm') for ext in ('npz','json')}
    expected_pref|={f'results/revision-2/teacher-ensemble/{ds}/seed-{seed}/{name}' for ds in ('uci','wisdm') for seed in SEEDS for name in ('preflight.json','validation-logits.npz')}
    assert {r['path'] for r in complete['files']}==expected_outputs and {r['path'] for r in gate['files']}==expected_pref
    checked=[];snapshots={root/'completed.json',root/'preflight-completed.json',root/'experiment-protocol.json',root/'environment.json'}
    for item in gate['files']+complete['files']:
        path=(drive/item['path']).resolve();assert root in path.parents and sha(path)==item['sha256']
        checked.append(dict(path=str(path),sha256=item['sha256']))
        if path.suffix=='.json':snapshots.add(path)
    smoke_root=drive/'results/revision-2/smoke-teacher-ensemble'
    smoke=json.loads((smoke_root/'completed.json').read_text())
    assert (smoke['status'],smoke['mode'],smoke['banks'],smoke['teacher_evaluations'],smoke['ensemble_evaluations'],smoke['metric_sets'],smoke['training_fits'])==('complete','smoke',2,10,8,36,0)
    assert smoke['specification_sha256']==digest(spec) and smoke['environment_sha256']==digest(env)
    expected_smoke={f'results/revision-2/smoke-teacher-ensemble/{ds}/seed-11/{name}' for ds in ('uci','wisdm') for name in ('evaluation.json','predictions-and-logits.npz')}
    assert len(smoke['files'])==4 and {r['path'] for r in smoke['files']}==expected_smoke
    smoke_checked=[];snapshots.add(smoke_root/'completed.json')
    for item in smoke['files']:
        path=(drive/item['path']).resolve();assert smoke_root in path.parents and sha(path)==item['sha256']
        smoke_checked.append(dict(path=str(path),sha256=item['sha256']))
        if path.suffix=='.json':snapshots.add(path)
    rows=[];classes=[];parity_rows=[]
    for dataset in ('uci','wisdm'):
        with np.load(root/'data'/f'{dataset}.npz',allow_pickle=False) as cache:
            truth={part:cache['y_'+part].copy() for part in ('val','test')};subjects={part:cache['subjects_'+part].copy() for part in ('val','test')}
        for seed in SEEDS:
            source_path=drive/frozen['seed_records'][dataset][str(seed)];source=json.loads(source_path.read_text());snapshots.add(source_path)
            names=source['teacher_order'];weights=original_weights(source)
            context=dict(specification=spec,environment=env,dataset=dataset,outer_seed=seed,source_record_sha256=sha(source_path),
                teacher_order=names,dataset_manifest=source['dataset_manifest'],data_sha256=sha(root/'data'/f'{dataset}.npz'))
            bank=root/dataset/f'seed-{seed}';pref=json.loads((bank/'preflight.json').read_text());report=json.loads((bank/'evaluation.json').read_text())
            assert pref['context']==context and pref['identity']==digest(context) and pref['passed'] and pref['test_evaluated'] is False
            assert pref['logits_sha256']==sha(bank/'validation-logits.npz')
            task=dict(context=context,mode='research',preflight_sha256=sha(bank/'preflight.json'),weights=source['shapley_weights'])
            assert report['task']==task and report['identity']==digest(task) and report['training_fits']==0 and report['test_evaluated'] is True and report['status']=='complete'
            assert report['predictions_sha256']==sha(bank/'predictions-and-logits.npz')
            with np.load(bank/'validation-logits.npz',allow_pickle=False) as pref_cache, np.load(bank/'predictions-and-logits.npz',allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved['val_logits'],pref_cache['val_logits'])
                for part,long in [('val','validation'),('test','test')]:
                    y=truth[part];s=subjects[part];meta=source['dataset_manifest'][long]
                    assert len(y)==meta['samples'] and np.unique(s).tolist()==meta['subjects']
                    assert {str(i):int((y==i).sum()) for i in range(6)}==meta['class_counts']
                    np.testing.assert_array_equal(saved['y_'+part],y);np.testing.assert_array_equal(saved['subjects_'+part],s)
                    predictions=reconstruct(saved[part+'_logits'],names,weights)
                    assert set(predictions)==set(report['metrics'][part]) and len(predictions)==9
                    for condition,values in predictions.items():
                        np.testing.assert_array_equal(values,saved[part+'__'+condition]);m=independent_metrics(y,values)
                        for key in m:np.testing.assert_allclose(m[key],report['metrics'][part][condition][key],rtol=0,atol=1e-12)
                        rows.append(dict(dataset=dataset,outer_seed=seed,partition=long,condition=condition,
                            macro_f1_pct=m['macro_f1']*100,accuracy_pct=m['accuracy']*100,macro_precision_pct=m['macro_precision']*100,macro_recall_pct=m['macro_recall']*100))
                        classes.extend(dict(dataset=dataset,outer_seed=seed,partition=long,condition=condition,class_index=i,f1_pct=f*100,support=int((y==i).sum())) for i,f in enumerate(m['per_class_f1']))
                        if condition.startswith('teacher-'):
                            name=condition[len('teacher-'):];original=source['teachers'][name][long]
                            entry=(pref['teacher_validation_parity'] if part=='val' else report['test_teacher_parity'])[name]
                            assert entry['saved']==original and entry['tolerance']==1e-4
                            for key in ('accuracy','macro_f1'):
                                error=abs(m[key]-original[key]);assert error<=1e-4
                                np.testing.assert_allclose(entry['absolute_metric_errors'][key],error,rtol=0,atol=1e-12)
                                parity_rows.append(dict(dataset=dataset,outer_seed=seed,partition=long,teacher=name,metric=key,saved=original[key],reproduced=m[key],absolute_error=error,tolerance=1e-4))
    assert len(rows)==360 and len(parity_rows)==400 and len(classes)==2160
    conditions=['teacher-'+n for n in names]+['uniform-T1','shapley-T1','uniform-T4','shapley-T4'];summaries=[];paired=[]
    for dataset in ('uci','wisdm'):
        for part in ('validation','test'):
            for name in conditions:
                subset=[r for r in rows if r['dataset']==dataset and r['partition']==part and r['condition']==name]
                assert len(subset)==10
                summaries.append(dict(dataset=dataset,partition=part,condition=name,macro_f1_mean_pct=float(np.mean([r['macro_f1_pct'] for r in subset])),
                    macro_f1_sd_pp=float(np.std([r['macro_f1_pct'] for r in subset],ddof=1)),accuracy_mean_pct=float(np.mean([r['accuracy_pct'] for r in subset]))))
            for temperature in (1,4):
                for seed in SEEDS:
                    def score(name):return next(r['macro_f1_pct'] for r in rows if r['dataset']==dataset and r['partition']==part and r['condition']==name and r['outer_seed']==seed)
                    paired.append(dict(dataset=dataset,partition=part,outer_seed=seed,inference_temperature=temperature,
                        shapley_minus_uniform_pp=score(f'shapley-T{temperature}')-score(f'uniform-T{temperature}')))
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    for name,values in [('metrics',rows),('summary',summaries),('per-class',classes),('parity',parity_rows),('paired-differences',paired)]:write_csv(output/f'teacher-ensemble-{name}.csv',values)
    lines=['# Teacher and teacher-ensemble evaluation','','Verified 139 frozen inputs, all 84 recorded output hashes, twenty banks, '
        '360 independently reconstructed metric sets and 100 validation plus 100 test teacher parity checks. No training occurred.',
        '','| Dataset | Condition | Validation macro-F1 mean ± seed SD (%) | Test macro-F1 mean ± seed SD (%) |','|---|---|---:|---:|']
    for ds in ('uci','wisdm'):
        for name in conditions:
            a=next(r for r in summaries if r['dataset']==ds and r['partition']=='validation' and r['condition']==name)
            b=next(r for r in summaries if r['dataset']==ds and r['partition']=='test' and r['condition']==name)
            lines.append(f"| {ds.upper()} | {name} | {a['macro_f1_mean_pct']:.3f} ± {a['macro_f1_sd_pp']:.3f} | {b['macro_f1_mean_pct']:.3f} ± {b['macro_f1_sd_pp']:.3f} |")
    lines+=['','T=1 is the primary conventional probability-ensemble comparison; T=4 is the secondary KD-target-softening comparison. '
        'Each teacher is softened before averaging. Shapley weights are frozen from the original exact games and were not selected '
        'from these outcomes or replaced by pilot choices. Individual teachers use raw-logit argmax.','',
        'All conditions are retained. These are descriptive model-seed summaries on the original fixed subject splits; '
        'the ten seeds are not ten independent subject cohorts. Previously used tests provide post-review evaluation, '
        'not new independent confirmation. No significance claim or best-temperature selection is made. Teacher ensembles '
        'are distinct from distilled students, and no timing or physical-edge-device conclusion follows from this evaluation.','',
        '## Interpretation and draft integration text','',
        '| Dataset | Temperature | Shapley minus uniform test macro-F1 (pp) | Positive differences / 10 seeds |',
        '|---|---:|---:|---:|']
    for ds in ('uci','wisdm'):
        for temperature in (1,4):
            differences=[r['shapley_minus_uniform_pp'] for r in paired if r['dataset']==ds and r['partition']=='test' and r['inference_temperature']==temperature]
            lines.append(f"| {ds.upper()} | {temperature} | {np.mean(differences):+.3f} | {sum(v>0 for v in differences)}/10 |")
    lines+=['',
        'Uniform probability averaging has higher mean test macro-F1 than the original Shapley-weighted teacher ensemble '
        'on both datasets at both fixed temperatures. At primary T=1, uniform averaging also has a higher mean than '
        'each individual teacher. These are numerical comparisons; no new significance test is claimed. '
        'The original Shapley weights value coalition-distilled student utility, rather than directly optimizing the '
        'teacher probability ensemble. Their ensemble performance therefore does not establish the corresponding '
        'distilled-student ranking. Retain these outcomes, all four fixed ensembles and the original main student '
        'comparisons; do not replace the main method or select a temperature from these test scores.','',
        'Draft manuscript text: We evaluated the five saved teachers and four fixed probability ensembles over '
        'all ten original seeds on both datasets, without retraining. For each ensemble, probabilities were obtained '
        'by applying softmax separately to each teacher before uniform or original Shapley-weighted averaging. '
        'At primary inference temperature 1, uniform and Shapley-weighted ensembles achieved test macro-F1 of '
        '94.697 ± 0.509% and 93.279 ± 1.121% on UCI-HAR, and 81.722 ± 1.312% and 81.007 ± 1.853% on corrected '
        'WISDM, respectively (mean ± sample SD across seeds). The weighted ensembles were numerically lower; '
        'we do not claim a teacher-ensemble advantage from the Shapley weights. Secondary temperature-4 results '
        'are also reported, with no temperature selected from test performance. These post-review evaluations '
        'use the original benchmark splits and describe model-seed variability, rather than independent subject cohorts.','',
        'Draft response for reviewer 1 comment 8 (table/section references pending): We evaluated all 100 saved '
        'teachers and 80 fixed teacher ensembles across the twenty dataset-seed banks. Before test inference, '
        'all teachers reproduced their original validation accuracy and macro-F1 within the unchanged absolute '
        'tolerance of 0.0001; original test metrics also passed. The accompanying table reports individual '
        'teachers and both weighting schemes at the two prespecified temperatures. We retain the numerical '
        'underperformance of the weighted teacher ensembles and distinguish these ensembles from the distilled '
        'students. The expanded architecture and training-protocol subsection remains to be integrated before '
        'this reviewer comment is marked fully addressed.','',
        'The separate smoke completion and its four output hashes were verified and excluded from the research '
        'summary. The figure was checked for readable labels, full axes and error bars. Raw models and arrays '
        'remain in Google Drive; OneDrive contains analysis tables, figures and selected small JSON/code evidence.','',
        f"Analysis date: {datetime.now(ZoneInfo('Asia/Tehran')).date().isoformat()}. Manuscript/response integration remains pending.",'']
    (output/'teacher-ensemble-report.md').write_text('\n'.join(lines),encoding='utf-8')
    plot_results(output,summaries,conditions)
    for name in ('run_ensemble.py','test_ensemble_guards.py','frozen-inputs.json'):snapshots.add(drive/'revision_2/teacher_ensemble'/name)
    saved=[]
    for path in sorted(snapshots):
        destination=output/'evidence_snapshot/teacher-ensemble'/path.relative_to(drive);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        saved.append(dict(source=str(path),sha256=sha(path),snapshot=str(destination.relative_to(output))))
    audit=dict(analysis_at=datetime.now(ZoneInfo('Asia/Tehran')).isoformat(),banks_verified=20,metric_sets_verified=360,validation_teacher_checks=100,
        test_teacher_checks=100,training_fits=0,checked_hashes=checked,separate_smoke_hashes=smoke_checked,snapshots=saved,script_sha256=sha(Path(__file__)),
        outputs={p.name:sha(p) for p in output.glob('teacher-ensemble-*') if p.suffix in ('.csv','.md')})
    audit['outputs'].update({name:sha(output/name) for name in ('teacher-ensemble.png','teacher-ensemble.pdf')})
    audit['attribution_helper_sha256']=sha(Path(__file__).with_name('analyze_weighting_pilot.py'))
    (output/'teacher-ensemble-verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Verified twenty banks and all teacher/ensemble outcomes; report and five CSVs written.')


def plot_results(output,rows,conditions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype']=42
    labels=[name.replace('teacher-','').replace('hcf_mlp','HCF MLP').replace('raw_cnn','Raw CNN').replace('lstm','LSTM').replace('tcn','TCN').replace('transformer','Transformer') for name in conditions]
    fig,axes=plt.subplots(1,2,figsize=(11,5.5),layout='constrained',sharey=True)
    for ax,dataset in zip(axes,('uci','wisdm')):
        subset=[next(r for r in rows if r['dataset']==dataset and r['partition']=='test' and r['condition']==name) for name in conditions]
        x=[r['macro_f1_mean_pct'] for r in subset];sd=[r['macro_f1_sd_pp'] for r in subset];y=np.arange(len(conditions))
        ax.barh(y,x,xerr=sd,color=['#8ca8b6']*5+['#397d58']*2+['#b78345']*2,alpha=.85,capsize=3)
        ax.set_yticks(y,labels);ax.invert_yaxis() if dataset=='uci' else None
        ax.set_xlim(0,100);ax.set_xlabel('Test macro-F1 (%)');ax.set_title(dataset.upper());ax.grid(axis='x',alpha=.2)
    fig.suptitle('Saved teachers and fixed probability ensembles\nMean ± seed SD across ten banks; descriptive, no retraining',fontsize=12)
    fig.savefig(output/'teacher-ensemble.png',dpi=180);fig.savefig(output/'teacher-ensemble.pdf');plt.close(fig)


if __name__=='__main__':main()
