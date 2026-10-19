"""Descriptive audit of the original WISDM gap; no fitting, inference, or test selection."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np

SEEDS=[11,23,37,53,71,89,107,131,167,197]
CLASSES=['Walking','Jogging','Sitting','Standing','Upstairs','Downstairs']
MAIN=['proposed','uniform_teachers','supervised_student','cnn','lstm','slfn']
LABELS={'proposed':'Proposed student','uniform_teachers':'Uniform KD student','supervised_student':'Supervised student','cnn':'Original CNN baseline','lstm':'Original LSTM baseline','slfn':'Original SLFN baseline','svm-selected':'Selected RBF SVC','svm-linear-alternative':'Auxiliary linear SVC'}

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write_csv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def cm_metrics(cm):
    cm=np.asarray(cm,dtype=float);tp=np.diag(cm);actual=cm.sum(1);pred=cm.sum(0)
    f=np.divide(2*tp,actual+pred,out=np.zeros(6),where=actual+pred>0)
    recall=np.divide(tp,actual,out=np.zeros(6),where=actual>0)
    precision=np.divide(tp,pred,out=np.zeros(6),where=pred>0)
    observed=actual>0
    return dict(accuracy=float(tp.sum()/cm.sum()),macro_f1=float(f.mean()),balanced_accuracy=float(recall.mean()),
        observed_macro_f1=float(f[observed].mean()),observed_balanced_accuracy=float(recall[observed].mean()),
        per_class_f1=f,recall=recall,precision=precision,support=actual,classes_present=int(observed.sum()))

def confusion(y,p):
    assert y.ndim==p.ndim==1 and len(y)==len(p)>0 and np.isin(y,range(6)).all() and np.isin(p,range(6)).all()
    cm=np.zeros((6,6),dtype=np.int64);np.add.at(cm,(y,p),1);return cm

def compare_metrics(cm,stored):
    m=cm_metrics(cm)
    for key in ('accuracy','macro_f1'):np.testing.assert_allclose(m[key],stored[key],rtol=0,atol=1e-12)
    np.testing.assert_allclose(m['per_class_f1'],stored['per_class_f1'],rtol=0,atol=1e-12)
    if 'confusion_matrix' in stored:np.testing.assert_array_equal(cm,stored['confusion_matrix'])

def standardize_pair(val,test):
    a=cm_metrics(val);b=cm_metrics(test);pv=val.sum(1)/val.sum();pt=test.sum(1)/test.sum()
    assert (val.sum(1)>0).all() and (test.sum(1)>0).all()
    rv=val/val.sum(1,keepdims=True);rt=test/test.sum(1,keepdims=True)
    av=cm_metrics(rv/6);at=cm_metrics(rt/6)
    composition=float(np.sum((pv-pt)*(a['recall']+b['recall'])/2))
    within=float(np.sum((a['recall']-b['recall'])*(pv+pt)/2))
    np.testing.assert_allclose(composition+within,a['accuracy']-b['accuracy'],rtol=0,atol=1e-12)
    return dict(raw_accuracy_gap_pp=(a['accuracy']-b['accuracy'])*100,class_composition_term_pp=composition*100,
        within_class_recall_term_pp=within*100,equal_prior_validation_accuracy_pct=av['accuracy']*100,
        equal_prior_test_accuracy_pct=at['accuracy']*100,equal_prior_accuracy_gap_pp=(av['accuracy']-at['accuracy'])*100,
        equal_prior_validation_macro_f1_pct=av['macro_f1']*100,equal_prior_test_macro_f1_pct=at['macro_f1']*100,
        equal_prior_macro_f1_gap_pp=(av['macro_f1']-at['macro_f1'])*100)

def independent_split(y,subjects,config):
    def holdout(indices,fraction,seed):
        people=np.unique(subjects[indices]);count=max(1,round(len(people)*fraction))
        for attempt in range(1000):
            selected=np.random.default_rng(seed+attempt).permutation(people)[:count]
            mask=np.isin(subjects[indices],selected);a=indices[~mask];b=indices[mask]
            if len(np.unique(y[a]))==len(np.unique(y[b]))==6:return a,b
        raise AssertionError('No original class-complete holdout')
    all_indices=np.arange(len(y));development,test=holdout(all_indices,config['wisdm_test_fraction'],config['split_seed'])
    train,val=holdout(development,config['validation_fraction']/(1-config['wisdm_test_fraction']),config['split_seed']+10000)
    return dict(train=train,validation=val,test=test)

def parser_labels(drive,source):
    sys.path.insert(0,str(drive/'revision_2/pipeline_snapshot'))
    from makd_har.data import _iter_wisdm_segments
    audit={};ys=[];ss=[];segments=0
    for subject,label,segment in _iter_wisdm_segments(drive/'resources/WISDM_ar_v1.1_raw.txt',audit):
        n=max(0,(len(segment)-128)//64+1);ys.extend([label]*n);ss.extend([subject]*n);segments+=1
    assert audit==source['dataset_manifest']['preprocessing']['parse_audit'] and segments==audit['segments']
    y=np.array(ys,dtype=np.int64);subjects=np.array(ss,dtype=np.int64);parts=independent_split(y,subjects,source['config'])
    for name,indices in parts.items():
        actual=source['dataset_manifest'][name]
        assert len(indices)==actual['samples'] and np.unique(subjects[indices]).tolist()==actual['subjects']
        assert {str(i):int((y[indices]==i).sum()) for i in range(6)}==actual['class_counts']
    return y,subjects,parts,audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--drive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();drive=args.drive.resolve();out=args.output.resolve();checked={}
    def verify(path,expected=None):
        path=path.resolve();h=sha(path)
        if expected is not None:assert h==expected,f'Changed input: {path}'
        if str(path) in checked:assert checked[str(path)]==h
        checked[str(path)]=h;return h
    # Revalidate the frozen sources and all three completed phases of each contributing study.
    rt=drive/'results/revision-2/runtime-benchmark/session-20261019-013326-b29922da';ens=drive/'results/revision-2/teacher-ensemble'
    for manifest in [drive/'revision_2/runtime_benchmark/frozen-inputs.json',drive/'revision_2/teacher_ensemble/frozen-inputs.json']:
        frozen=read(manifest);verify(manifest)
        for name,h in frozen['files'].items():verify(drive/name,h)
    runtime_frozen=read(drive/'revision_2/runtime_benchmark/frozen-inputs.json')
    ensemble_frozen=read(drive/'revision_2/teacher_ensemble/frozen-inputs.json')
    assert runtime_frozen['seed_records']['wisdm']==ensemble_frozen['seed_records']['wisdm']
    for root,names,path_base in [(rt,['preflight-completed.json','smoke-completed.json','completed.json'],rt),
            (ens,['preflight-completed.json','completed.json'],drive),
            (drive/'results/revision-2/smoke-teacher-ensemble',['completed.json'],drive)]:
        for name in names:
            marker=read(root/name);assert marker.get('status')=='complete' or marker.get('passed') is True;verify(root/name)
            for item in marker['files']:verify(path_base/item['path'],item['sha256'])
    completed=read(rt/'completed.json');assert completed['conditions']==65 and completed['timing_records']==195
    assert read(ens/'completed.json')['banks']==20
    run=(rt/completed['results_file']).parent
    records={seed:read(drive/runtime_frozen['seed_records']['wisdm'][str(seed)]) for seed in SEEDS};source=records[11]
    assert all(r['dataset_manifest']==source['dataset_manifest'] and r['selection_provenance']==source['selection_provenance'] for r in records.values())
    y_all,s_all,parts,parse_audit=parser_labels(drive,source)
    with np.load(rt/'wisdm/data.npz',allow_pickle=False) as z:data={k:z[k].copy() for k in z.files}
    with np.load(ens/'data/wisdm.npz',allow_pickle=False) as z:
        for k in data:np.testing.assert_array_equal(data[k],z[k])
    for part,short in [('validation','val'),('test','test')]:
        np.testing.assert_array_equal(y_all[parts[part]],data['y_'+short]);np.testing.assert_array_equal(s_all[parts[part]],data['subjects_'+short])
    sets={name:set(s_all[idx]) for name,idx in parts.items()}
    assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    distributions=[];subject_classes=[]
    for part,indices in parts.items():
        for c in range(6):distributions.append(dict(partition=part,class_index=c,class_name=CLASSES[c],windows=int((y_all[indices]==c).sum()),share_pct=float((y_all[indices]==c).mean()*100),subjects=len(sets[part])))
        for subject in sorted(sets[part]):
            labels=y_all[indices][s_all[indices]==subject]
            for c in range(6):subject_classes.append(dict(partition=part,subject=int(subject),class_index=c,class_name=CLASSES[c],windows=int((labels==c).sum()),subject_windows=len(labels),classes_present=len(np.unique(labels))))
    models=[];class_rows=[];subject_rows=[];confusions=[];standardized=[];influence=[];predictions={}
    def add_summary(name,seed,val,test,origin):
        models.append(dict(condition=name,seed=seed,evidence=origin,validation_macro_f1_pct=val['macro_f1']*100,test_macro_f1_pct=test['macro_f1']*100,gap_macro_f1_pp=(val['macro_f1']-test['macro_f1'])*100,
            validation_accuracy_pct=val['accuracy']*100,test_accuracy_pct=test['accuracy']*100,gap_accuracy_pp=(val['accuracy']-test['accuracy'])*100,
            validation_balanced_accuracy_pct=val.get('balanced_accuracy',val.get('macro_recall'))*100,test_balanced_accuracy_pct=test.get('balanced_accuracy',test.get('macro_recall'))*100))
        for c in range(6):class_rows.append(dict(condition=name,seed=seed,evidence=origin,class_index=c,class_name=CLASSES[c],validation_f1_pct=float(val['per_class_f1'][c]*100),test_f1_pct=float(test['per_class_f1'][c]*100),gap_f1_pp=float((val['per_class_f1'][c]-test['per_class_f1'][c])*100)))
    def add_predictions(name,seed,val,test,expected=None):
        values={'val':val,'test':test};cms={part:confusion(data['y_'+part],pred) for part,pred in values.items()}
        for part,long in [('val','validation'),('test','test')]:
            if expected is not None:compare_metrics(cms[part],expected[long])
        measured={part:cm_metrics(cm) for part,cm in cms.items()};add_summary(name,seed,measured['val'],measured['test'],'reconstructed from saved predictions')
        standardized.append(dict(condition=name,seed=seed,**standardize_pair(cms['val'],cms['test'])))
        predictions[(name,seed)]=values
        for part,long in [('val','validation'),('test','test')]:
            cm=cms[part]
            for c in range(6):
                for j in range(6):confusions.append(dict(condition=name,seed=seed,partition=long,true_class=c,true_name=CLASSES[c],predicted_class=j,predicted_name=CLASSES[j],count=int(cm[c,j]),row_pct=float(cm[c,j]/cm[c].sum()*100)))
            total_errors=int((values[part]!=data['y_'+part]).sum())
            for subject in np.unique(data['subjects_'+part]):
                mask=data['subjects_'+part]==subject;subcm=confusion(data['y_'+part][mask],values[part][mask]);m=cm_metrics(subcm);errors=int(mask.sum()-np.trace(subcm))
                subject_rows.append(dict(condition=name,seed=seed,partition=long,subject=int(subject),windows=int(mask.sum()),classes_present=m['classes_present'],accuracy_pct=m['accuracy']*100,observed_class_macro_f1_pct=m['observed_macro_f1']*100,observed_class_balanced_accuracy_pct=m['observed_balanced_accuracy']*100,errors=errors,error_share_pct=100*errors/total_errors if total_errors else 0))
    for seed,r in records.items():
        for name in MAIN:
            saved=r['methods'][name]
            for part in ('validation','test'):
                assert len(saved[part]['per_class_f1'])==6
                np.testing.assert_allclose(np.mean(saved[part]['per_class_f1']),saved[part]['macro_f1'],rtol=0,atol=1e-12)
            if name!='proposed':add_summary(name,seed,saved['validation'],saved['test'],'saved original aggregate metrics; no participant predictions')
        with np.load(rt/f'wisdm/seed-{seed}/validation.npz',allow_pickle=False) as v,np.load(run/f'wisdm/seed-{seed}/test.npz',allow_pickle=False) as t:
            add_predictions('proposed',seed,v['keras_logits'].argmax(1),t['keras_logits'].argmax(1))
            for part,pred in [('validation',v['keras_logits'].argmax(1)),('test',t['keras_logits'].argmax(1))]:
                current=cm_metrics(confusion(data['y_val' if part=='validation' else 'y_test'],pred))
                for k in ('accuracy','macro_f1'):assert abs(current[k]-r['methods']['proposed'][part][k])<=1e-4
        with np.load(ens/f'wisdm/seed-{seed}/predictions-and-logits.npz',allow_pickle=False) as z:
            report=read(ens/f'wisdm/seed-{seed}/evaluation.json')
            for part in ('val','test'):
                np.testing.assert_array_equal(z['y_'+part],data['y_'+part]);np.testing.assert_array_equal(z['subjects_'+part],data['subjects_'+part])
                logits=z[part+'_logits'];expected={'teacher-'+name:logits[i].argmax(1) for i,name in enumerate(r['teacher_order'])}
                for temperature in (1,4):
                    a=logits.astype(float)/temperature;a-=a.max(2,keepdims=True);p=np.exp(a);p/=p.sum(2,keepdims=True)
                    expected[f'uniform-T{temperature}']=p.mean(0).argmax(1);expected[f'shapley-T{temperature}']=np.einsum('i,ijk->jk',r['shapley_weights'],p).argmax(1)
                for key,pred in expected.items():np.testing.assert_array_equal(pred,z[part+'__'+key])
            for name in report['metrics']['val']:
                add_predictions(name,seed,z['val__'+name],z['test__'+name],dict(validation=report['metrics']['val'][name],test=report['metrics']['test'][name]))
    with np.load(rt/'wisdm/svm-validation-predictions.npz',allow_pickle=False) as v,np.load(run/'wisdm-svm-test.npz',allow_pickle=False) as t:
        for name in ('selected','linear-alternative'):add_predictions('svm-'+name,None,v[name],t['svm-'+name])
    assert len(models)==152 and len(predictions)==102 and len(class_rows)==912 and len(subject_rows)==1428 and len(confusions)==7344
    # Report all subject omission influences without changing the benchmark or fitting a model.
    for seed in SEEDS:
        y=data['y_test'];pred=predictions[('proposed',seed)]['test'];full=cm_metrics(confusion(y,pred))
        for subject in np.unique(data['subjects_test']):
            mask=data['subjects_test']!=subject;m=cm_metrics(confusion(y[mask],pred[mask]))
            influence.append(dict(seed=seed,omitted_subject=int(subject),retained_windows=int(mask.sum()),classes_present=m['classes_present'],full_test_macro_f1_pct=full['macro_f1']*100,omission_macro_f1_pct=m['macro_f1']*100,delta_macro_f1_pp=(m['macro_f1']-full['macro_f1'])*100,full_test_accuracy_pct=full['accuracy']*100,omission_accuracy_pct=m['accuracy']*100,description='post-hoc influence diagnostic only; no subject removed from primary result'))
    summaries=[]
    for name in dict.fromkeys(v['condition'] for v in models):
        rows=[v for v in models if v['condition']==name];n=len(rows);assert n==(1 if name.startswith('svm-') else 10)
        result=dict(condition=name,models=n,evidence=rows[0]['evidence'])
        for key in ('validation_macro_f1_pct','test_macro_f1_pct','gap_macro_f1_pp','validation_accuracy_pct','test_accuracy_pct','validation_balanced_accuracy_pct','test_balanced_accuracy_pct'):
            values=[v[key] for v in rows];result[key+'_mean']=float(np.mean(values));result[key+'_seed_sd']=float(np.std(values,ddof=1)) if n>1 else None
        summaries.append(result)
    subject_summaries=[]
    for name in dict.fromkeys(v['condition'] for v in subject_rows):
        for part in ('validation','test'):
            for subject in sorted(sets[part]):
                rows=[v for v in subject_rows if v['condition']==name and v['partition']==part and v['subject']==subject]
                result=dict(condition=name,partition=part,subject=int(subject),models=len(rows),windows=rows[0]['windows'],classes_present=rows[0]['classes_present'])
                for k in ('accuracy_pct','observed_class_macro_f1_pct','observed_class_balanced_accuracy_pct','errors','error_share_pct'):
                    values=[v[k] for v in rows];result[k+'_mean']=float(np.mean(values));result[k+'_seed_sd']=float(np.std(values,ddof=1)) if len(rows)>1 else None
                subject_summaries.append(result)
    assert len(summaries)==17 and len(subject_summaries)==168 and len(influence)==70
    pilot_rows,pilot_evidence=audit_selection(drive,source,records,verify)
    out.mkdir(parents=True,exist_ok=True)
    for name,rows in [('distributions',distributions),('subject-class',subject_classes),('models',models),('summary',summaries),('per-class',class_rows),('subjects',subject_rows),('subject-summary',subject_summaries),('confusions',confusions),('standardized',standardized),('influence',influence),('pilot-selection',pilot_rows)]:write_csv(out/f'wisdm-gap-{name}.csv',rows)
    selection_path=out/'evidence_snapshot/wisdm-gap/selection-provenance.json';selection_path.parent.mkdir(parents=True,exist_ok=True);selection_path.write_text(json.dumps(pilot_evidence,indent=2),encoding='utf-8')
    make_report(out,source,summaries,subject_summaries,class_rows,standardized,influence,distributions,pilot_evidence,parse_audit)
    plot(out,distributions,summaries,subject_summaries,predictions,data)
    for name,seed in [('proposed',11)]:assert (name,seed) in predictions
    outputs={p.name:sha(p) for p in out.glob('wisdm-gap-*') if p.suffix in ('.csv','.md','.png','.pdf')}
    audit=dict(analysis_at=datetime.now(ZoneInfo('Asia/Tehran')).isoformat(),new_model_fits=0,new_inference_runs=0,all_window_labels_regenerated=len(y_all),subjects=36,
        prediction_pairs_reconstructed=102,validation_test_metric_sets_reconstructed=204,model_pairs_including_record_only=152,
        subject_metric_rows=len(subject_rows),all_test_subject_omission_rows=len(influence),source_files_checked=len(checked),checked_hashes=checked,
        original_pilot_summary_sha256=pilot_evidence['summary_sha256'],original_approval_sha256=pilot_evidence['approval_sha256'],
        outputs=outputs,analyzer_sha256=sha(Path(__file__)),selection_evidence_sha256=sha(selection_path),
        limitations='Descriptive, already-inspected fixed split; original uniform/supervised subject predictions unavailable; no causal attribution of selection bias or subject effects; no primary participant exclusion.')
    (out/'wisdm-gap-verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    shutil.copy2(Path(__file__),out/'evidence_snapshot/wisdm-gap/analyze_wisdm_gap.py')
    print(f'Verified {len(checked)} source/artifact hashes; regenerated all {len(y_all)} window labels; reconstructed 204 metric sets. Eleven CSVs, report and figures saved.')

def audit_selection(drive,source,records,verify):
    archive=drive/'results/wisdm/weighting-pilot-v2-results.zip';archive_hash=verify(archive);root='weighting-pilot-v2/'
    with zipfile.ZipFile(archive) as z:
        sb=z.read(root+'pilot-summary.json');ab=z.read(root+'production-approved.json');summary=json.loads(sb);approval=json.loads(ab)
        assert hashlib.sha256(sb).hexdigest()==source['selection_provenance']['record']['pilot_summary_sha256']
        assert hashlib.sha256(ab).hexdigest()==source['selection_provenance']['record_sha256'] and approval==source['selection_provenance']['record']
        assert json.loads(z.read(root+'dataset-manifest.json'))==source['dataset_manifest']
        runs={seed:json.loads(z.read(root+f'seed-{seed}.json')) for seed in (11,23,37)}
        assert summary['test_metrics_computed'] is False and approval['test_metrics_used_for_selection'] is False
        assert all(r['test_metrics_computed'] is False and r['model_seed']==s for s,r in runs.items())
        rows=[]
        for row in summary['rows']:
            values=[runs[s]['candidates'][row['candidate']]['validation_macro_f1'] for s in (11,23,37)]
            np.testing.assert_allclose(np.mean(values),row['validation_macro_f1_mean'],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.std(values,ddof=1),row['validation_macro_f1_sd'],rtol=0,atol=1e-12)
            rows.append(dict(candidate=row['candidate'],runs=3,validation_macro_f1_mean_pct=float(np.mean(values)*100),validation_macro_f1_seed_sd_pp=float(np.std(values,ddof=1)*100),selected=row['candidate']==summary['selected_candidate'],seed11_pct=values[0]*100,seed23_pct=values[1]*100,seed37_pct=values[2]*100))
        best=max(summary['rows'],key=lambda r:r['validation_macro_f1_mean'])['candidate'];assert best==summary['selected_candidate']==approval['selected_candidate']
        differences={str(s):runs[s]['candidates'][best]['validation_macro_f1']-records[s]['methods']['proposed']['validation']['macro_f1'] for s in runs}
    evidence=dict(archive=str(archive),archive_sha256=archive_hash,summary=summary,approval=approval,summary_sha256=hashlib.sha256(sb).hexdigest(),approval_sha256=hashlib.sha256(ab).hexdigest(),selected_pilot_minus_production_validation=differences,scope='Original pilot validation reuse; distinct from revision-2 four-role training-only pilot. No test selection recorded; selected validation score is not independent confirmation.')
    return rows,evidence

def make_report(out,source,summaries,subjects,classes,standardized,influence,distribution,pilot,parse_audit):
    def get(name):return next(v for v in summaries if v['condition']==name)
    def mean_std(name,field):
        rows=[v[field] for v in standardized if v['condition']==name];return float(np.mean(rows))
    proposed=get('proposed');uniform=get('uniform_teachers');supervised=get('supervised_student')
    problem=next(v for v in subjects if v['condition']=='proposed' and v['partition']=='test' and v['subject']==30)
    priors={part:np.array([v['share_pct']/100 for v in distribution if v['partition']==part]) for part in ('train','validation','test')}
    lines=['# Original WISDM validation–test gap audit','',
        'Post-review descriptive analysis of corrected production-v2. No model was fitted and no new model '
        'inference was run. Original results, all participants and all test outcomes are retained. '
        'The completed teacher-ensemble and CPU studies supply saved predictions; original method records '
        'supply the additional baseline summaries. All contributing frozen inputs and recorded outputs were '
        'hash-checked before analysis.','',
        '## Main findings','',
        f"Across ten original seeds, proposed-student validation/test macro-F1 is {proposed['validation_macro_f1_pct_mean']:.3f}% / "
        f"{proposed['test_macro_f1_pct_mean']:.3f}%, a {proposed['gap_macro_f1_pp_mean']:.3f} pp gap. "
        f"Uniform KD has a {uniform['gap_macro_f1_pp_mean']:.3f} pp gap and supervised learning a "
        f"{supervised['gap_macro_f1_pp_mean']:.3f} pp gap. The phenomenon is not confined to the proposed weighting rule. "
        f"Proposed minus uniform is {proposed['validation_macro_f1_pct_mean']-uniform['validation_macro_f1_pct_mean']:+.3f} pp on validation "
        f"and {proposed['test_macro_f1_pct_mean']-uniform['test_macro_f1_pct_mean']:+.3f} pp on test. Preserve this WISDM test loss; "
        'neither the gap audit nor the new pilot replaces that main-table outcome.','',
        '| Original method | Models | Validation macro-F1 mean ± seed SD (%) | Test macro-F1 mean ± seed SD (%) | Paired gap mean ± seed SD (pp) |',
        '|---|---:|---:|---:|---:|']
    for name in MAIN:
        r=get(name);lines.append(f"| {LABELS[name]} | 10 | {r['validation_macro_f1_pct_mean']:.3f} ± {r['validation_macro_f1_pct_seed_sd']:.3f} | {r['test_macro_f1_pct_mean']:.3f} ± {r['test_macro_f1_pct_seed_sd']:.3f} | {r['gap_macro_f1_pp_mean']:.3f} ± {r['gap_macro_f1_pp_seed_sd']:.3f} |")
    lines+=['','Original uniform-KD, supervised, CNN, LSTM and SLFN summaries/per-class F1 come from saved '
        'aggregate records. Their original participant-level predictions were not available in the completed '
        'evaluations, so no participant or confusion analysis is invented for them. The proposed student, '
        'five teachers, four ensembles and two SVCs have prediction-level audits. Later teacher-removal controls '
        'are not substituted for original baseline students. The full tables retain all 17 conditions.','',
        '## Original split and class composition','',
        'All 16,529 window labels and subject IDs were regenerated from the original hashed raw file using '
        'the frozen records-v2 parser, 128-sample windows and stride 64. The original deterministic '
        'subject split, parser counts, class counts and exact cached validation/test label order match. '
        'Subject sets are disjoint; this check does not remove validation-selection reuse.','']
    for part in ('train','validation','test'):
        meta=source['dataset_manifest'][part];lines.append(f"- {part.capitalize()}: {len(meta['subjects'])} subjects, {meta['samples']:,} windows; IDs {', '.join(map(str,meta['subjects']))}.")
    lines+=['','| Class | Train windows / share | Validation windows / share | Test windows / share |','|---|---:|---:|---:|']
    for c,name in enumerate(CLASSES):
        fields=[]
        for part in ('train','validation','test'):
            r=next(v for v in distribution if v['partition']==part and v['class_index']==c);fields.append(f"{r['windows']} / {r['share_pct']:.2f}%")
        lines.append(f"| {name} | {' | '.join(fields)} |")
    lines+=['',f"Validation/test class-prior total-variation distance is {0.5*np.abs(priors['validation']-priors['test']).sum():.4f}. "
        'For example, sitting has 354 validation windows but only 88 test windows; the held-out subject mixtures '
        'differ. All six classes remain present in each full partition. These distributions characterize the '
        'split; they do not establish the causal source of the model gap. Overlapping windows are not '
        'independent subject observations.','',
        '## Participant and error concentration','',
        f"Subject 30 accounts for {problem['windows']} / 2,964 test windows "
        f"({problem['windows']/2964*100:.2f}%) and has proposed-student accuracy "
        f"{problem['accuracy_pct_mean']:.2f} ± {problem['accuracy_pct_seed_sd']:.2f}% over ten models. "
        f"Its mean share of all test errors is {problem['error_share_pct_mean']:.2f}%. "
        'This observation motivated a complete participant audit, not an exclusion rule. The table below '
        'retains every validation and test participant. Participant class coverage varies, so accuracy is '
        'shown here; CSV participant macro-F1/balanced accuracy explicitly average only true classes '
        'present in that participant and are not full-split six-class metrics.','',
        '| Split | Subject | Windows | Classes present | Proposed accuracy mean ± seed SD (%) | Selected RBF SVC accuracy (%) |',
        '|---|---:|---:|---:|---:|---:|']
    for part in ('validation','test'):
        for row in [v for v in subjects if v['condition']=='proposed' and v['partition']==part]:
            svm=next(v for v in subjects if v['condition']=='svm-selected' and v['partition']==part and v['subject']==row['subject'])
            lines.append(f"| {part} | {row['subject']} | {row['windows']} | {row['classes_present']} | {row['accuracy_pct_mean']:.2f} ± {row['accuracy_pct_seed_sd']:.2f} | {svm['accuracy_pct_mean']:.2f} |")
    lines+=['','All seven test-subject omission influences are reported below. Each is a descriptive '
        'recalculation on an already-used test subset, with no model refitting. These are not new benchmark '
        'scores, proposed exclusions or independent test evaluations. The primary result keeps all seven '
        'test subjects. If an omission loses a class, its six-class macro-F1 is flagged as noncomparable.','',
        '| Omitted subject (diagnostic only) | Mean change in test macro-F1 (pp) | Minimum remaining class count |','|---:|---:|---:|']
    for subject in sorted({v['omitted_subject'] for v in influence}):
        rows=[v for v in influence if v['omitted_subject']==subject];lines.append(f"| {subject} | {np.mean([v['delta_macro_f1_pp'] for v in rows]):+.3f} | {min(v['classes_present'] for v in rows)} |")
    lines+=['','## Class errors and descriptive standardization','',
        '| Class | Proposed validation F1 mean (%) | Proposed test F1 mean (%) | Validation minus test (pp) |','|---|---:|---:|---:|']
    for c,name in enumerate(CLASSES):
        rows=[v for v in classes if v['condition']=='proposed' and v['class_index']==c]
        lines.append(f"| {name} | {np.mean([v['validation_f1_pct'] for v in rows]):.3f} | {np.mean([v['test_f1_pct'] for v in rows]):.3f} | {np.mean([v['gap_f1_pp'] for v in rows]):+.3f} |")
    lines+=['','Mean row-normalized confusion matrices show the corresponding errors without changing '
        'class definitions or selecting outcomes. Each row uses the actual split support and each model is '
        'weighted equally. Ten seeds describe model variability on the same held-out people.','',
        f"After row-normalizing each confusion matrix and applying the same equal true-class prior to both splits, "
        f"the proposed validation/test macro-F1 is {mean_std('proposed','equal_prior_validation_macro_f1_pct'):.3f}% / "
        f"{mean_std('proposed','equal_prior_test_macro_f1_pct'):.3f}%, leaving a "
        f"{mean_std('proposed','equal_prior_macro_f1_gap_pp'):.3f} pp descriptive gap. "
        'This standardization is a diagnostic, not a replacement metric or causal counterfactual.','',
        f"The raw proposed accuracy gap is {mean_std('proposed','raw_accuracy_gap_pp'):.3f} pp. Its symmetric "
        f"algebraic decomposition yields a class-composition term of {mean_std('proposed','class_composition_term_pp'):+.3f} pp "
        f"and a within-class recall term of {mean_std('proposed','within_class_recall_term_pp'):+.3f} pp, which sum exactly "
        'to the gap. This partition is arithmetic; the within-class term can contain subject differences, '
        'selection optimism, model error and other unseparated factors. It is not an estimate of any one cause.','',
        '## Verified historical selection and interpretation','',
        'The original pilot summary and production approval files were read directly from the archived '
        'weighting-pilot-v2 results. Their exact byte hashes match the production provenance. The pilot and '
        'production dataset manifests match. All five candidate means/SDs were independently reconstructed '
        'from seeds 11, 23 and 37. The selection record states that test metrics were not used and the pilot '
        'records contain no computed test metrics.','',
        '| Original pilot candidate | Mean validation macro-F1 (%) | Seed SD (pp) | Selected |','|---|---:|---:|---|']
    for row in pilot['summary']['rows']:lines.append(f"| {row['candidate']} | {row['validation_macro_f1_mean']*100:.3f} | {row['validation_macro_f1_sd']*100:.3f} | {'yes' if row['candidate']==pilot['summary']['selected_candidate'] else 'no'} |")
    lines+=['','The original validation subjects were reused for checkpoint stopping, coalition valuation '
        'and candidate comparison. Withholding test metrics from that choice does not make the selected '
        'validation gain independent evidence. The original three-seed pilot and ten-seed production '
        'summary also differ in their seed sets; their difference is not an estimate of optimism bias. '
        'The new revision-2 four-role pilot separates development roles using original training subjects '
        'and fresh teachers. It does not retrospectively correct the historical selection or explain '
        'away this original gap.','',
        'These results support a narrower discussion: the split is subject-disjoint, class composition '
        'differs, errors are heterogeneous across people/classes, and validation was reused for selection. '
        'They do not isolate how much of the gap was caused by subject/domain differences versus selection '
        'optimism. Do not diagnose label errors, sensor orientation effects or faulty participants without '
        'additional evidence; do not delete subject 30, retune weights on the test set or replace only a '
        'favorable main-table row. Keep the original WISDM loss to uniform KD and the previously reported '
        'nonsignificant weighting comparisons. No additional major training is required for these narrower claims.','',
        '## Draft manuscript text','',
        f"On corrected WISDM, mean validation/test macro-F1 over ten seeds was {proposed['validation_macro_f1_pct_mean']:.3f}%/"
        f"{proposed['test_macro_f1_pct_mean']:.3f}% for the proposed student, {uniform['validation_macro_f1_pct_mean']:.3f}%/"
        f"{uniform['test_macro_f1_pct_mean']:.3f}% for uniform KD, and {supervised['validation_macro_f1_pct_mean']:.3f}%/"
        f"{supervised['test_macro_f1_pct_mean']:.3f}% for the supervised student. "
        'An audit of the original subject-disjoint split confirmed exact class/subject counts and label order. '
        'The validation and test populations differ in class composition and participant-specific error. '
        f"In particular, test subject 30 had {problem['accuracy_pct_mean']:.2f}% mean proposed-student accuracy, "
        'but all subjects are retained in the reported result. A common-class-prior diagnostic leaves a '
        'substantial gap. We also acknowledge reuse of the original validation set for checkpoint selection, '
        'coalition valuation and weighting-candidate choice. These descriptive checks do not identify the '
        'relative causal contributions of subject differences and selection optimism. The original WISDM '
        'test loss to uniform KD is retained, and the additional training-only pilot is reported separately.','',
        '## Draft response to reviewer 1 comment 5','',
        'We added a descriptive split/error audit of the original corrected WISDM experiment. The audit '
        'regenerates all window labels and subject membership from the frozen raw-data parser, verifies the '
        'recorded split, and reports all-condition validation/test gaps, class distributions, per-class '
        'errors and every held-out participant. We include equal-class-prior and complete subject-influence '
        'diagnostics without excluding any participant or changing the main benchmark. We explicitly '
        'acknowledge the original validation reuse for stopping, coalition valuation and candidate selection. '
        'The separate revision-2 training-only pilot addresses development-role separation prospectively, '
        'rather than removing the historical limitation. We retain all original test outcomes and avoid '
        'attributing the gap to a single unisolated cause. Final section/table/page references and actual '
        'manuscript/response integration remain pending.','',
        f"Parser provenance: {parse_audit['accepted_records']:,} accepted sensor records and {parse_audit['segments']} segments; "
        'the recorded duplicate/invalid-clock handling is preserved. Parser-event counts alone do not explain '
        'the model gap.','',f"Analysis date: {datetime.now(ZoneInfo('Asia/Tehran')).date().isoformat()}. No new runs are needed for this audit.",'']
    (out/'wisdm-gap-report.md').write_text('\n'.join(lines),encoding='utf-8')

def plot(out,distribution,summaries,subjects,predictions,data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype']=42
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    ax=axes[0,0];x=np.arange(6)
    for i,(part,color) in enumerate([('train','#9faab6'),('validation','#397d58'),('test','#b78345')]):
        ax.bar(x+(i-1)*.25,[v['share_pct'] for v in distribution if v['partition']==part],width=.25,label=part,color=color)
    ax.set_xticks(x,CLASSES,rotation=25,ha='right');ax.set_ylabel('Window share (%)');ax.set_title('Original class composition');ax.legend(fontsize=9)
    ax=axes[0,1]
    for i,name in enumerate(['proposed','uniform_teachers','supervised_student']):
        r=next(v for v in summaries if v['condition']==name)
        ax.plot([0,1],[r['validation_macro_f1_pct_mean'],r['test_macro_f1_pct_mean']],marker='o',label=LABELS[name])
    ax.set_xticks([0,1],['Validation','Test']);ax.set_ylabel('Mean macro-F1 (%)');ax.set_title('Original student methods: ten seeds');ax.legend(fontsize=9)
    ax=axes[1,0];proposed=[v for v in subjects if v['condition']=='proposed'];svm=[v for v in subjects if v['condition']=='svm-selected'];xs=np.arange(len(proposed))
    ax.errorbar(xs,[v['accuracy_pct_mean'] for v in proposed],yerr=[v['accuracy_pct_seed_sd'] for v in proposed],fmt='o',capsize=2,color='#397d58',label='Student mean ± seed SD')
    ax.scatter(xs,[next(s['accuracy_pct_mean'] for s in svm if s['subject']==v['subject'] and s['partition']==v['partition']) for v in proposed],marker='x',color='#9a612d',label='Selected RBF SVC')
    ax.set_xticks(xs,[('V' if v['partition']=='validation' else 'T')+str(v['subject']) for v in proposed],rotation=45,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_title('Every held-out subject (V: validation; T: test)');ax.legend(fontsize=9)
    ax=axes[1,1]
    for part,long,color in [('val','Validation','#397d58'),('test','Test','#b78345')]:
        values=[]
        for seed in SEEDS:values.append(cm_metrics(confusion(data['y_'+part],predictions[('proposed',seed)][part]))['per_class_f1']*100)
        values=np.array(values);ax.errorbar(x,values.mean(0),yerr=values.std(0,ddof=1),marker='o',capsize=3,label=long,color=color)
    ax.set_xticks(x,CLASSES,rotation=25,ha='right');ax.set_ylim(0,105);ax.set_ylabel('Class F1 (%)');ax.set_title('Proposed student: mean ± model-seed SD');ax.legend(fontsize=9)
    fig.suptitle('Original corrected WISDM: descriptive gap audit\nFixed subject split; all subjects and original outcomes retained',fontsize=13)
    fig.savefig(out/'wisdm-gap-overview.png',dpi=180);fig.savefig(out/'wisdm-gap-overview.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,part,title in zip(axes,('val','test'),('Validation','Test')):
        matrices=[confusion(data['y_'+part],predictions[('proposed',seed)][part]) for seed in SEEDS]
        average=np.mean([cm/cm.sum(1,keepdims=True)*100 for cm in matrices],axis=0)
        im=ax.imshow(average,vmin=0,vmax=100,cmap='Blues')
        for i in range(6):
            for j in range(6):ax.text(j,i,f'{average[i,j]:.1f}',ha='center',va='center',fontsize=9,color='white' if average[i,j]>50 else 'black')
        ax.set_xticks(range(6),CLASSES,rotation=45,ha='right');ax.set_yticks(range(6),CLASSES);ax.set_xlabel('Predicted class');ax.set_ylabel('True class');ax.set_title(title)
    fig.colorbar(im,ax=axes,label='Mean row percentage (%)',shrink=.8)
    fig.suptitle('Proposed student: mean confusion percentages across ten models\nRows normalized within each original split; no pooled independent-trial claim',fontsize=12)
    fig.savefig(out/'wisdm-gap-confusion.png',dpi=180);fig.savefig(out/'wisdm-gap-confusion.pdf');plt.close(fig)

if __name__=='__main__':main()
