"""Independently audit the completed CPU session and reconstruct metrics/timing tables."""
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
RUNTIMES=['tflite','keras-eager','tf-function']
SVMS={'uci':['selected','selected-compact'],'wisdm':['selected','linear-alternative','linear-alternative-compact']}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))

def csv_write(p,rows):
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def metric(y,p):
    assert y.ndim==p.ndim==1 and len(y)==len(p)>0
    assert np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer)
    assert np.isin(y,range(6)).all() and np.isin(p,range(6)).all()
    matrix=np.zeros((6,6),dtype=np.int64);np.add.at(matrix,(y,p),1)
    tp=np.diag(matrix).astype(float);den=matrix.sum(0)+matrix.sum(1)
    f=np.divide(2*tp,den,out=np.zeros(6),where=den>0)
    return dict(accuracy=float(tp.sum()/len(y)),macro_f1=float(f.mean()),per_class_f1=f.tolist(),confusion_matrix=matrix.tolist())

def same_metric(actual,saved):
    assert set(actual)==set(saved)
    for k in actual:np.testing.assert_allclose(actual[k],saved[k],rtol=0,atol=1e-12)

def compact_predict(x,state):
    score=x.astype(np.float64)@state['coef'].T+state['intercept'];votes=np.zeros((len(x),6),int)
    column=0
    for first in range(6):
        for second in range(first+1,6):
            selected=np.where(score[:,column]>0,first,second)
            np.add.at(votes,(np.arange(len(x)),selected),1);column+=1
    return state['classes'][np.argmax(votes,axis=1)]

def summarize(ns,batch):
    a=np.array(ns,dtype=np.float64)
    assert a.ndim==1 and len(a)>1 and np.isfinite(a).all() and (a>0).all()
    ms=a/1e6
    return dict(repetitions=len(a),batch_size=batch,mean_ms=float(sum(ms)/len(ms)),median_ms=float(np.median(ms)),
        p95_ms=float(np.quantile(ms,.95)),std_ms=float(np.std(ms,ddof=1)),min_ms=float(min(ms)),max_ms=float(max(ms)),
        samples_per_second=float(batch*1e9/np.mean(a)))

def verify_timing_summary(ns,batch,saved):
    actual=summarize(ns,batch);assert set(actual)==set(saved)
    for k in actual:np.testing.assert_allclose(actual[k],saved[k],rtol=1e-12,atol=1e-12)
    return actual

def check_source(ds,seed,part,method,actual,expected,rows):
    for key in ('accuracy','macro_f1'):
        if key not in expected:continue
        error=abs(actual[key]-expected[key]);assert error<=1e-4
        rows.append(dict(dataset=ds,seed=seed,partition=part,method=method,metric=key,saved=expected[key],reproduced=actual[key],absolute_error=error,tolerance=1e-4))

def check_logits(ds,seed,part,name,reference,other,record,rows):
    assert reference.shape==other.shape and reference.ndim==2 and reference.shape[1]==6
    assert np.isfinite(reference).all() and np.isfinite(other).all()
    np.testing.assert_allclose(other,reference,rtol=1e-4,atol=1e-4)
    assert np.array_equal(reference.argmax(1),other.argmax(1))
    err=float(np.abs(reference-other).max())
    assert record['label_disagreements']==0 and record['atol']==record['rtol']==1e-4
    np.testing.assert_allclose(err,record['max_absolute_logit_error'],rtol=0,atol=1e-12)
    rows.append(dict(dataset=ds,seed=seed,partition=part,runtime=name,label_disagreements=0,max_absolute_logit_error=err,atol=1e-4,rtol=1e-4))

def expected_cases(smoke=False):
    cases=[]
    for ds in ('uci','wisdm'):
        cases.extend(f'{ds}/svm-{name}' for name in SVMS[ds])
        cases.extend(f'{ds}/{name}/{seed}' for seed in ([11] if smoke else SEEDS) for name in RUNTIMES)
    return cases

def expected_phase_files(prefix,smoke=False):
    names={f'{prefix}/{ds}-{kind}' for ds in ('uci','wisdm') for kind in ('timing-corpus.json','svm-test.npz')}
    names|={f'{prefix}/{ds}/seed-{seed}/test.{ext}' for ds in ('uci','wisdm') for seed in ([11] if smoke else SEEDS) for ext in ('npz','json')}
    names.add(f'{prefix}/results.json');return names

def expected_preflight():
    names={f'{ds}/{name}' for ds in ('uci','wisdm') for name in ('data.npz','data.json','svm-preflight.json','svm-validation-predictions.npz')}
    names|={'uci/selected-compact.npz','wisdm/linear-alternative.joblib','wisdm/linear-alternative-compact.npz'}
    names|={f'{ds}/seed-{seed}/{name}' for ds in ('uci','wisdm') for seed in SEEDS for name in ('student-float32.tflite','validation.npz','preflight.json')}
    assert len(names)==71;return names

def verify_marker(root,name,context,paths,checked):
    marker=read(root/name);assert marker['status']=='complete' and marker['context']==context
    assert len(marker['files'])==len(paths) and {r['path'] for r in marker['files']}==paths
    for row in marker['files']:
        p=(root/row['path']).resolve();assert root in p.parents and p.is_file() and sha(p)==row['sha256']
        checked.append(dict(path=str(p),sha256=row['sha256']))
    return marker

def verify_architecture(path,report,shape,source):
    with zipfile.ZipFile(path) as z:config=json.loads(z.read('config.json'))['config']
    layers=report['architecture']['layers'];types=[v['class_name'] for v in layers]
    assert types==['InputLayer','Conv1D','BatchNormalization','MaxPooling1D','Conv1D','GlobalAveragePooling1D','Dense','Dense']
    assert types==[v['class_name'] for v in config['layers']]
    for a,b in zip(layers,config['layers']):
        # Keras regenerates object-sharing IDs; compare actual layer settings.
        for key in ('filters','kernel_size','strides','padding','activation','use_bias','units','axis','epsilon','momentum','center','scale','pool_size','batch_shape'):
            assert a['config'].get(key)==b['config'].get(key)
    assert layers[0]['config']['batch_shape']==[None,*shape]
    assert (layers[1]['config']['filters'],layers[1]['config']['kernel_size'],layers[1]['config']['padding'])==(24,[5],'same')
    assert (layers[4]['config']['filters'],layers[4]['config']['kernel_size'],layers[4]['config']['padding'])==(48,[3],'same')
    assert layers[3]['config']['pool_size']==[2] and layers[6]['config']['units']==32 and layers[7]['config']['units']==6
    count=5*shape[-1]*24+24+4*24+3*24*48+48+48*32+32+32*6+6
    assert report['keras_parameter_count']==source['methods']['proposed']['footprint']['parameters']==count
    assert report['keras_trainable_parameters']==count-48 and report['keras_numeric_weight_bytes']==count*4

def main():
    p=argparse.ArgumentParser();p.add_argument('--drive',type=Path,required=True);p.add_argument('--session',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();drive=args.drive.resolve();root=args.session.resolve();out=args.output.resolve()
    assert (drive/'results/revision-2/runtime-benchmark').resolve() in root.parents
    if not (root/'completed.json').is_file():raise SystemExit('Research incomplete; no analysis written.')
    sys.path.insert(0,str(drive/'revision_2/runtime_benchmark'))
    from run_runtime import verify_inputs,protocol
    frozen=verify_inputs(drive);spec=protocol(frozen);env=read(root/'environment.json')
    assert spec['version']=='revision2-runtime-benchmark-v2' and read(root/'protocol.json')==spec
    assert env['cpu_only'] and env['threads']==env['tf_intra_threads']==env['tf_inter_threads']==1
    assert env['deterministic_operations'] and not env['tf32'] and not env['onednn']
    assert env['sklearn']=='1.6.1' and all(v['num_threads']==1 for v in env['threadpools'])
    context=dict(environment_sha256=digest(env),frozen_sha256=digest(frozen),specification_sha256=digest(spec))
    completed=read(root/'completed.json');smoke=read(root/'smoke-completed.json');checked=[]
    research_prefix=Path(completed['results_file']).parent.as_posix();smoke_prefix=Path(smoke['results_file']).parent.as_posix()
    assert research_prefix.startswith('research/') and smoke_prefix.startswith('smoke/')
    gate=verify_marker(root,'preflight-completed.json',context,expected_preflight(),checked)
    assert gate['students']==20 and gate['svm_fits']==1 and gate['neural_training_fits']==0 and gate['test_evaluated'] is False
    for name,is_smoke,prefix in [('smoke-completed.json',True,smoke_prefix),('completed.json',False,research_prefix)]:
        m=verify_marker(root,name,context,expected_phase_files(prefix,is_smoke),checked)
        assert m['mode']==('smoke' if is_smoke else 'research') and m['student_checkpoints']==(2 if is_smoke else 20)
        assert m['conditions']==(11 if is_smoke else 65) and m['timing_records']==(11 if is_smoke else 195)
        assert m['neural_training_fits']==0 and m['auxiliary_svm_fits']==1 and m['test_evaluated']
    result=read(root/completed['results_file']);smoke_result=read(root/smoke['results_file'])
    assert result['session_id']!=smoke_result['session_id'] and result['timing_process_pid']!=smoke_result['timing_process_pid']
    assert result['case_keys']==expected_cases() and len(result['metrics'])==65
    assert result['neural_training_fits']==result['svm_fits_in_timing_phase']==0
    metrics_rows=[];source_parity=[];conversion_rows=[];storage=[];features=[];snapshots=set()
    data={};sources={};saved_metric={(r['dataset'],r['condition'],r['seed']):r for r in result['metrics']}
    assert len(saved_metric)==65
    import joblib,sklearn
    assert sklearn.__version__=='1.6.1'
    legacy={}
    for ds in ('uci','wisdm'):
        source=read(drive/frozen['seed_records'][ds]['11']);sources[ds]=source
        with np.load(root/ds/'data.npz',allow_pickle=False) as z:a={k:z[k].copy() for k in z.files}
        data[ds]=a;metadata=read(root/ds/'data.json')
        assert metadata['context']==context and metadata['manifest']==source['dataset_manifest'] and metadata['cache_sha256']==sha(root/ds/'data.npz')
        for part,long in [('val','validation'),('test','test')]:
            y=a['y_'+part];subjects=a['subjects_'+part];m=source['dataset_manifest'][long]
            assert len(y)==m['samples'] and np.unique(subjects).tolist()==m['subjects']
            assert {str(i):int((y==i).sum()) for i in range(6)}==m['class_counts']
            assert a['raw_'+part].shape[1:]==tuple(source['dataset_manifest']['raw_shape'])
            assert a['features_'+part].shape[1]==source['dataset_manifest']['feature_count']
        assert not set(a['subjects_val'])&set(a['subjects_test'])
        features.extend(dict(dataset=ds,index=i+1,name=n) for i,n in enumerate(source['dataset_manifest']['feature_names']))
        svm_info=read(root/ds/'svm-preflight.json');assert svm_info['context']==context and svm_info['svm_fits']==(ds=='wisdm')
        with np.load(root/ds/'svm-validation-predictions.npz',allow_pickle=False) as z:vp={k:z[k].copy() for k in z.files}
        with np.load(root/research_prefix/f'{ds}-svm-test.npz',allow_pickle=False) as z:tp={k:z[k].copy() for k in z.files}
        np.testing.assert_array_equal(tp['y_test'],a['y_test']);np.testing.assert_array_equal(tp['subjects_test'],a['subjects_test'])
        assert set(vp)==set(SVMS[ds]) and set(tp)=={'y_test','subjects_test'}|{'svm-'+n for n in SVMS[ds]}
        models={'selected':joblib.load(drive/frozen['svms'][ds])}
        if ds=='wisdm':models['linear-alternative']=joblib.load(root/ds/'linear-alternative.joblib')
        for name,model in models.items():
            expected=source['methods']['svm']['selected_params'] if name=='selected' else max([v for v in source['methods']['svm']['tuning_results'] if v['params']['kernel']=='linear'],key=lambda v:v['validation_macro_f1'])['params']
            assert all(model.get_params()[k]==v for k,v in expected.items()) and not model.break_ties
            assert svm_info['models'][name]['params']=={k:model.get_params()[k] for k in ('C','kernel','gamma','break_ties','decision_function_shape')}
            numeric=sum(getattr(model,k).nbytes for k in ('support_vectors_','dual_coef_','intercept_','classes_','n_support_','probA_','probB_'))
            assert numeric==svm_info['models'][name]['numeric_state_bytes']
            for part,pred in [('val',vp[name]),('test',tp['svm-'+name])]:np.testing.assert_array_equal(model.predict(a['features_'+part]),pred)
            if ds=='uci' and name=='selected':
                alternative=model.classes_[model.decision_function(a['features_test']).argmax(1)];old=metric(a['y_test'],alternative)
                legacy=dict(predict_vs_decision_argmax_disagreements=int((alternative!=tp['svm-selected']).sum()),decision_argmax_macro_f1=old['macro_f1'],production_macro_f1=source['methods']['svm']['test']['macro_f1'],predict_macro_f1=metric(a['y_test'],tp['svm-selected'])['macro_f1'])
                np.testing.assert_allclose(old['macro_f1'],legacy['production_macro_f1'],rtol=0,atol=1e-12)
            artifact=drive/frozen['svms'][ds] if name=='selected' else root/ds/'linear-alternative.joblib'
            assert saved_metric[(ds,'svm-'+name,None)]['artifact_bytes']==artifact.stat().st_size
            storage.append(dict(dataset=ds,condition='svm-'+name,seed=None,artifact_bytes=artifact.stat().st_size,numeric_state_bytes=numeric,parameters=None,support_vectors=len(model.support_vectors_),dtype='float64',scope='classifier only; no preprocessing/runtime'))
            if model.kernel=='linear':
                compact=root/ds/f'{name}-compact.npz'
                with np.load(compact,allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
                np.testing.assert_allclose(state['coef'],model.coef_,rtol=0,atol=1e-12);np.testing.assert_array_equal(state['intercept'],model.intercept_);np.testing.assert_array_equal(state['classes'],model.classes_)
                assert state['coef'].shape==(15,a['features_val'].shape[1])
                assert saved_metric[(ds,'svm-'+name+'-compact',None)]['artifact_bytes']==compact.stat().st_size
                for part,pred in [('val',vp[name+'-compact']),('test',tp['svm-'+name+'-compact'])]:
                    np.testing.assert_array_equal(compact_predict(a['features_'+part],state),pred)
                    np.testing.assert_array_equal(pred,vp[name] if part=='val' else tp['svm-'+name])
                storage.append(dict(dataset=ds,condition='svm-'+name+'-compact',seed=None,artifact_bytes=compact.stat().st_size,numeric_state_bytes=sum(v.nbytes for v in state.values()),parameters=None,support_vectors=0,dtype='float64',scope='classifier only; no preprocessing/runtime'))
        for name in SVMS[ds]:
            for part,long,pred in [('val','validation',vp[name]),('test','test',tp['svm-'+name])]:
                values=metric(a['y_'+part],pred);metrics_rows.append(dict(dataset=ds,condition='svm-'+name,seed=None,partition=long,accuracy_pct=values['accuracy']*100,macro_f1_pct=values['macro_f1']*100))
                if part=='test':same_metric(values,saved_metric[(ds,'svm-'+name,None)]['metrics'])
                if part=='val':same_metric(values,svm_info['models'][name.replace('-compact','')]['validation']['reproduced'])
                if name=='selected':check_source(ds,None,long,'svm-selected',values,dict(macro_f1=source['methods']['svm']['validation_macro_f1']) if part=='val' else source['methods']['svm']['test'],source_parity)
                if name=='linear-alternative' and part=='val':
                    best=max([v for v in source['methods']['svm']['tuning_results'] if v['params']['kernel']=='linear'],key=lambda v:v['validation_macro_f1'])
                    check_source(ds,None,long,'svm-linear-alternative',values,dict(macro_f1=best['validation_macro_f1']),source_parity)
        for seed in SEEDS:
            current=read(drive/frozen['seed_records'][ds][str(seed)]);bank=root/ds/f'seed-{seed}';pref=read(bank/'preflight.json');test=read(root/research_prefix/ds/f'seed-{seed}/test.json')
            assert pref['context']==test['context']==context and pref['training_fits']==0
            original=drive/frozen['students'][ds][str(seed)];flat=bank/'student-float32.tflite'
            assert pref['source_model_sha256']==sha(original) and pref['tflite_sha256']==sha(flat)
            assert flat.read_bytes()[4:8]==b'TFL3' and not any(v=='DELEGATE' or v.startswith(('Flex','CUSTOM')) for v in pref['ops'])
            assert pref['keras_checkpoint_bytes']==original.stat().st_size and pref['tflite_artifact_bytes']==flat.stat().st_size
            verify_architecture(original,pref,source['dataset_manifest']['raw_shape'],current)
            storage.append(dict(dataset=ds,condition='tflite',seed=seed,artifact_bytes=flat.stat().st_size,numeric_state_bytes=None,keras_checkpoint_bytes=original.stat().st_size,keras_numeric_weight_bytes=pref['keras_numeric_weight_bytes'],parameters=pref['keras_parameter_count'],trainable_parameters=pref['keras_trainable_parameters'],support_vectors=None,dtype='float32',scope='TFLite artifact; source Keras numeric weights separate; no preprocessing/runtime'))
            for part,long,path,record in [('val','validation',bank/'validation.npz',pref),('test','test',root/research_prefix/ds/f'seed-{seed}/test.npz',test)]:
                with np.load(path,allow_pickle=False) as z:logits={k:z[k].copy() for k in ('keras_logits','lite_logits','lite_batch1_logits','traced_logits')}
                if part=='test':
                    with np.load(path,allow_pickle=False) as z:
                        for k in ('y_test','subjects_test'):np.testing.assert_array_equal(z[k],a[k])
                reference=logits['keras_logits'];assert len(reference)==len(a['y_'+part])
                for name,key,rec_key in [('tflite','lite_logits','conversion'),('tflite-batch1','lite_batch1_logits','conversion_batch1'),('tf-function','traced_logits','traced')]:check_logits(ds,seed,long,name,reference,logits[key],record[rec_key],conversion_rows)
                original_metric=metric(a['y_'+part],reference.argmax(1));check_source(ds,seed,long,'proposed',original_metric,current['methods']['proposed'][long],source_parity)
                source_report=record['validation_source_parity'] if part=='val' else record['source_parity']
                same_metric(original_metric,source_report['reproduced']);assert source_report['tolerance']==1e-4 and source_report['saved']==current['methods']['proposed'][long]
                for k in ('accuracy','macro_f1'):np.testing.assert_allclose(source_report['errors'][k],abs(original_metric[k]-source_report['saved'][k]),rtol=0,atol=1e-12)
                for name,key in [('tflite','lite_logits'),('keras-eager','keras_logits'),('tf-function','traced_logits')]:
                    values=metric(a['y_'+part],logits[key].argmax(1));metrics_rows.append(dict(dataset=ds,condition=name,seed=seed,partition=long,accuracy_pct=values['accuracy']*100,macro_f1_pct=values['macro_f1']*100))
                    if part=='test':same_metric(values,saved_metric[(ds,name,seed)]['metrics'])
            snapshots.add(bank/'preflight.json');snapshots.add(root/research_prefix/ds/f'seed-{seed}/test.json')
        snapshots|={root/ds/'data.json',root/ds/'svm-preflight.json'}
    assert len(metrics_rows)==130 and len(conversion_rows)==120 and len(source_parity)==87 and len(storage)==25
    rounds=[];model_rows=[]
    for report,is_smoke,prefix in [(smoke_result,True,smoke_prefix),(result,False,research_prefix)]:
        case_keys=expected_cases(is_smoke);assert report['case_keys']==case_keys and report['context']==context
        assert report['mode']==('smoke' if is_smoke else 'research')
        num_rounds=1 if is_smoke else 3;assert len(report['round_orders'])==num_rounds
        for ds in ('uci','wisdm'):
            corpus=read(root/prefix/f'{ds}-timing-corpus.json');assert corpus['context']==context
            indices=np.random.default_rng(20261019).permutation(len(data[ds]['y_val']))
            expected=dict(warm1=indices[:30],sample1=indices[30:330],warm128=indices[:128],sample128=indices[128:256])
            for key,values in expected.items():
                np.testing.assert_array_equal(corpus['validation_indices'][key],values)
                np.testing.assert_array_equal(corpus['subject_ids'][key],data[ds]['subjects_val'][values])
            if not is_smoke:snapshots.add(root/prefix/f'{ds}-timing-corpus.json')
        assert len(report['timings'])==len(case_keys)*num_rounds
        for n in range(num_rounds):
            expected=[case_keys[i] for i in np.random.default_rng(20261019+n).permutation(len(case_keys))]
            assert report['round_orders'][n]==expected
            assert [row['key'] for row in report['timings'] if row['round']==n]==expected
        assert {(row['key'],row['round']) for row in report['timings']}=={(key,n) for key in case_keys for n in range(num_rounds)}
        for row in report['timings']:
            ds,name,*seed_text=row['key'].split('/');seed=int(seed_text[0]) if seed_text else None
            assert row['dataset']==ds and row['condition']==name and row['seed']==seed
            for batch,count in [(1,10 if is_smoke else 300),(128,3 if is_smoke else 30)]:
                assert len(row[f'batch{batch}_ns'])==count
                summary=verify_timing_summary(row[f'batch{batch}_ns'],batch,row[f'batch{batch}_summary'])
                if not is_smoke:rounds.append(dict(dataset=ds,condition=name,seed=seed,round=row['round']+1,**summary))
    for key in result['case_keys']:
        subset=[v for v in result['timings'] if v['key']==key];ds,name,*seed_text=key.split('/');seed=int(seed_text[0]) if seed_text else None
        a=summarize([n for v in subset for n in v['batch1_ns']],1);b=summarize([n for v in subset for n in v['batch128_ns']],128)
        row=dict(dataset=ds,condition=name,seed=seed,batch1_mean_round_median_ms=float(np.mean([v['batch1_summary']['median_ms'] for v in subset])),
            batch1_round_median_sd_ms=float(np.std([v['batch1_summary']['median_ms'] for v in subset],ddof=1)),
            batch1_pooled_median_ms=a['median_ms'],batch1_pooled_p95_ms=a['p95_ms'],batch1_pooled_mean_ms=a['mean_ms'],
            batch128_pooled_mean_ms=b['mean_ms'],batch128_pooled_median_ms=b['median_ms'],batch128_samples_per_second=b['samples_per_second'],batch1_calls=900,batch128_calls=90)
        m=next(v for v in metrics_rows if (v['dataset'],v['condition'],v['seed'],v['partition'])==(ds,name,seed,'test'))
        row.update(accuracy_pct=m['accuracy_pct'],macro_f1_pct=m['macro_f1_pct']);model_rows.append(row)
    summaries=[]
    for ds in ('uci','wisdm'):
        for name in RUNTIMES+['svm-'+v for v in SVMS[ds]]:
            values=[v for v in model_rows if v['dataset']==ds and v['condition']==name];n=len(values);assert n==(10 if name in RUNTIMES else 1)
            summary=dict(dataset=ds,condition=name,models=n,rounds_per_model=3)
            for key in ('batch1_mean_round_median_ms','batch1_pooled_p95_ms','batch128_samples_per_second','accuracy_pct','macro_f1_pct'):
                numbers=[v[key] for v in values];summary[key+'_mean']=float(np.mean(numbers));summary[key+'_seed_sd']=float(np.std(numbers,ddof=1)) if n>1 else None
            summaries.append(summary)
    # Historical summaries retain their actual measurement scopes; no invented causal decomposition.
    historical=[]
    for ds in ('uci','wisdm'):
        original=[read(drive/frozen['seed_records'][ds][str(seed)]) for seed in SEEDS]
        controlled=read(drive/f'results/svm-student-comparison-v1/{ds}/benchmark.json')
        historical.append(dict(dataset=ds,record='production proposed latency',mean_recorded_median_ms=float(np.mean([v['methods']['proposed']['latency']['median_ms'] for v in original])),tensorflow=original[0]['methods']['proposed']['latency']['tensorflow_version'],threads='not fully recorded',batch='1',scope='first-window eager call through logits; 30 warmups, 300 calls; no argmax',cpu_model='not recorded'))
        historical.append(dict(dataset=ds,record='controlled benchmark v1 proposed latency',mean_recorded_median_ms=float(np.mean([v['latency']['median_ms'] for v in controlled['details']['students']])),tensorflow=controlled['environment']['tensorflow'],threads=1,batch='1',scope='first-window eager call through logits; 30/300; no argmax',cpu_model='not recorded'))
        historical.append(dict(dataset=ds,record='production SVM amortized batch inference',mean_recorded_median_ms=None,mean_amortized_ms_per_sample=float(np.mean([v['methods']['svm']['inference']['mean_ms_per_sample'] for v in original])),threads='not fully recorded',batch='full test set',scope='full-batch duration divided by N; not batch-one latency',cpu_model='not recorded'))
    out.mkdir(parents=True,exist_ok=True)
    for name,values in [('metrics',metrics_rows),('models',model_rows),('rounds',rounds),('summary',summaries),('source-parity',source_parity),('conversion-parity',conversion_rows),('storage',storage),('features',features),('reconciliation',historical)]:csv_write(out/f'runtime-{name}.csv',values)
    make_report(out,root,env,summaries,storage,source_parity,legacy)
    plot(out,summaries,storage)
    snapshots|={root/name for name in ('completed.json','preflight-completed.json','smoke-completed.json','environment.json','protocol.json')}
    snapshots|={drive/'revision_2/runtime_benchmark'/name for name in ('run_runtime.py','test_runtime_guards.py','frozen-inputs.json')}
    snapshot_rows=[]
    for path in sorted(snapshots):
        if root in path.parents:
            relative=path.relative_to(root)
            # The source identity remains in the manifest; avoid duplicating long
            # session/UUID components inside Windows evidence paths.
            if relative.parts[0]=='research':relative=Path('research',*relative.parts[2:])
        else:relative=Path('code')/path.name
        target=out/'evidence_snapshot/runtime-benchmark'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
        snapshot_rows.append(dict(source=str(path),snapshot=str(target.relative_to(out)),sha256=sha(path)))
    shutil.copy2(Path(__file__),out/'evidence_snapshot/runtime-benchmark/analyze_runtime_benchmark.py')
    outputs={p.name:sha(p) for p in out.glob('runtime-*') if p.suffix in ('.csv','.md','.png','.pdf') and p.name.startswith(('runtime-metrics','runtime-models','runtime-rounds','runtime-summary','runtime-source-parity','runtime-conversion-parity','runtime-storage','runtime-features','runtime-reconciliation','runtime-report','runtime-latency','runtime-artifact'))}
    audit=dict(analysis_at=datetime.now(ZoneInfo('Asia/Tehran')).isoformat(),session=str(root),version=spec['version'],input_hashes_verified=64,output_hashes_verified=len(checked),
        research_conditions_verified=65,research_timing_records_verified=195,research_raw_batch1_calls=58500,research_raw_batch128_calls=5850,
        validation_test_metric_sets_reconstructed=130,conversion_agreements_verified=120,source_metric_comparisons=len(source_parity),source_max_absolute_error=max(v['absolute_error'] for v in source_parity),
        new_neural_training_fits=0,auxiliary_linear_svm_fits=1,smoke_excluded=True,legacy_uci_svm=legacy,checked_hashes=checked,snapshots=snapshot_rows,
        outputs=outputs,analyzer_sha256=sha(Path(__file__)),analysis_environment=dict(python=sys.version,numpy=np.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    (out/'runtime-verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'Verified 64 inputs, {len(checked)} output hashes, 130 metric sets and 195 research timing records. Nine CSVs, report and figures written.')

def make_report(out,root,env,summaries,storage,parity,legacy):
    def get(ds,name):return next(v for v in summaries if v['dataset']==ds and v['condition']==name)
    lines=['# Deployment-runtime and controlled CPU benchmark','',
        f'Completed session: `{root.name}`. All 64 frozen inputs and 125 recorded preflight/smoke/research output hashes passed. '
        'All 65 research conditions, three complete timing rounds, 58,500 batch-one and 5,850 batch-128 raw call durations '
        'were checked. Independent predictions reconstruct 130 validation/test metric sets. '
        'Twenty students passed original metric checks and 120 converted/traced logit-and-label agreement checks. '
        'Compact linear SVC predictions match their saved support-vector models exactly. No neural training occurred; '
        'one auxiliary WISDM linear SVC was fitted. Smoke timings are excluded.','',
        '## Accuracy and classifier timing','',
        'Primary per-model batch-one statistic is the equally weighted mean of its three round medians. '
        'Student rows show the mean and sample SD of that statistic across ten saved models; SVM rows describe one fitted model. '
        'The CSVs also retain within-model round SD, pooled median/p95, each round and batch-128 throughput. '
        'No timing repetition or SVM row is treated as an independent subject cohort.','',
        '| Dataset | Classifier / runtime | Test macro-F1 mean (%) | Batch-one latency (ms) | Batch-128 throughput (samples/s) |',
        '|---|---|---:|---:|---:|']
    labels={'tflite':'Student / TFLite','keras-eager':'Student / Keras eager','tf-function':'Student / traced TensorFlow','svm-selected':'Selected SVC','svm-selected-compact':'Same selected linear SVC / compact','svm-linear-alternative':'Auxiliary linear SVC','svm-linear-alternative-compact':'Same auxiliary linear SVC / compact'}
    for v in summaries:
        value=f"{v['batch1_mean_round_median_ms_mean']:.6f}"
        if v['models']>1:value+=f" ± {v['batch1_mean_round_median_ms_seed_sd']:.6f}"
        lines.append(f"| {v['dataset'].upper()} | {labels[v['condition']]} | {v['macro_f1_pct_mean']:.3f} | {value} | {v['batch128_samples_per_second_mean']:.0f} |")
    lines+=['','Uniform timing definitions do not make representations equivalent: students receive float32 standardized raw '
        '128-sample windows; UCI SVC receives 561 supplied features and WISDM SVC receives 139 engineered features. '
        'Classifier dispatch, runtime input copies, output materialization and argmax/voting are included. '
        'Feature extraction, normalization, loading, tracing, conversion and interpreter allocation are excluded. '
        'Batch-128 throughput is a repeated fixed validation batch, not the historical full-test throughput.','',
        'In this controlled session TFLite has lower batch-one latency than the measured SVC representations on '
        'both datasets. Compact linear SVCs have substantially higher batch-128 throughput than TFLite, and '
        'their measured test macro-F1 is also higher than the student. Therefore, do not claim uniform '
        'speed/accuracy/size superiority across batch sizes and representations. The auxiliary WISDM linear '
        'model is numerically higher on this already-used test than the validation-selected RBF model; retain '
        'that outcome without replacing the validation selection or claiming a significant advantage.','',
        '## Storage and compact baselines','',
        '| Dataset | Classifier | Artifact KiB | Numeric state KiB |', '|---|---|---:|---:|']
    for ds in ('uci','wisdm'):
        for name in ['tflite']+['svm-'+v for v in SVMS[ds]]:
            rows=[v for v in storage if v['dataset']==ds and v['condition']==name]
            artifact=np.mean([v['artifact_bytes'] for v in rows])/1024
            numeric='—' if name=='tflite' else f"{np.mean([v['numeric_state_bytes'] for v in rows])/1024:.3f}"
            lines.append(f'| {ds.upper()} | {labels[name]} | {artifact:.3f} | {numeric} |')
    lines+=['','The TFLite artifact includes graph metadata and constants; it is not the source Keras numeric-weight byte count. '
        'Source student numeric weights are 25,880 bytes (UCI; 6,470 total / 6,422 trainable parameters) and '
        '23,000 bytes (WISDM; 5,750 total / 5,702 trainable). Batch-normalization moving statistics account for '
        '48 nontrainable values. The compact SVC stores fifteen float64 pairwise coefficient rows, intercepts and '
        'class labels, using the same vote/tie convention, rather than a newly trained LinearSVC. '
        'Its NPZ is uncompressed; the original joblib is uncompressed. These sizes exclude runtime binaries, '
        'program code, preprocessing/scaling state, feature extraction and peak working memory.','',
        'Compare against the compact linear representation when discussing storage, rather than attributing all '
        'support-vector storage to an unavoidable linear-SVM requirement. The WISDM compact linear alternative '
        'has its own measured accuracy and can be smaller than the student; it does not replace the '
        'validation-selected RBF model. Keep both baselines and the accuracy trade-off.','',
        '## Verified scope, historical reconciliation and limits','',
        f"The new session used {next(v for v in env['cpu_identification'] if v.startswith('model name')).split(':',1)[1].strip()}, "
        f"two available logical CPUs, affinity {env['affinity']}, one numerical/runtime thread, TensorFlow {env['tensorflow']}, "
        f"Keras {env['keras']}, NumPy {env['numpy']} and scikit-learn {env['sklearn']}. "
        'TFLite uses optimized builtin CPU kernels with default delegates explicitly disabled; no quantization, '
        'GPU, custom or Select TF operations were used. OneDNN and TF32 are disabled. '
        'All conditions ran in one research process with fixed validation indices and deterministic interleaved order. '
        'This establishes a controlled Colab classifier benchmark, not physical-edge-device, energy, end-to-end '
        'or independent generalization evidence.','',
        'Historical production SVM timings are full-batch duration/N, not measured batch-one latency. '
        'Production neural and the earlier controlled-v1 benchmark use eager Keras on the first window, through '
        'logit materialization without final argmax. The earlier controlled benchmark recorded one CPU thread '
        'and TensorFlow 2.20.0 but not a complete CPU identity. The new corpus, runtime version, execution path '
        'and output scope differ. Preserve old values as historical; do not invent a single hardware or thread '
        'explanation, or interpret old/new ratios as a controlled causal decomposition. Prefer this new unified '
        'benchmark for the revised classifier-runtime table if adopted during manuscript integration.','',
        f"The largest original-metric discrepancy is {max(v['absolute_error'] for v in parity):.12g}, below the unchanged 0.0001 tolerance. "
        f"For UCI SVC, the saved production test macro-F1 is {legacy['production_macro_f1']:.12f}; "
        f"the explicit SVC.predict rule gives {legacy['predict_macro_f1']:.12f}. "
        f"Decision-function argmax differs on {legacy['predict_vs_decision_argmax_disagreements']} test window and reproduces "
        'the legacy value exactly, consistent with the previously documented prediction-rule distinction. '
        'The current compact and support-vector representations both use SVC.predict voting semantics and agree '
        'exactly; this is not a conversion discrepancy. Preserve original records and report the rule explicitly.','',
        'The aborted version-1 session passed conversions but failed before smoke/timing due to unordered '
        'threadpool enumeration. Version 2 retains every library/version/count while sorting that list; '
        'the completed session repeats all gates. No version-1 timing is included and no tolerance was relaxed.','',
        '## Draft manuscript and reviewer-response integration','',
        'Draft manuscript text: We converted all twenty saved proposed-student models to float32 TFLite and '
        'verified original validation/test metrics and exact predicted-label agreement with Keras for both '
        'batch-one and batched execution. We benchmarked all classifiers in one CPU session with one runtime '
        'thread, prespecified validation-window indices and three interleaved timing rounds. We report '
        'classifier-only latency and throughput, excluding normalization and feature extraction. Selected SVC '
        'models are shown alongside equivalent coefficient-and-vote exports for linear SVCs, with the WISDM '
        'linear model reported as a separate validation-specified accuracy/size alternative. These measurements '
        'do not establish physical-device, end-to-end or energy performance.','']
    for ds in ('uci','wisdm'):
        lite=get(ds,'tflite');selected=get(ds,'svm-selected')
        compact=get(ds,'svm-selected-compact' if ds=='uci' else 'svm-linear-alternative-compact')
        lines.append(f"Draft numerical insertion ({ds.upper()}): student TFLite batch-one latency is {lite['batch1_mean_round_median_ms_mean']:.6f} ± {lite['batch1_mean_round_median_ms_seed_sd']:.6f} ms across ten models. Selected SVC latency is {selected['batch1_mean_round_median_ms_mean']:.6f} ms; the separately labeled compact linear representation is {compact['batch1_mean_round_median_ms_mean']:.6f} ms. Test macro-F1 is {lite['macro_f1_pct_mean']:.3f}%, {selected['macro_f1_pct_mean']:.3f}% and {compact['macro_f1_pct_mean']:.3f}%, respectively.")
        lines.append('')
    lines+=['Draft response for reviewer 1 comments 6–7 (final section/table/page references pending): We added a '
        'deployment-runtime conversion and controlled CPU benchmark with explicit hardware, thread, runtime, '
        'warm-up, repetition, input and cost scope. All twenty saved students pass original metric and conversion '
        'parity checks. We now distinguish historical amortized SVM throughput from batch-one latency and '
        'distinguish eager Keras from deployment-runtime measurements. We also evaluate the existing linear '
        'SVC in an equivalent compact representation and include the original validation-selected WISDM linear '
        'alternative alongside the selected RBF SVC. Storage claims must use these representation-aware '
        'measurements and clearly exclude preprocessing/runtime state. Manuscript and response edits, numerical '
        'table integration and full claim consistency checks remain pending.','',
        f"Analysis date: {datetime.now(ZoneInfo('Asia/Tehran')).date().isoformat()}. Original WISDM gap/error audit remains pending.",'']
    (out/'runtime-report.md').write_text('\n'.join(lines),encoding='utf-8')

def plot(out,summaries,storage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype']=42
    labels={'tflite':'Student: TFLite','keras-eager':'Student: eager Keras','tf-function':'Student: traced TF','svm-selected':'Selected SVC','svm-selected-compact':'Selected linear: compact','svm-linear-alternative':'Auxiliary linear SVC','svm-linear-alternative-compact':'Auxiliary linear: compact'}
    for kind in ('latency','artifacts'):
        fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
        for ax,ds in zip(axes,('uci','wisdm')):
            names=(RUNTIMES if kind=='latency' else ['tflite'])+['svm-'+n for n in SVMS[ds]]
            values=[];errors=[]
            for n in names:
                if kind=='latency':
                    s=next(v for v in summaries if v['dataset']==ds and v['condition']==n);values.append(s['batch1_mean_round_median_ms_mean']);errors.append(s['batch1_mean_round_median_ms_seed_sd'] or 0)
                else:values.append(np.mean([v['artifact_bytes'] for v in storage if v['dataset']==ds and v['condition']==n])/1024);errors.append(0)
            y=np.arange(len(names));ax.barh(y,values,xerr=errors,color=['#397d58' if n=='tflite' else '#86a7b7' for n in names],capsize=3)
            ax.set_yticks(y,[labels[n] for n in names]);ax.invert_yaxis();ax.set_xscale('log');ax.grid(axis='x',alpha=.2);ax.set_title(ds.upper())
            ax.set_xlabel('Batch-one latency (ms; log scale)' if kind=='latency' else 'Classifier artifact (KiB; log scale)')
        fig.suptitle('Controlled one-thread CPU classifier timing\nThree round medians per model; student error bars: seed SD' if kind=='latency' else 'Classifier artifact sizes\nPreprocessing and runtime binaries excluded',fontsize=12)
        name='runtime-latency' if kind=='latency' else 'runtime-artifacts';fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.pdf'));plt.close(fig)

if __name__=='__main__':main()
