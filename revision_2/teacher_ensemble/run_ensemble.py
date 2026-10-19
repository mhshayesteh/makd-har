"""Frozen teacher inference only: individuals and four predeclared probability ensembles."""
import argparse
import dataclasses
import json
import math
import os
import platform
import sys
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parent
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(PACKAGE/'pipeline_snapshot'))
from run_coalition_noise import sha,digest,atomic_json,ensure_same,verify_file,load_bundle

SEEDS=[11,23,37,53,71,89,107,131,167,197]
VERSION='revision2-teacher-ensemble-v1'
TOLERANCE=1e-4


def require(ok,message):
    if not ok:raise RuntimeError(message)


def metrics(truth,pred):
    truth=np.asarray(truth);pred=np.asarray(pred)
    require(truth.ndim==pred.ndim==1 and len(truth)==len(pred)>0,'Invalid label dimensions')
    require(np.issubdtype(truth.dtype,np.integer) and np.issubdtype(pred.dtype,np.integer),'Labels must be integer')
    require(np.isin(truth,range(6)).all() and np.isin(pred,range(6)).all(),'Class outside six-class task')
    matrix=np.bincount(truth*6+pred,minlength=36).reshape(6,6);tp=np.diag(matrix).astype(float)
    support=matrix.sum(axis=1);predicted=matrix.sum(axis=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(6),where=support+predicted>0)
    precision=np.divide(tp,predicted,out=np.zeros(6),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(6),where=support>0)
    return dict(accuracy=float(tp.sum()/len(truth)),macro_f1=float(f1.mean()),macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()),per_class_f1=f1.tolist(),confusion_matrix=matrix.tolist())


def source_weights(source):
    import itertools
    values={tuple(map(int,k.split(','))) if k else ():v for k,v in source['shapley']['coalition_values'].items()}
    require(len(values)==32 and len(source['teacher_order'])==5,'Incomplete original attribution game')
    phi=[]
    for player in range(5):
        value=0.
        for size in range(5):
            for c in itertools.combinations([j for j in range(5) if j!=player],size):
                value+=math.factorial(size)*math.factorial(4-size)/120*(values[tuple(sorted((*c,player)))]-values[c])
        phi.append(value)
    np.testing.assert_allclose(phi,source['shapley']['values'],rtol=0,atol=1e-12)
    phi=np.array(phi);z=np.zeros(5) if phi.std()<=1e-12 else (phi-phi.mean())/phi.std()
    exp=np.exp(z-z.max());weights=exp/exp.sum()
    np.testing.assert_allclose(weights,source['shapley_weights'],rtol=0,atol=1e-12)
    return np.array(source['shapley_weights'],dtype=float)


def probabilities(logits,temperature):
    logits=np.asarray(logits,dtype=np.float64)
    require(logits.ndim==3 and logits.shape[0]==5 and logits.shape[2]==6 and np.isfinite(logits).all(),'Invalid five-teacher logits')
    require(temperature in (1.,4.),'Only predeclared inference temperatures are allowed')
    z=logits/temperature;z-=z.max(axis=2,keepdims=True);e=np.exp(z)
    return e/e.sum(axis=2,keepdims=True)


def predictions(logits,names,weights):
    require(len(names)==5 and len(set(names))==5,'Teacher order invalid')
    weights=np.asarray(weights,dtype=float)
    require(weights.shape==(5,) and np.isfinite(weights).all() and (weights>=0).all() and abs(weights.sum()-1)<1e-12,'Invalid original weights')
    result={'teacher-'+name:logits[i].argmax(axis=1) for i,name in enumerate(names)}
    for temperature in (1.,4.):
        p=probabilities(logits,temperature)
        result[f'uniform-T{int(temperature)}']=p.mean(axis=0).argmax(axis=1)
        result[f'shapley-T{int(temperature)}']=np.tensordot(weights,p,axes=(0,0)).argmax(axis=1)
    return result


def parity(observed,expected):
    errors={k:abs(observed[k]-expected[k]) for k in ('accuracy','macro_f1')}
    require(max(errors.values())<=TOLERANCE,f'Teacher source metric parity failed: {errors}')
    return dict(saved=expected,reproduced=observed,absolute_metric_errors=errors,tolerance=TOLERANCE)


def verify_inputs(drive):
    frozen=json.loads((HERE/'frozen-inputs.json').read_text())
    for rel,expected in frozen['files'].items():
        path=(drive/rel).resolve();require(drive in path.parents,'Frozen path escapes project');verify_file(path,expected)
    for ds,records in frozen['seed_records'].items():
        require(set(map(int,records))==set(SEEDS),'Wrong source seed set')
        for seed,path in records.items():
            source=json.loads((drive/path).read_text());require(source['model_seed']==int(seed),'Wrong source seed identity');source_weights(source)
    return frozen


def protocol(frozen):
    return dict(version=VERSION,outer_seeds=SEEDS,datasets=['uci','wisdm'],teacher_count=100,training_fits=0,
        primary_ensembles=['uniform-T1','shapley-T1'],secondary_ensembles=['uniform-T4','shapley-T4'],
        individual_prediction='Argmax of each teacher raw logits',
        ensemble='Arithmetic weighted mean of per-teacher softmax(logits/T), then argmax; first class index for exact ties',
        weights='Uniform 0.2 each or original per-seed standardized-Shapley weights, independently reconstructed from original game',
        temperature='T=1 conventional inference primary; T=4 KD-target aggregation secondary; no temperature selection',
        test_policy='Previously used fixed benchmark split; post-review descriptive evaluation; no model/weight/temperature selection',
        parity_tolerance=TOLERANCE,validation_teacher_checks=100,test_teacher_checks=100,
        research_banks=20,conditions_per_bank=9,validation_and_test_metric_sets=360,
        statistics='Mean and sample SD across ten paired seeds; descriptive only, no new significance claim',
        original_main_results_preserved=True,pilot_weight_choices_not_used=True,
        manifest_sha256=sha(HERE/'frozen-inputs.json'))


def save_npz(path,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:np.savez_compressed(stream,**arrays)
    tmp.replace(path)


def data_arrays(drive,dataset,source,config,context,root):
    cache=root/'data'/f'{dataset}.npz';info=cache.with_suffix('.json')
    identity=digest(dict(dataset=dataset,source_manifest=source['dataset_manifest'],environment=context['environment'],
        manifest_sha256=sha(HERE/'frozen-inputs.json')))
    if info.exists():
        saved=json.loads(info.read_text());require(saved['identity']==identity,'Changed data identity');verify_file(cache,saved['sha256'])
    else:
        print(f'Loading original {dataset} subject split and preprocessing.',flush=True)
        bundle=load_bundle(drive,dataset,source,config)
        arrays={f'{kind}_{part}':getattr(bundle,f'{kind}_{part}') for part in ('val','test') for kind in ('raw','features','y','subjects')}
        save_npz(cache,**arrays)
        atomic_json(info,dict(identity=identity,sha256=sha(cache),dataset_manifest=bundle.manifest(),test_inference_done=False))
    with np.load(cache,allow_pickle=False) as archive:arrays={key:archive[key].copy() for key in archive.files}
    for part,long in [('val','validation'),('test','test')]:
        y=arrays[f'y_{part}'];subjects=arrays[f'subjects_{part}'];expected=source['dataset_manifest'][long]
        require(len(y)==expected['samples'] and np.unique(subjects).tolist()==expected['subjects'],'Changed split arrays')
        require({str(i):int((y==i).sum()) for i in range(6)}==expected['class_counts'],'Class counts differ')
    return arrays,[cache,info]


def teacher_model(drive,frozen,dataset,seed,name):
    import tensorflow as tf
    prefix=Path(frozen['seed_records'][dataset][str(seed)]).with_suffix('')
    path=drive/prefix/'models'/f'teacher-{name}.keras'
    verify_file(path,frozen['files'][path.relative_to(drive).as_posix()])
    tf.keras.backend.clear_session()
    return tf.keras.models.load_model(path,compile=False,safe_mode=True)


def validate_preflight(path,npz,context,source,arrays):
    require(path.is_file() and npz.is_file(),'Missing validation preflight')
    r=json.loads(path.read_text());require(r['context']==context and r['identity']==digest(context) and r['passed'],'Changed preflight identity')
    verify_file(npz,r['logits_sha256'])
    with np.load(npz,allow_pickle=False) as cache:
        require(set(cache.files)=={'val_logits','y_val','subjects_val'},'Unexpected validation cache arrays')
        logits=cache['val_logits'].copy()
        np.testing.assert_array_equal(cache['y_val'],arrays['y_val']);np.testing.assert_array_equal(cache['subjects_val'],arrays['subjects_val'])
    require(logits.shape==(5,len(arrays['y_val']),6) and np.isfinite(logits).all(),'Invalid validation logits')
    for i,name in enumerate(source['teacher_order']):
        expected=parity(metrics(arrays['y_val'],logits[i].argmax(axis=1)),source['teachers'][name]['validation'])
        require(r['teacher_validation_parity'][name]==expected,'Changed validation parity evidence')
    return logits,r


def preflight(drive,frozen,dataset,seed,source,arrays,context,root):
    path=root/dataset/f'seed-{seed}'/'preflight.json';npz=path.with_name('validation-logits.npz')
    if not path.exists():
        from makd_har.models import teacher_registry
        from makd_har.training import predict_logits
        logits=[];checks={}
        for name in source['teacher_order']:
            kind=teacher_registry()[name].input_kind;require(kind==source['teachers'][name]['input_kind'],'Teacher input representation differs')
            model=teacher_model(drive,frozen,dataset,seed,name)
            values=predict_logits(model,arrays[f'{kind}_val'],128);require(values.shape==(len(arrays['y_val']),6),'Invalid teacher output shape')
            checks[name]=parity(metrics(arrays['y_val'],values.argmax(axis=1)),source['teachers'][name]['validation']);logits.append(values);del model
        save_npz(npz,val_logits=np.stack(logits),y_val=arrays['y_val'],subjects_val=arrays['subjects_val'])
        atomic_json(path,dict(passed=True,context=context,identity=digest(context),logits_sha256=sha(npz),
            teacher_validation_parity=checks,test_evaluated=False))
    validate_preflight(path,npz,context,source,arrays)
    print(f'PREFLIGHT PASS {dataset} seed {seed}: all five teachers match original validation metrics.',flush=True)
    return [path,npz]


def evaluate(drive,frozen,dataset,seed,source,arrays,context,root,output,mode):
    from makd_har.models import teacher_registry
    from makd_har.training import predict_logits
    pref=root/dataset/f'seed-{seed}'/'preflight.json'
    val_logits,_=validate_preflight(pref,pref.with_name('validation-logits.npz'),context,source,arrays)
    task=dict(context=context,mode=mode,preflight_sha256=sha(pref),weights=source_weights(source).tolist())
    path=output/dataset/f'seed-{seed}'/'evaluation.json';npz=path.with_name('predictions-and-logits.npz')
    if path.exists():
        r=json.loads(path.read_text());require(r['task']==task and r['identity']==digest(task) and r['status']=='complete','Unsafe cached evaluation');verify_file(npz,r['predictions_sha256'])
        with np.load(npz,allow_pickle=False) as archive:results={key:archive[key].copy() for key in archive.files}
    else:
        logits=[]
        for name in source['teacher_order']:
            model=teacher_model(drive,frozen,dataset,seed,name);kind=teacher_registry()[name].input_kind
            values=predict_logits(model,arrays[f'{kind}_test'],128)
            parity(metrics(arrays['y_test'],values.argmax(axis=1)),source['teachers'][name]['test'])
            logits.append(values);del model
        results=dict(val_logits=val_logits,test_logits=np.stack(logits),y_val=arrays['y_val'],y_test=arrays['y_test'],
            subjects_val=arrays['subjects_val'],subjects_test=arrays['subjects_test'])
        reports={}
        for part in ('val','test'):
            pred=predictions(results[part+'_logits'],source['teacher_order'],task['weights'])
            reports[part]={name:metrics(results['y_'+part],values) for name,values in pred.items()}
            results.update({f'{part}__{name}':values for name,values in pred.items()})
        save_npz(npz,**results)
        checks={name:parity(reports['test']['teacher-'+name],source['teachers'][name]['test']) for name in source['teacher_order']}
        r=dict(status='complete',identity=digest(task),task=task,metrics=reports,test_teacher_parity=checks,
               predictions_sha256=sha(npz),test_evaluated=True,training_fits=0)
        atomic_json(path,r)
    np.testing.assert_array_equal(results['val_logits'],val_logits)
    for part in ('val','test'):
        for key in ('y','subjects'):np.testing.assert_array_equal(results[f'{key}_{part}'],arrays[f'{key}_{part}'])
        recomputed=predictions(results[part+'_logits'],source['teacher_order'],task['weights'])
        require(set(recomputed)==set(r['metrics'][part]),'Incomplete nine-condition metrics')
        for name,values in recomputed.items():
            np.testing.assert_array_equal(results[f'{part}__{name}'],values)
            require(metrics(results['y_'+part],values)==r['metrics'][part][name],'Saved metric mismatch')
    require(r['test_teacher_parity']=={n:parity(r['metrics']['test']['teacher-'+n],source['teachers'][n]['test']) for n in source['teacher_order']},'Changed test parity evidence')
    print(f'EVALUATED {dataset} seed {seed}: five teachers + four fixed ensembles; all test teacher parity checks pass.',flush=True)
    return [path,npz]


def records(paths,drive):return [dict(path=p.relative_to(drive).as_posix(),sha256=sha(p)) for p in paths]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--drive-root',type=Path,default=PACKAGE.parent)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--preflight-only',action='store_true');p.add_argument('--smoke',action='store_true')
    args=p.parse_args();drive=args.drive_root.resolve();frozen=verify_inputs(drive);spec=protocol(frozen)
    print(f'Frozen source checks passed: {len(frozen["files"])} files; training fits = 0.',flush=True)
    if args.verify_only:return
    os.environ['TF_DETERMINISTIC_OPS']='1';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    import tensorflow as tf
    from makd_har.config import ExperimentConfig
    tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
    devices=tf.config.list_physical_devices('GPU');require(devices,'Use a GPU for the saved-model inference pass')
    env=json.loads(json.dumps(dict(python=sys.version,tensorflow=tf.__version__,keras=tf.keras.__version__,numpy=np.__version__,
        platform=platform.platform(),deterministic_operations=True,tensor_float_32_execution=False,
        gpu=[tf.config.experimental.get_device_details(d) for d in devices])))
    root=drive/'results/revision-2/teacher-ensemble';ensure_same(root/'experiment-protocol.json',spec);ensure_same(root/'environment.json',env)
    if not args.preflight_only:
        gate=json.loads((root/'preflight-completed.json').read_text())
        require(gate['passed'] and gate['teacher_checks']==100 and len(gate['files'])==44 and gate['specification_sha256']==digest(spec) and gate['environment_sha256']==digest(env),'All-teacher preflight gate differs')
        for item in gate['files']:verify_file(drive/item['path'],item['sha256'])
        if not args.smoke:
            marker=json.loads((drive/'results/revision-2/smoke-teacher-ensemble/completed.json').read_text())
            require(marker['status']=='complete' and marker['mode']=='smoke' and marker['banks']==2 and marker['specification_sha256']==digest(spec) and marker['environment_sha256']==digest(env),'Smoke not verified')
            for item in marker['files']:verify_file(drive/item['path'],item['sha256'])
    preflight_paths=[];outputs=[]
    mode='smoke' if args.smoke else 'research';output=drive/'results/revision-2/smoke-teacher-ensemble' if args.smoke else root
    for dataset in ('uci','wisdm'):
        source=json.loads((drive/frozen['seed_records'][dataset]['11']).read_text());config=ExperimentConfig(**source['config']);config.validate()
        arrays,paths=data_arrays(drive,dataset,source,config,dict(environment=env),root);preflight_paths.extend(paths)
        for seed in (SEEDS if args.preflight_only or not args.smoke else [11]):
            current=json.loads((drive/frozen['seed_records'][dataset][str(seed)]).read_text())
            require(current['dataset_manifest']==source['dataset_manifest'] and current['config']==source['config'],'Source split/config differs across seeds')
            context=dict(specification=spec,environment=env,dataset=dataset,outer_seed=seed,
                source_record_sha256=sha(drive/frozen['seed_records'][dataset][str(seed)]),teacher_order=current['teacher_order'],
                dataset_manifest=current['dataset_manifest'],data_sha256=sha(root/'data'/f'{dataset}.npz'))
            if args.preflight_only:preflight_paths.extend(preflight(drive,frozen,dataset,seed,current,arrays,context,root))
            else:outputs.extend(evaluate(drive,frozen,dataset,seed,current,arrays,context,root,output,mode))
    if args.preflight_only:
        require(len(preflight_paths)==44,'Preflight incomplete')
        atomic_json(root/'preflight-completed.json',dict(passed=True,teacher_checks=100,banks=20,test_evaluated=False,
            specification_sha256=digest(spec),environment_sha256=digest(env),files=records(preflight_paths,drive)))
        print('PREFLIGHT COMPLETE: all 100 teachers match original validation metrics; no test inference yet.',flush=True)
    else:
        expected=2 if args.smoke else 20;require(len(outputs)==expected*2,'Incomplete evaluations')
        atomic_json(output/'completed.json',dict(status='complete',mode=mode,banks=expected,teacher_evaluations=expected*5,
            ensemble_evaluations=expected*4,metric_sets=expected*18,training_fits=0,test_evaluated=True,
            specification_sha256=digest(spec),environment_sha256=digest(env),files=records(outputs,drive)))
        print(f'COMPLETE {mode}: {expected} banks, {expected*5} teachers, {expected*4} ensembles; ZERO training fits.',flush=True)


if __name__=='__main__':main()
