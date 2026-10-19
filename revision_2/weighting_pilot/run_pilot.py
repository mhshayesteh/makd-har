"""Four-role training-only weighting pilot. Never evaluates original validation or test data."""
from __future__ import annotations
import argparse
import dataclasses
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
sys.path.insert(0,str(PACKAGE))
sys.path.insert(0,str(PACKAGE/'pipeline_snapshot'))
from run_coalition_noise import sha,digest,atomic_json,ensure_same,verify_file
from pilot_data import ROLES,load_roles,read_training,validate_groups,require,source_record

VERSION='revision2-training-only-weighting-v1'
SEEDS=[11,23,37]
CANDIDATES=['uniform','standardized','raw-tau-0p001','raw-tau-0p01','raw-tau-0p1']


def weights_for(phi):
    import numpy as np
    from makd_har.shapley import shapley_to_weights
    phi=np.asarray(phi,dtype=float)
    require(phi.shape==(5,) and np.isfinite(phi).all(),'Invalid Shapley values')
    def raw(tau):
        z=phi/tau;z-=z.max();w=np.exp(z);return (w/w.sum()).tolist()
    return dict(uniform=[.2]*5,standardized=shapley_to_weights(phi,1.).tolist(),
        **{'raw-tau-0p001':raw(.001),'raw-tau-0p01':raw(.01),'raw-tau-0p1':raw(.1)})


def select_candidate(rows):
    require(len(rows)==3 and all(list(row)==CANDIDATES for row in rows),'Incomplete pilot candidate set')
    means={name:sum(row[name] for row in rows)/3 for name in CANDIDATES}
    require(all(math.isfinite(v) and 0<=v<=1 for v in means.values()),'Invalid selection scores')
    winner=max(CANDIDATES,key=lambda name:(round(means[name],12),-CANDIDATES.index(name)))
    return dict(selected_candidate=winner,selection_macro_f1_means=means,
        selection_rule='Maximum three-seed mean selection macro-F1 rounded to 12 decimals; exact rounded ties use fixed candidate order, uniform first',
        exploratory_only=True,test_evaluated=False,original_validation_evaluated=False,
        limitation='Selected development score is optimistic for generalization; no independent test or main-table replacement')


def verify_inputs(drive):
    manifest=json.loads((HERE/'frozen-inputs.json').read_text())
    for rel,expected in manifest['files'].items():
        path=(drive/rel).resolve();require(drive in path.parents,'Frozen path escapes project');verify_file(path,expected)
    for dataset in ('uci','wisdm'):
        split=json.loads((HERE/f'{dataset}-split.json').read_text())
        require(split==manifest['splits'][dataset],'Frozen split differs')
    return manifest


def protocol(frozen):
    return dict(version=VERSION,outer_seeds=SEEDS,datasets=['uci','wisdm'],splits=frozen['splits'],
        roles=dict(fit='Teacher and student gradient fitting; scalers fitted only here',
                   stop='Teacher val_loss and student macro-F1 checkpoint selection with patience ten',
                   value='Restored coalition checkpoint macro-F1, all 32 utilities; never checkpoint selection',
                   select='Candidate restored-checkpoint macro-F1; chooses weighting rule after all seeds'),
        candidates=CANDIDATES,weight_temperatures=[.001,.01,.1],kd_temperature=4.,hard_label_weight=.5,
        teacher_epochs=60,coalition_epochs=60,student_epochs=60,patience=10,
        teacher_seed='outer*100 + teacher index',coalition_seed='outer*100000+17 common across all coalitions',
        candidate_seed='outer*1000000+7 common across all candidates',
        empty_coalition='Supervised student; utility is not fixed to zero',
        selection='Three-seed mean select-role macro-F1 rounded to 12 decimals; candidate list is tie priority, uniform first',
        inference='Descriptive training-only pilot; no significance claim from three seeds; no original validation/test evaluation',
        research_teacher_fits=30,research_coalition_fits=192,research_candidate_fits=30,
        source_manifest_sha256=sha(HERE/'frozen-inputs.json'))


def history_check(report,task):
    import numpy as np
    hist=report['history'];teacher=task['kind']=='teacher'
    values=hist['val_loss' if teacher else 'val_macro_f1']
    require(len(values)==report['epochs_completed'] and 1<=len(values)<=task['epochs'],'Invalid history length')
    require(all(np.isfinite(v) for key in hist for v in hist[key]) and all(len(v)==len(values) for v in hist.values()),'Invalid history values')
    best=math.inf if teacher else -math.inf;stale=0;best_epoch=0
    for epoch,value in enumerate(values,1):
        improved=value<best if teacher else value>best+1e-7
        if improved:best=value;stale=0;best_epoch=epoch
        else:stale+=1
        require(stale<10 or epoch==len(values),'Continued after early stopping')
    require(len(values)==task['epochs'] or stale==10,'Truncated training history')
    require(report['best_epoch']==best_epoch,'Incorrect checkpoint epoch')
    if not teacher:require(abs(best-report['stop_metrics']['macro_f1'])<=1e-4,'Selected stop checkpoint differs')
    for key in ('stop_metrics','evaluation_metrics'):
        if key not in report:continue
        m=report[key]
        require(len(m['per_class_f1'])==6 and abs(sum(m['per_class_f1'])/6-m['macro_f1'])<1e-12,'Class metric mismatch')
        require(all(math.isfinite(m[k]) and 0<=m[k]<=1 for k in ('macro_f1','accuracy','macro_precision','macro_recall')),'Invalid metric')


def checked(path,task):
    import numpy as np
    from makd_har.training import classification_metrics_from_labels
    if not path.exists():return None
    report=json.loads(path.read_text())
    require(report.get('status')=='complete' and report.get('task')==task and report.get('identity')==digest(task),'Unsafe cached fit')
    require(report.get('test_evaluated') is False and report.get('original_validation_evaluated') is False,'Forbidden evaluation in cached result')
    history_check(report,task)
    for key in ('model','predictions'):
        artifact=(path.parent/report[key+'_file']).resolve();require(path.parent.resolve() in artifact.parents,'Artifact path escape')
        verify_file(artifact,report[key+'_sha256'])
    with np.load(path.parent/report['predictions_file'],allow_pickle=False) as arrays:
        expected={'stop_y','stop_predictions'}
        if task['kind']!='teacher':expected|={'evaluation_y','evaluation_predictions'}
        require(set(arrays.files)==expected,'Unexpected prediction partitions')
        for part in ('stop','evaluation'):
            if part+'_metrics' not in report:continue
            metrics=classification_metrics_from_labels(arrays[part+'_y'],arrays[part+'_predictions'])
            for key in metrics:np.testing.assert_allclose(metrics[key],report[part+'_metrics'][key],rtol=0,atol=1e-12)
    return report


def save_fit(model,history,task,path,roles,input_kind,evaluation_role,started):
    import numpy as np
    from makd_har.training import predict_logits,classification_metrics_from_labels
    teacher=task['kind']=='teacher';values=history['val_loss' if teacher else 'val_macro_f1']
    best=math.inf if teacher else -math.inf;best_epoch=0
    for epoch,value in enumerate(values,1):
        if (value<best if teacher else value>best+1e-7):best=value;best_epoch=epoch
    arrays={};metrics={}
    for label,role in [('stop','stop')]+([] if teacher else [('evaluation',evaluation_role)]):
        data=roles[role];pred=predict_logits(model,data[input_kind],128).argmax(axis=1)
        arrays[label+'_y']=data['y'];arrays[label+'_predictions']=pred
        metrics[label+'_metrics']=classification_metrics_from_labels(data['y'],pred)
    report=dict(status='complete',identity=digest(task),task=task,history=history,epochs_completed=len(values),
        best_epoch=best_epoch,elapsed_seconds=time.perf_counter()-started,test_evaluated=False,
        original_validation_evaluated=False,**metrics)
    history_check(report,task)
    path.parent.mkdir(parents=True,exist_ok=True);model_path=path.with_suffix('.keras');model.save(model_path,include_optimizer=False)
    pred_path=path.with_suffix('.npz')
    with pred_path.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
    pred_path.with_suffix('.npz.tmp').replace(pred_path)
    report.update(model_file=model_path.name,model_sha256=sha(model_path),predictions_file=pred_path.name,predictions_sha256=sha(pred_path))
    atomic_json(path,report)
    print(f"SAVED {task['context']['dataset']} seed {task['context']['outer_seed']} {task['name']}: {len(values)} epochs, {report['elapsed_seconds']:.1f}s",flush=True)
    return report


def train_teacher(roles,config,task,path):
    import tensorflow as tf
    from makd_har.models import teacher_registry
    from makd_har.reproducibility import set_global_seed
    from makd_har.training import train_supervised
    spec=teacher_registry()[task['name']];kind=spec.input_kind
    # Registry uses 'features' or 'raw', matching these role arrays.
    tf.keras.backend.clear_session();set_global_seed(task['seed'])
    model=spec.builder(tuple(roles['fit'][kind].shape[1:]),6);started=time.perf_counter()
    model,history=train_supervised(model,roles['fit'][kind],roles['fit']['y'],roles['stop'][kind],roles['stop']['y'],
        seed=task['seed'],epochs=task['epochs'],batch_size=128,learning_rate=.001,patience=10,verbose=0)
    return save_fit(model,history,task,path,roles,kind,None,started)


def teacher_bank(roles,config,context,seed_dir,epochs):
    import numpy as np
    import tensorflow as tf
    from makd_har.models import teacher_registry
    from makd_har.training import predict_logits,logits_to_probabilities
    reports={};paths=[]
    for index,name in enumerate(config.teacher_names):
        task=dict(context=context,kind='teacher',name=name,seed=context['outer_seed']*100+index,epochs=epochs)
        path=seed_dir/'teachers'/f'{name}.json';report=checked(path,task)
        if report is None:
            print(f"FITTING {context['dataset']} seed {context['outer_seed']} teacher {name}",flush=True)
            report=train_teacher(roles,config,task,path)
        reports[name]=report;paths.append(path)
    identity=digest(dict(context=context,models={name:r['model_sha256'] for name,r in reports.items()}))
    cache=seed_dir/'teacher-probabilities.npz';info_path=cache.with_suffix('.json')
    if info_path.exists():
        info=json.loads(info_path.read_text());require(info['identity']==identity,'Changed teacher cache context');verify_file(cache,info['sha256'])
    else:
        arrays={}
        for name,report in reports.items():
            tf.keras.backend.clear_session()
            model=tf.keras.models.load_model(seed_dir/'teachers'/report['model_file'],compile=False,safe_mode=True)
            kind=teacher_registry()[name].input_kind
            for role in ('fit','stop'):
                logits=predict_logits(model,roles[role][kind],128)
                arrays[f'{name}_{role}']=logits_to_probabilities(logits,4.)
            # Loading the saved checkpoint must reproduce its recorded stop labels exactly.
            loaded=predict_logits(model,roles['stop'][kind],128).argmax(axis=1)
            with np.load(seed_dir/'teachers'/report['predictions_file'],allow_pickle=False) as original:
                np.testing.assert_array_equal(loaded,original['stop_predictions'])
            del model
        with cache.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
        cache.with_suffix('.npz.tmp').replace(cache);atomic_json(info_path,dict(identity=identity,sha256=sha(cache),teacher_checkpoint_parity='Exact saved stop-label sequence for all five teachers'))
    with np.load(cache,allow_pickle=False) as arrays:
        bank={name:{role:arrays[f'{name}_{role}'].copy() for role in ('fit','stop')} for name in config.teacher_names}
    for name in bank:
        for role,values in bank[name].items():
            require(values.shape==(len(roles[role]['y']),6) and np.isfinite(values).all() and (values>=0).all()
                and np.allclose(values.sum(axis=1),1,rtol=0,atol=1e-5),'Invalid teacher probabilities')
    return bank,reports,paths,[cache,info_path]


def train_student(roles,config,task,path,bank):
    import tensorflow as tf
    from makd_har.models import build_student
    from makd_har.reproducibility import set_global_seed
    from makd_har.training import aggregate_teacher_probabilities,train_distilled_student
    names=task['teachers'];weights=task['weights']
    targets={r:aggregate_teacher_probabilities([bank[n][r] for n in names],weights) if names else None for r in ('fit','stop')}
    tf.keras.backend.clear_session();set_global_seed(task['seed'])
    model=build_student(tuple(roles['fit']['raw'].shape[1:]),6);started=time.perf_counter()
    model,history=train_distilled_student(model,roles['fit']['raw'],roles['fit']['y'],roles['stop']['raw'],roles['stop']['y'],
        targets['fit'],targets['stop'],seed=task['seed'],epochs=task['epochs'],batch_size=128,
        learning_rate=.001,patience=10,temperature=4.,hard_label_weight=.5)
    return save_fit(model,history,task,path,roles,'raw',task['evaluation_role'],started)


def cached_roles(drive,dataset,frozen,environment,root):
    import numpy as np
    split=frozen['splits'][dataset];context=dict(dataset=dataset,split=split,environment=environment,
        frozen_manifest_sha256=sha(HERE/'frozen-inputs.json'))
    cache=root/dataset/'inner-data.npz';info_path=cache.with_suffix('.json')
    if info_path.exists():
        info=json.loads(info_path.read_text());require(info['context']==context,'Changed inner data context');verify_file(cache,info['sha256'])
    else:
        print(f'Loading {dataset}: original training subjects only; fitting fresh inner scalers.',flush=True)
        roles,scalers=load_roles(drive,dataset,split);cache.parent.mkdir(parents=True,exist_ok=True)
        arrays={f'{role}_{key}':value for role,part in roles.items() for key,value in part.items()}
        arrays.update({'scaler_'+k:v for k,v in scalers.items()})
        with cache.with_suffix('.npz.tmp').open('wb') as stream:np.savez_compressed(stream,**arrays)
        cache.with_suffix('.npz.tmp').replace(cache)
        info=dict(context=context,sha256=sha(cache),scalers_fitted_on='fit role only',test_evaluated=False)
        atomic_json(info_path,info)
    with np.load(cache,allow_pickle=False) as arrays:
        roles={r:{k:arrays[f'{r}_{k}'].copy() for k in ('raw','features','y','subjects')} for r in ROLES}
    y=np.concatenate([roles[r]['y'] for r in ROLES]);s=np.concatenate([roles[r]['subjects'] for r in ROLES])
    validate_groups(y,s,split)
    return roles,[cache,info_path]


def file_record(path,drive):return dict(path=path.relative_to(drive).as_posix(),sha256=sha(path))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drive-root',type=Path,default=PACKAGE.parent)
    p.add_argument('--verify-only',action='store_true');p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--smoke',action='store_true');args=p.parse_args();drive=args.drive_root.resolve()
    frozen=verify_inputs(drive);spec=protocol(frozen)
    print(f'Frozen inputs passed: {len(frozen["files"])} files. No original validation/test evaluation.',flush=True)
    if args.verify_only:return
    os.environ['TF_DETERMINISTIC_OPS']='1';os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
    import numpy as np
    import tensorflow as tf
    import sklearn
    from makd_har.config import ExperimentConfig
    from makd_har.shapley import exact_shapley,weight_diagnostics
    tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
    devices=tf.config.list_physical_devices('GPU');require(devices,'GPU required for this training pilot')
    env=json.loads(json.dumps(dict(python=sys.version,platform=platform.platform(),tensorflow=tf.__version__,keras=tf.keras.__version__,
        numpy=np.__version__,sklearn=sklearn.__version__,deterministic_operations=True,tensor_float_32_execution=False,
        gpu=[tf.config.experimental.get_device_details(d) for d in devices])))
    root=drive/'results/revision-2/weighting-pilot';ensure_same(root/'experiment-protocol.json',spec);ensure_same(root/'environment.json',env)
    if not args.preflight_only:
        gate=json.loads((root/'preflight-completed.json').read_text())
        require(gate['passed'] and gate['specification_sha256']==digest(spec) and gate['environment_sha256']==digest(env),'Preflight context mismatch')
        require(len(gate['files'])==4,'Incomplete preflight evidence')
        for item in gate['files']:verify_file(drive/item['path'],item['sha256'])
    if not args.smoke and not args.preflight_only:
        smoke_root=drive/'results/revision-2/smoke-weighting-pilot'
        smoke=json.loads((smoke_root/'completed.json').read_text())
        require(smoke['status']=='complete' and smoke['fits']==24 and smoke['mode']=='smoke'
            and smoke['specification_sha256']==digest(spec) and smoke['environment_sha256']==digest(env),'Smoke gate not passed')
        for item in smoke['datasets']:
            verify_file(drive/item['path'],item['sha256'])
            for artifact in json.loads((drive/item['path']).read_text())['files']:
                verify_file(drive/artifact['path'],artifact['sha256'])
    preflight_files=[];dataset_markers=[]
    for dataset in ('uci','wisdm'):
        roles,datafiles=cached_roles(drive,dataset,frozen,env,root);preflight_files.extend(datafiles)
        print(f'ROLE COVERAGE PASS {dataset}: four disjoint roles, six classes, >=2 subjects/class in each.',flush=True)
        if args.preflight_only:continue
        output=drive/'results/revision-2/smoke-weighting-pilot' if args.smoke else root
        mode='smoke' if args.smoke else 'research';epochs=2 if args.smoke else 60
        seeds=[11] if args.smoke else SEEDS;paths=[];fits=0;selection_rows=[]
        config=ExperimentConfig(**source_record(drive,dataset)['config']);config.validate()
        require(config.learning_rate==.001 and config.batch_size==128 and config.temperature==4. and config.hard_label_weight==.5 and config.patience==10,'Unexpected source optimization settings')
        for seed in seeds:
            context=dict(specification=spec,environment=env,dataset=dataset,outer_seed=seed,mode=mode,
                source_config=dataclasses.asdict(config),data_sha256=sha(root/dataset/'inner-data.npz'))
            # JSON normalizes tuples exactly as persisted task records do.
            context=json.loads(json.dumps(context))
            seed_dir=output/dataset/f'seed-{seed}'
            bank,teachers,teacher_paths,bank_paths=teacher_bank(roles,config,context,seed_dir,epochs)
            fits+=5;paths.extend(teacher_paths);paths.extend(bank_paths)
            bank_identity=digest({name:r['model_sha256'] for name,r in teachers.items()})
            coalition_reports={}
            def utility(coalition):
                nonlocal fits
                key='empty' if not coalition else '-'.join(map(str,coalition));path=seed_dir/'coalitions'/f'{key}.json'
                names=[config.teacher_names[i] for i in coalition]
                task=dict(context=context,kind='coalition',name=key,teachers=names,weights=[1/len(names)]*len(names) if names else None,
                    seed=seed*100000+17,epochs=epochs,evaluation_role='value',teacher_models_sha256=bank_identity)
                report=checked(path,task)
                if report is None:
                    print(f'FITTING {dataset} seed {seed} coalition {key}',flush=True)
                    report=train_student(roles,config,task,path,bank)
                coalition_reports[coalition]=report;paths.append(path);fits+=1
                return report['evaluation_metrics']['macro_f1']
            if args.smoke:
                # Exercise supervised and full KD paths. Smoke values are not research attributions.
                utility(());utility((0,1,2,3,4))
                # Fixed synthetic vector exercises nonuniform numerical paths without research valuation.
                phi=[-.002,-.001,0.,.001,.003];game=None
            else:
                game=exact_shapley(5,utility);phi=game.values.tolist()
                require(game.evaluations==32 and abs(sum(phi)-(game.coalition_values[(0,1,2,3,4)]-game.coalition_values[()]))<1e-10,'Shapley efficiency check failed')
            weights=weights_for(phi);candidate_reports={}
            for name in CANDIDATES:
                task=dict(context=context,kind='candidate',name=name,teachers=list(config.teacher_names),weights=weights[name],
                    seed=seed*1000000+7,epochs=epochs,evaluation_role='select',teacher_models_sha256=bank_identity,
                    valuation_game_sha256=None if game is None else digest({','.join(map(str,k)):v for k,v in game.coalition_values.items()}))
                path=seed_dir/'candidates'/f'{name}.json';report=checked(path,task)
                if report is None:
                    print(f'FITTING {dataset} seed {seed} candidate {name}',flush=True)
                    report=train_student(roles,config,task,path,bank)
                candidate_reports[name]=report;paths.append(path);fits+=1
            summary=dict(context=context,teacher_order=list(config.teacher_names),
                shapley=None if game is None else game.to_dict(),weights=weights,
                weight_diagnostics={name:weight_diagnostics(np.array(w)) for name,w in weights.items()},
                candidates={name:dict(selection_metrics=r['evaluation_metrics'],stop_metrics=r['stop_metrics']) for name,r in candidate_reports.items()},
                test_evaluated=False,original_validation_evaluated=False,smoke_values_not_research=args.smoke)
            summary_path=seed_dir/'summary.json';atomic_json(summary_path,summary);paths.append(summary_path)
            selection_rows.append({name:r['evaluation_metrics']['macro_f1'] for name,r in candidate_reports.items()})
            print(f'BANK COMPLETE {dataset} {seed}: five candidates retained; no test evaluation.',flush=True)
        if not args.smoke:
            selection=select_candidate(selection_rows);selection_path=output/dataset/'selection.json';atomic_json(selection_path,selection);paths.append(selection_path)
        files=[]
        for path in paths:
            files.append(file_record(path,drive))
            if path.suffix=='.json':
                record=json.loads(path.read_text())
                if 'model_file' in record:
                    files.extend(file_record(path.parent/record[key+'_file'],drive) for key in ('model','predictions'))
        require(fits==(12 if args.smoke else 126),'Incomplete dataset fits')
        marker=output/dataset/'completed.json';atomic_json(marker,dict(status='complete',mode=mode,dataset=dataset,
            fits=fits,teacher_fits=5*len(seeds),student_fits=fits-5*len(seeds),files=files,
            specification_sha256=digest(spec),environment_sha256=digest(env),test_evaluated=False,original_validation_evaluated=False))
        dataset_markers.append(marker)
    if args.preflight_only:
        atomic_json(root/'preflight-completed.json',dict(passed=True,specification_sha256=digest(spec),environment_sha256=digest(env),
            files=[file_record(path,drive) for path in preflight_files],test_evaluated=False))
        print('PREFLIGHT COMPLETE: both datasets; original validation/test excluded; inner scalers saved.',flush=True)
    else:
        total=sum(json.loads(path.read_text())['fits'] for path in dataset_markers)
        require(total==(24 if args.smoke else 252),'Incomplete total fits')
        atomic_json(output/'completed.json',dict(status='complete',mode=mode,fits=total,
            teacher_fits=10 if args.smoke else 30,student_fits=14 if args.smoke else 222,
            datasets=[file_record(path,drive) for path in dataset_markers],specification_sha256=digest(spec),environment_sha256=digest(env),
            test_evaluated=False,original_validation_evaluated=False))
        print(f'COMPLETE {mode}: {total} fits; no original validation/test metrics computed.',flush=True)


if __name__=='__main__':main()
