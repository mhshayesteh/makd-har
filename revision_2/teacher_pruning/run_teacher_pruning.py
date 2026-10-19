"""Fixed-bank teacher-removal study; all conditions and controls are predeclared."""
from __future__ import annotations
import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
sys.path.insert(0,str(PACKAGE))
from run_coalition_noise import sha,digest,atomic_json,ensure_same,verify_file,load_bundle,frozen_probability_bank

VERSION='revision2-teacher-pruning-v1'
SEEDS=[11,23,37,53,71,89,107,131,167,197]


def require(condition,message):
    if not condition:raise RuntimeError(message)


def removal_choice(source):
    values=source['shapley']['values'];names=source['teacher_order']
    require(len(values)==len(names)==5 and len(set(names))==5,'Expected five named teachers')
    require(all(isinstance(v,(int,float)) and math.isfinite(v) for v in values),'Invalid raw Shapley vector')
    return min(range(5),key=lambda i:(values[i],i))


def retained_indices(removed):
    require(removed is None or (type(removed) is int and 0<=removed<5),'Invalid removal index')
    return [i for i in range(5) if i!=removed]


def verify_inputs(drive):
    manifest=json.loads((HERE/'frozen-inputs.json').read_text())
    for rel,expected in manifest['files'].items():
        path=(drive/rel).resolve();require(drive in path.parents,'Frozen path escapes project')
        verify_file(path,expected)
    for dataset,records in manifest['seed_records'].items():
        require(list(map(int,records))==SEEDS or set(map(int,records))==set(SEEDS),'Wrong seed set')
        for seed,path in records.items():
            source=json.loads((drive/path).read_text());choice=removal_choice(source)
            require(source['model_seed']==int(seed),'Source model seed mismatch')
            require(source['config']['student_epochs']==60,'Unexpected final-student budget')
            require(source['methods']['uniform_teachers']['seed']==int(seed)*1000000+7,'Source final seed rule differs')
            require(choice==manifest['choices'][dataset][seed]['removed_index'],'Frozen choice mismatch')
    return manifest


def history_check(report,epochs,patience=10):
    losses=report['history']['train_loss'];scores=report['history']['val_macro_f1']
    require(len(losses)==len(scores)==report['epochs_completed'] and 1<=len(scores)<=epochs,'Invalid history length')
    require(all(math.isfinite(v) for v in losses),'Nonfinite training loss')
    best=-math.inf;stale=0;best_epoch=0
    for epoch,score in enumerate(scores,1):
        require(math.isfinite(score) and 0<=score<=1,'Invalid validation utility')
        if score>best+1e-7:best=score;stale=0;best_epoch=epoch
        else:stale+=1
        if stale>=patience:require(epoch==len(scores),'Training continued after early stopping')
    require(len(scores)==epochs or stale==patience,'Prematurely truncated history')
    require(abs(best-report['validation_metrics']['macro_f1'])<=1e-4,'Checkpoint restoration differs from history')
    require(report['best_epoch']==best_epoch,'Incorrect selected checkpoint epoch')
    for partition in ('validation_metrics','test_metrics'):
        metrics=report[partition]
        for key in ('accuracy','macro_f1','macro_precision','macro_recall'):
            require(math.isfinite(metrics[key]) and 0<=metrics[key]<=1,'Invalid metric')
        require(len(metrics['per_class_f1'])==6 and all(math.isfinite(x) and 0<=x<=1 for x in metrics['per_class_f1']),
                'Invalid class metrics')
        require(abs(sum(metrics['per_class_f1'])/6-metrics['macro_f1'])<1e-12,'Macro-F1/class metrics disagree')


def checked_fit(path,task):
    if not path.exists():return None
    report=json.loads(path.read_text())
    require(report.get('identity')==digest(task) and report.get('task')==task and report.get('status')=='complete',
            f'Unsafe cached fit: {path}')
    require(report.get('test_evaluated') is True,'Missing predeclared test evaluation')
    history_check(report,task['epochs'],task['context']['config']['patience'])
    for name in ('model','predictions'):
        artifact=(path.parent/report[f'{name}_file']).resolve()
        require(path.parent.resolve() in artifact.parents,'Artifact path escapes result directory')
        verify_file(artifact,report[f'{name}_sha256'])
    return report


def validate_probability_bank(bank,bundle,names):
    import numpy as np
    require(set(bank)==set(names),'Wrong teacher probability bank')
    for name in names:
        for part,count in (('train',len(bundle.y_train)),('val',len(bundle.y_val))):
            values=bank[name][part]
            require(values.shape==(count,len(bundle.class_names)),'Probability dimensions differ from split')
            require(np.isfinite(values).all() and not (values<0).any(),'Invalid teacher targets')
            require(np.allclose(values.sum(axis=1),1,rtol=0,atol=1e-5),'Teacher targets are not normalized')


def validate_parity(parity,source):
    require(set(parity)==set(source['teacher_order']),'Incomplete teacher parity checks')
    for name in source['teacher_order']:
        p=parity[name]
        require(p['saved']==source['teachers'][name]['validation'] and p['tolerance']==1e-4,'Source or tolerance changed')
        for metric in ('accuracy','macro_f1'):
            error=abs(p['saved'][metric]-p['reproduced'][metric])
            require(error==p['absolute_metric_errors'][metric] and error<=1e-4,'Teacher validation parity failed')


def paired_summary(reports,choice):
    require(set(reports)=={'full',*[f'remove-{i}' for i in range(5)]},'Incomplete removal outcomes')
    summary={}
    for part in ('validation','test'):
        values=[reports[f'remove-{i}'][part+'_metrics']['macro_f1'] for i in range(5)]
        chosen=values[choice];random=sum(values)/5;full=reports['full'][part+'_metrics']['macro_f1']
        summary[part]=dict(lowest_phi_removal=chosen,expected_random_removal=random,full_uniform=full,
            selected_minus_random=chosen-random,selected_minus_full=chosen-full,random_minus_full=random-full,
            all_five_removals=values)
    return summary


def train_fit(bundle,config,task,path,bank):
    import numpy as np
    import tensorflow as tf
    from makd_har.models import build_student
    from makd_har.reproducibility import set_global_seed
    from makd_har.training import aggregate_teacher_probabilities,train_distilled_student,predict_logits,classification_metrics_from_labels
    retained=task['retained_indices'];names=task['context']['teacher_order'];seed=task['training_seed']
    train_probs=aggregate_teacher_probabilities([bank[names[i]]['train'] for i in retained])
    val_probs=aggregate_teacher_probabilities([bank[names[i]]['val'] for i in retained])
    tf.keras.backend.clear_session();set_global_seed(seed)
    model=build_student(tuple(bundle.raw_train.shape[1:]),len(bundle.class_names))
    started=time.perf_counter()
    model,history=train_distilled_student(model,bundle.raw_train,bundle.y_train,bundle.raw_val,bundle.y_val,
        train_probs,val_probs,seed=seed,epochs=task['epochs'],batch_size=config.batch_size,
        learning_rate=config.learning_rate,patience=config.patience,temperature=config.temperature,
        hard_label_weight=config.hard_label_weight)
    val_predictions=predict_logits(model,bundle.raw_val,config.batch_size).argmax(axis=1)
    test_predictions=predict_logits(model,bundle.raw_test,config.batch_size).argmax(axis=1)
    best=-math.inf;best_epoch=0
    for epoch,score in enumerate(history['val_macro_f1'],1):
        if score>best+1e-7:best=score;best_epoch=epoch
    result=dict(status='complete',identity=digest(task),task=task,history=history,
        epochs_completed=len(history['train_loss']),best_epoch=best_epoch,elapsed_seconds=time.perf_counter()-started,
        validation_metrics=classification_metrics_from_labels(bundle.y_val,val_predictions),
        test_metrics=classification_metrics_from_labels(bundle.y_test,test_predictions),test_evaluated=True)
    history_check(result,task['epochs'],config.patience)
    path.parent.mkdir(parents=True,exist_ok=True)
    model_path=path.with_suffix('.keras');model.save(model_path,include_optimizer=False)
    prediction_path=path.with_suffix('.npz');temporary=path.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,y_val=bundle.y_val,val_predictions=val_predictions,
                            y_test=bundle.y_test,test_predictions=test_predictions)
    temporary.replace(prediction_path)
    result.update(model_file=model_path.name,model_sha256=sha(model_path),
                  predictions_file=prediction_path.name,predictions_sha256=sha(prediction_path))
    atomic_json(path,result)
    return result


def protocol(manifest):
    return dict(version=VERSION,datasets=['uci','wisdm'],outer_seeds=SEEDS,teacher_count=5,
        conditions=['full',*[f'remove-{i}' for i in range(5)]],epochs=60,
        choice='Lowest original raw validation Shapley; exact ties choose earliest saved teacher order',
        aggregation='Uniform over retained teachers; no revaluation or reweighting after removal',
        training_seed='outer_seed * 1000000 + 7 for every matched condition',
        random_removal='Arithmetic mean of all five leave-one-out macro-F1 outcomes; includes selected removal',
        primary_endpoint='Test macro-F1: selected removal minus expected uniform random removal',
        secondary_endpoints=['selected minus matched full','random minus matched full'],
        inference='Two-sided exact signed-rank sign enumeration with midranks; drop zeros; Holm over six dataset/comparison tests',
        interval='Descriptive unadjusted 95% t interval over ten paired seed differences per dataset',
        scope='Fixed subject split, post-review robustness; test outcomes never choose removal, checkpoints or seeds',
        frozen_inputs_sha256=sha(HERE/'frozen-inputs.json'),research_fits=120)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drive-root',type=Path,default=PACKAGE.parent)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--dataset',choices=('uci','wisdm','all'),default='all')
    args=p.parse_args();drive=args.drive_root.resolve();manifest=verify_inputs(drive)
    print('FROZEN INPUTS PASS: code, data, twenty production records and one hundred teacher checkpoints.',flush=True)
    if args.verify_only:return
    sys.path.insert(0,str(PACKAGE/'pipeline_snapshot'))
    import numpy as np
    import tensorflow as tf
    import sklearn
    from makd_har.config import ExperimentConfig
    tf.keras.utils.set_random_seed(2026)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    devices=tf.config.list_physical_devices('GPU');require(bool(devices),'Connect a GPU before training')
    env=json.loads(json.dumps(dict(python=sys.version,platform=platform.platform(),tensorflow=tf.__version__,
        keras=tf.keras.__version__,numpy=np.__version__,sklearn=sklearn.__version__,deterministic_operations=True,
        tensor_float_32_execution=tf.config.experimental.tensor_float_32_execution_enabled(),
        gpu=[tf.config.experimental.get_device_details(d) for d in devices])))
    specification=protocol(manifest)
    root=drive/'results/revision-2/teacher-pruning'
    ensure_same(root/'experiment-protocol.json',specification)
    ensure_same(root/'environment.json',env)
    if not args.preflight_only:
        gate=root/'preflight-completed.json'
        require(gate.is_file(),'Run the all-dataset preflight before smoke or research fits')
        completed_preflight=json.loads(gate.read_text())
        require(completed_preflight['passed'] is True and completed_preflight['teacher_checks']==100
            and completed_preflight['specification_sha256']==digest(specification)
            and completed_preflight['environment_sha256']==digest(env),'Preflight context differs from this runtime')
        require(len(completed_preflight['files'])==60,'Incomplete teacher-bank preflight evidence')
        for item in completed_preflight['files']:
            artifact=(drive/item['path']).resolve();require(root in artifact.parents,'Preflight path escapes output root')
            verify_file(artifact,item['sha256'])
    datasets=('uci','wisdm') if args.dataset=='all' else (args.dataset,)
    all_markers=[]
    for dataset in datasets:
        sources=manifest['seed_records'][dataset]
        first=json.loads((drive/sources['11']).read_text());config=ExperimentConfig(**first['config']);config.validate()
        bundle=load_bundle(drive,dataset,first,config)
        fits_root=drive/'results/revision-2/smoke-teacher-pruning' if args.smoke else root
        run_protocol=dict(specification=specification,mode='smoke' if args.smoke else 'research',dataset=dataset,
            outer_seeds=[11] if args.smoke else SEEDS,epochs=2 if args.smoke else 60,conditions=specification['conditions'])
        ensure_same(fits_root/dataset/'run-protocol.json',run_protocol)
        files=[];fit_count=0
        for outer in ([11] if args.smoke else SEEDS):
            source_path=drive/sources[str(outer)];source=json.loads(source_path.read_text())
            require(source['config']==first['config'] and source['dataset_manifest']==bundle.manifest(),
                    'Source configuration or split differs across banks')
            choice=removal_choice(source)
            context=dict(specification=specification,environment=env,dataset=dataset,outer_seed=outer,
                source_record_sha256=sha(source_path),config=source['config'],teacher_order=source['teacher_order'],
                frozen_choice=manifest['choices'][dataset][str(outer)])
            bank,parity=frozen_probability_bank(drive,dataset,outer,source,bundle,config,root/dataset,context)
            validate_parity(parity,source);validate_probability_bank(bank,bundle,source['teacher_order'])
            cache=root/dataset/f'seed-{outer}'
            preflight=dict(passed=True,context=context,context_sha256=digest(context),dataset_manifest=bundle.manifest(),
                teacher_validation_parity=parity,test_evaluated=False)
            ensure_same(cache/'preflight.json',preflight)
            print(f'PREFLIGHT PASS {dataset} bank {outer}: all five teachers and split match; remove {source["teacher_order"][choice]} by rule.',flush=True)
            if args.preflight_only:continue
            seed_dir=fits_root/dataset/f'seed-{outer}';reports={}
            for condition in specification['conditions']:
                removed=None if condition=='full' else int(condition.split('-')[1])
                retained=retained_indices(removed)
                task=dict(context=context,mode=run_protocol['mode'],condition=condition,removed_index=removed,
                    retained_indices=retained,uniform_weights=[1/len(retained)]*len(retained),
                    training_seed=outer*1000000+7,epochs=run_protocol['epochs'])
                path=seed_dir/f'{condition}.json';report=checked_fit(path,task)
                if report is None:
                    print(f'FITTING {dataset} bank {outer}: {condition}, {len(retained)} teachers, limit {task["epochs"]}',flush=True)
                    report=train_fit(bundle,config,task,path,bank)
                    print(f'SAVED {dataset} bank {outer}: {condition}; {report["epochs_completed"]} epochs; {report["elapsed_seconds"]:.1f} seconds.',flush=True)
                else:print(f'RESUMED verified {dataset} bank {outer}: {condition}',flush=True)
                reports[condition]=report;fit_count+=1
                for artifact in (path,path.parent/report['model_file'],path.parent/report['predictions_file']):
                    files.append(dict(path=artifact.relative_to(drive).as_posix(),sha256=sha(artifact)))
            summary=dict(context=context,run_protocol=run_protocol,removed_index=choice,
                removed_teacher=source['teacher_order'][choice],raw_shapley_values=source['shapley']['values'],
                comparisons=paired_summary(reports,choice))
            summary_path=seed_dir/'removal-comparison.json';atomic_json(summary_path,summary)
            files.append(dict(path=summary_path.relative_to(drive).as_posix(),sha256=sha(summary_path)))
            print(f'BANK COMPLETE {dataset} {outer}: all six matched outcomes saved.',flush=True)
        if not args.preflight_only:
            expected=len(run_protocol['outer_seeds'])*6;require(fit_count==expected,'Incomplete dataset fit count')
            marker=fits_root/dataset/'completed.json'
            atomic_json(marker,dict(status='complete',mode=run_protocol['mode'],dataset=dataset,fits=fit_count,
                run_protocol_sha256=digest(run_protocol),environment_sha256=digest(env),files=files,test_evaluated=True))
            all_markers.append(marker)
        else:print(f'ALL BANKS PREFLIGHT PASS: {dataset}',flush=True)
    if args.preflight_only:
        if args.dataset=='all':
            preflight_files=[]
            for dataset in ('uci','wisdm'):
                for seed in SEEDS:
                    for name in ('preflight.json','teacher-probabilities.json','teacher-probabilities.npz'):
                        path=root/dataset/f'seed-{seed}'/name
                        preflight_files.append(dict(path=path.relative_to(drive).as_posix(),sha256=sha(path)))
            atomic_json(root/'preflight-completed.json',dict(passed=True,teacher_checks=100,
                banks=20,specification_sha256=digest(specification),environment_sha256=digest(env),files=preflight_files))
    elif args.dataset=='all':
        total=sum(json.loads(path.read_text())['fits'] for path in all_markers)
        require(total==(12 if args.smoke else 120),'Incomplete combined fit set')
        atomic_json(fits_root/'completed.json',dict(status='complete',mode='smoke' if args.smoke else 'research',
            fits=total,datasets=[dict(path=path.relative_to(drive).as_posix(),sha256=sha(path)) for path in all_markers],
            specification_sha256=digest(specification),environment_sha256=digest(env),test_evaluated=True))
        print(f'COMPLETE {"smoke" if args.smoke else "research"}: {total} fits. Outputs: {fits_root}',flush=True)


if __name__=='__main__':main()
