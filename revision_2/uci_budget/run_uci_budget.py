"""Paired UCI-HAR coalition budget sensitivity; original production files are read only."""
from __future__ import annotations
import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
sys.path.insert(0,str(PACKAGE))
from run_coalition_noise import sha,digest,atomic_json,ensure_same,verify_file,load_bundle

VERSION='revision2-uci-budget-v1'


def verify_inputs(drive):
    manifest=json.loads((HERE/'frozen-inputs.json').read_text())
    for relative,expected in manifest['files'].items():
        path=(drive/relative).resolve()
        if drive not in path.parents:raise RuntimeError('Input path escapes the project')
        verify_file(path,expected)
    return manifest


def prefix_utility(history,budget):
    """Apply the frozen trainer's strict checkpoint rule to the available prefix."""
    losses=history.get('train_loss',[])
    scores=history.get('val_macro_f1',[])
    if not scores or len(scores)!=len(losses):raise RuntimeError('Invalid training history')
    import math
    if not all(math.isfinite(x) for x in losses):raise RuntimeError('Nonfinite loss')
    if not all(math.isfinite(x) and 0<=x<=1 for x in scores):raise RuntimeError('Invalid validation history')
    if budget<1:raise RuntimeError('Invalid prefix budget')
    best=-float('inf');best_epoch=None
    for epoch,score in enumerate(scores[:budget],1):
        if score>best+1e-7:best,best_epoch=score,epoch
    return dict(validation_macro_f1=best,best_epoch=best_epoch,observed_epochs=min(budget,len(scores)))


def checked_result(path,identity,kind):
    if not path.exists():return None
    x=json.loads(path.read_text())
    if x.get('status')!='complete' or x.get('identity')!=identity or x.get('kind')!=kind:
        raise RuntimeError(f'Resume identity mismatch: {path}')
    if x.get('test_evaluated') is not (kind=='final'):
        raise RuntimeError(f'Wrong test-evaluation scope: {path}')
    for budget,report in x['budget_utilities'].items():
        if prefix_utility(x['history'],int(budget))!=report:raise RuntimeError('Cached utility/history mismatch')
    if kind=='final':
        verify_file(path.parent/x['model_file'],x['model_sha256'])
    return x


def load_fixed_bank(drive,seed,source,bundle,frozen):
    import numpy as np
    noise=drive/'results/revision-2/coalition-noise/uci'
    old_frozen=json.loads((PACKAGE/'frozen-inputs.json').read_text())
    old_protocol=json.loads((noise/'experiment-protocol.json').read_text())
    info=json.loads((noise/f'seed-{seed}/teacher-probabilities.json').read_text())
    preflight=json.loads((noise/f'seed-{seed}/preflight.json').read_text())
    environment_files=list(noise.glob('environment-*.json'))
    contexts=[]
    for path in environment_files:
        env=json.loads(path.read_text())
        context=dict(protocol=old_protocol,outer_seed=seed,
                     source_record_sha256=sha(drive/frozen['seed_records'][str(seed)]),
                     source_files=old_frozen['datasets']['uci']['source_files'],
                     runtime_files=old_frozen['runtime_files'],environment=env)
        contexts.append(digest(context))
    if info['identity'] not in contexts or preflight['identity']!=info['identity'] or not preflight['passed']:
        raise RuntimeError('Frozen probability bank identity mismatch')
    if info['teacher_validation_parity']!=preflight['teacher_validation_parity']:
        raise RuntimeError('Inconsistent teacher parity evidence')
    for name in source['teacher_order']:
        p=info['teacher_validation_parity'][name]
        if p['saved']!=source['teachers'][name]['validation'] or p['tolerance']!=1e-4:
            raise RuntimeError('Teacher source or tolerance mismatch')
        for metric in ('accuracy','macro_f1'):
            error=abs(p['saved'][metric]-p['reproduced'][metric])
            if error!=p['absolute_metric_errors'][metric] or error>1e-4:
                raise RuntimeError('Teacher validation parity failed')
    path=noise/f'seed-{seed}/teacher-probabilities.npz'
    verify_file(path,info['sha256'])
    bank={}
    with np.load(path,allow_pickle=False) as arrays:
        for name in source['teacher_order']:
            bank[name]={}
            for part,count in [('train',len(bundle.y_train)),('val',len(bundle.y_val))]:
                values=arrays[f'{name}_{part}'].copy()
                if values.shape!=(count,len(bundle.class_names)) or not np.isfinite(values).all() or (values<0).any():
                    raise RuntimeError('Invalid probability bank shape or values')
                if not np.allclose(values.sum(axis=1),1,rtol=0,atol=1e-5):raise RuntimeError('Unnormalized teacher targets')
                bank[name][part]=values
    return bank,info


def fit(bundle,config,seed,epochs,train_probs,val_probs,final_model=None):
    import tensorflow as tf
    from makd_har.models import build_student
    from makd_har.reproducibility import set_global_seed
    from makd_har.training import train_distilled_student,predict_logits,classification_metrics
    tf.keras.backend.clear_session()
    set_global_seed(seed)
    model=build_student(tuple(bundle.raw_train.shape[1:]),len(bundle.class_names))
    started=time.perf_counter()
    model,history=train_distilled_student(model,bundle.raw_train,bundle.y_train,bundle.raw_val,bundle.y_val,
        train_probs,val_probs,seed=seed,epochs=epochs,batch_size=config.batch_size,
        learning_rate=config.learning_rate,patience=config.patience,temperature=config.temperature,
        hard_label_weight=config.hard_label_weight)
    metrics=classification_metrics(bundle.y_val,predict_logits(model,bundle.raw_val,config.batch_size))
    if abs(metrics['macro_f1']-prefix_utility(history,epochs)['validation_macro_f1'])>1e-4:
        raise RuntimeError('Restored best validation checkpoint does not match history')
    result=dict(status='complete',history=history,epochs_completed=len(history['train_loss']),
                validation_metrics=metrics,elapsed_seconds=time.perf_counter()-started,
                test_evaluated=final_model is not None)
    if final_model:
        # Both predeclared final conditions are reported; test scores never select weights or checkpoints.
        result['test_metrics']=classification_metrics(bundle.y_test,predict_logits(model,bundle.raw_test,config.batch_size))
        final_model.parent.mkdir(parents=True,exist_ok=True)
        model.save(final_model,include_optimizer=False)
        result.update(model_file=final_model.name,model_sha256=sha(final_model))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--drive-root',type=Path,default=PACKAGE.parent)
    p.add_argument('--verify-only',action='store_true')
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--preflight-only',action='store_true')
    args=p.parse_args()
    drive=args.drive_root.resolve()
    frozen=verify_inputs(drive)
    print('All frozen UCI data, teachers, probability banks and code hashes passed.',flush=True)
    if args.verify_only:return
    sys.path.insert(0,str(PACKAGE/'pipeline_snapshot'))
    import numpy as np
    import tensorflow as tf
    import sklearn
    from makd_har.config import ExperimentConfig
    from makd_har.shapley import exact_shapley,shapley_to_weights
    from makd_har.training import aggregate_teacher_probabilities
    # These settings apply before any model is built or used in this process.
    tf.keras.utils.set_random_seed(2026)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    devices=tf.config.list_physical_devices('GPU')
    if not devices:raise RuntimeError('A GPU runtime is required; stop before research training.')
    env=dict(python=sys.version,platform=platform.platform(),tensorflow=tf.__version__,keras=tf.keras.__version__,
             numpy=np.__version__,sklearn=sklearn.__version__,deterministic_operations=True,
             tensor_float_32_execution=tf.config.experimental.tensor_float_32_execution_enabled(),
             gpu=[tf.config.experimental.get_device_details(d) for d in devices])
    # GPU details may include tuples, normalized to JSON arrays for stable identities.
    env=json.loads(json.dumps(env))
    seeds=[11] if args.smoke else [11,23,37]
    budgets=[1,2] if args.smoke else [20,60]
    final_epochs=2 if args.smoke else 60
    out=drive/'results/revision-2'/('smoke-uci-budget' if args.smoke else 'uci-budget')
    protocol=dict(version=VERSION,mode='smoke' if args.smoke else 'research',dataset='uci',outer_seeds=seeds,
        coalition_budgets=budgets,final_epochs=final_epochs,players=2 if args.smoke else 5,
        frozen_inputs_sha256=sha(HERE/'frozen-inputs.json'),checkpoint='validation macro-F1 improvement > 1e-7; patience 10',
        coalition_seed='outer_seed * 100000 + 17',final_student_seed='outer_seed * 1000000 + 7',
        comparison='20-epoch prefix and 60-epoch utility from the same seeded training trajectory',
        test_scope='No coalition test evaluation; both predeclared final students evaluated; no test selection',
        historical_scope='Original 20-epoch game is a historical reference; runtime differences may prevent reproduction',
        teacher_targets='Previously verified UCI probability banks from the noise block; no teacher retraining')
    ensure_same(out/'experiment-protocol.json',protocol)
    ensure_same(out/f'environment-{digest(env)[:16]}.json',env)
    all_files=[];fit_count=0
    for outer in seeds:
        source_path=drive/frozen['seed_records'][str(outer)]
        source=json.loads(source_path.read_text())
        if source['config']['coalition_epochs']!=20 or source['config']['student_epochs']!=60:
            raise RuntimeError('Unexpected production epoch budget')
        coalition_seed=outer*100000+17
        final_seed=outer*1000000+7
        for x in source['coalition_reports_created_this_run'].values():
            if x['seed']!=coalition_seed:raise RuntimeError('Original coalition seed rule changed')
        if source['methods']['proposed']['seed']!=final_seed:raise RuntimeError('Original final seed rule changed')
        config=ExperimentConfig(**source['config']);config.validate()
        bundle=load_bundle(drive,'uci',source,config)
        bank,bank_info=load_fixed_bank(drive,outer,source,bundle,frozen)
        names=source['teacher_order'][:protocol['players']]
        context=dict(protocol=protocol,environment=env,source_record_sha256=sha(source_path),
                     teacher_probabilities_sha256=bank_info['sha256'],config=source['config'],teacher_order=names)
        seed_dir=out/f'seed-{outer}'
        ensure_same(seed_dir/'preflight.json',dict(passed=True,context_sha256=digest(context),
                    teacher_probability_identity=bank_info['identity'],dataset_manifest=bundle.manifest(),
                    teacher_validation_parity=bank_info['teacher_validation_parity']))
        print(f'PRECHECK PASS bank {outer}: split and all five saved teacher targets verified.',flush=True)
        if args.preflight_only:continue
        reports={}
        for size in range(len(names)+1):
            for coalition in itertools.combinations(range(len(names)),size):
                key='-'.join(map(str,coalition)) if coalition else 'empty'
                path=seed_dir/'coalitions'/f'{key}.json'
                task=dict(context=context,kind='coalition',coalition_indices=list(coalition),training_seed=coalition_seed)
                identity=digest(task)
                report=checked_result(path,identity,'coalition')
                if report is None:
                    train_probs=aggregate_teacher_probabilities([bank[names[i]]['train'] for i in coalition]) if coalition else None
                    val_probs=aggregate_teacher_probabilities([bank[names[i]]['val'] for i in coalition]) if coalition else None
                    print(f'FITTING bank {outer} coalition {key} ({len(reports)+1}/{2**len(names)}); limits {budgets}',flush=True)
                    report=fit(bundle,config,coalition_seed,max(budgets),train_probs,val_probs)
                    report.update(identity=identity,kind='coalition',outer_seed=outer,training_seed=coalition_seed,
                                  coalition_indices=list(coalition),teacher_names=[names[i] for i in coalition],
                                  budget_utilities={str(b):prefix_utility(report['history'],b) for b in budgets})
                    atomic_json(path,report)
                else:print(f'RESUMED bank {outer} coalition {key}',flush=True)
                reports[coalition]=report
                all_files.append(dict(path=str(path.relative_to(drive)),sha256=sha(path)));fit_count+=1
        games={}
        for budget in budgets:
            result=exact_shapley(len(names),lambda c:reports[c]['budget_utilities'][str(budget)]['validation_macro_f1'])
            weights=shapley_to_weights(result.values,config.shapley_softmax_temperature)
            residual=float(result.values.sum()-(result.coalition_values[tuple(range(len(names)))]-result.coalition_values[()]))
            if abs(residual)>1e-10:raise RuntimeError('Shapley efficiency identity failed')
            games[str(budget)]=dict(shapley=result.to_dict(),weights=weights.tolist(),efficiency_residual=residual)
        # No candidate selection: train and report both conditions regardless of utility or test outcome.
        finals={}
        for budget in budgets:
            weights=np.asarray(games[str(budget)]['weights'])
            path=seed_dir/f'final-from-{budget}-epoch-game.json'
            model_path=seed_dir/f'final-from-{budget}-epoch-game.keras'
            task=dict(context=context,kind='final',weight_game_epochs=budget,weights=weights.tolist(),
                      training_seed=final_seed,epochs=final_epochs)
            identity=digest(task)
            report=checked_result(path,identity,'final')
            if report is None:
                train_probs=aggregate_teacher_probabilities([bank[name]['train'] for name in names],weights)
                val_probs=aggregate_teacher_probabilities([bank[name]['val'] for name in names],weights)
                print(f'FITTING bank {outer} final student from {budget}-epoch game; limit {final_epochs}',flush=True)
                report=fit(bundle,config,final_seed,final_epochs,train_probs,val_probs,model_path)
                report.update(identity=identity,kind='final',outer_seed=outer,training_seed=final_seed,
                              weight_game_epochs=budget,weights=weights.tolist(),
                              budget_utilities={str(final_epochs):prefix_utility(report['history'],final_epochs)})
                atomic_json(path,report)
            finals[str(budget)]=report
            all_files.append(dict(path=str(path.relative_to(drive)),sha256=sha(path)));fit_count+=1
            all_files.append(dict(path=str(model_path.relative_to(drive)),sha256=sha(model_path)))
        historical=source['shapley']
        comparison=dict(context=context,games=games,finals=finals,teacher_order=names,
            historical_20_epoch_game=historical,historical_20_epoch_weights=source['shapley_weights'],
            historical_prefix_utility_max_absolute_difference=(None if args.smoke else max(
                abs(reports[c]['budget_utilities']['20']['validation_macro_f1']-historical['coalition_values'][','.join(map(str,c))])
                for c in reports)),
            note='Use the paired prefix games for budget effects. Original published game is separately labeled historical.')
        path=seed_dir/'budget-comparison.json';atomic_json(path,comparison)
        all_files.append(dict(path=str(path.relative_to(drive)),sha256=sha(path)))
        print(f'BANK COMPLETE {outer}: paired games and both final students saved.',flush=True)
    if not args.preflight_only:
        expected=len(seeds)*(2**protocol['players']+2)
        if fit_count!=expected:raise RuntimeError('Incomplete fit count')
        atomic_json(out/'completed.json',dict(status='complete',mode=protocol['mode'],fits=fit_count,
                    protocol_sha256=digest(protocol),environment_sha256=digest(env),files=all_files,
                    coalition_test_evaluated=False,final_test_evaluated=True))
        print(f'COMPLETE {protocol["mode"]}: {fit_count} fits; all files saved under {out}',flush=True)


if __name__=='__main__':main()
