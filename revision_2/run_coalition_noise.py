"""Second-review fixed-bank noise experiment. Never evaluates test predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSION = 'revision2-coalition-noise-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def atomic_json(path, payload):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    temporary.replace(path)


def ensure_same(path,payload):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text()) != payload:
            raise RuntimeError(f'Protocol differs from {path}. Use a separate output folder.')
    else:
        atomic_json(path,payload)


def checked_resume(path, identity):
    path=Path(path)
    if not path.exists():
        return None
    report=json.loads(path.read_text())
    if report.get('identity') != identity or report.get('status')!='complete':
        raise RuntimeError(f'Unsafe result reuse at {path}')
    score=report.get('validation_macro_f1')
    if not isinstance(score,(int,float)) or not 0 <= score <= 1:
        raise RuntimeError(f'Invalid result at {path}')
    return report


def verify_file(path, expected):
    if not Path(path).is_file() or sha(path)!=expected:
        raise RuntimeError(f'Frozen source missing or changed: {path}')


def compare_manifest(current, saved):
    # These include subjects, per-class counts, feature names, and parser provenance.
    if current != saved:
        raise RuntimeError('Loaded data differs from the original production manifest; stop before fitting.')


def verify_inputs(drive, dataset, frozen):
    for rel,expected in frozen['runtime_files'].items():
        verify_file(HERE/rel,expected)
    for rel,expected in frozen['datasets'][dataset]['source_files'].items():
        verify_file(drive/rel,expected)


def uci_root(drive):
    archive=drive/'resources/UCI HAR Dataset.zip'
    # Extract into local runtime storage, not into Google Drive.
    target=Path('/content/revision2_data') if Path('/content').exists() else HERE/'local_data'
    marker=target/'archive-sha256.txt'
    archive_hash=sha(archive)
    if marker.exists() and marker.read_text().strip()==archive_hash:
        candidates=list(target.rglob('train/X_train.txt'))
        if len(candidates)==1:
            return candidates[0].parent.parent
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f'Extraction directory is not empty or differs from the frozen archive: {target}')
    target.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        root=target.resolve()
        for member in z.infolist():
            resolved=(target/member.filename).resolve()
            if resolved != root and root not in resolved.parents:
                raise RuntimeError('Unsafe path in UCI archive')
        z.extractall(target)
    # Support the source archive's actual top-level structure without guessing data files.
    candidates=list(target.rglob('train/X_train.txt'))
    if len(candidates)!=1:
        raise RuntimeError('Ambiguous UCI extraction')
    actual=candidates[0].parent.parent
    marker.write_text(archive_hash)
    return actual


def load_bundle(drive,dataset,source,config):
    from makd_har.data import load_uci_har,load_wisdm_v1
    if dataset=='uci':
        bundle=load_uci_har(uci_root(drive),validation_fraction=config.validation_fraction,split_seed=config.split_seed)
    else:
        bundle=load_wisdm_v1(drive/'resources/WISDM_ar_v1.1_raw.txt',
                             validation_fraction=config.validation_fraction,
                             test_fraction=config.wisdm_test_fraction,split_seed=config.split_seed)
    compare_manifest(bundle.manifest(),source['dataset_manifest'])
    return bundle


def frozen_probability_bank(drive,dataset,seed,source,bundle,config,out,context):
    import numpy as np
    import tensorflow as tf
    from makd_har.models import teacher_registry
    from makd_har.training import predict_logits,logits_to_probabilities,classification_metrics
    relative='results/uci/production' if dataset=='uci' else 'results/wisdm/production-v2'
    cache=out/f'seed-{seed}'/'teacher-probabilities.npz'
    cache_info=cache.with_suffix('.json')
    bank_identity=digest(context)
    if cache.exists() and cache_info.exists():
        info=json.loads(cache_info.read_text())
        if info['identity']!=bank_identity or sha(cache)!=info['sha256']:
            raise RuntimeError('Probability cache differs from frozen teacher/data/code identity')
        with np.load(cache,allow_pickle=False) as arrays:
            result={name:{part:arrays[f'{name}_{part}'].copy() for part in ('train','val')}
                    for name in source['teacher_order']}
        return result,info['teacher_validation_parity']
    bank,parity={},{}
    for name in source['teacher_order']:
        path=drive/relative/f'seed-{seed}/models/teacher-{name}.keras'
        tf.keras.backend.clear_session()
        model=tf.keras.models.load_model(path,compile=False,safe_mode=True)
        kind=teacher_registry()[name].input_kind
        val_logits=predict_logits(model,getattr(bundle,f'{kind}_val'),config.batch_size)
        current=classification_metrics(bundle.y_val,val_logits)
        expected=source['teachers'][name]['validation']
        errors={metric:abs(float(current[metric])-float(expected[metric])) for metric in ('accuracy','macro_f1')}
        if max(errors.values())>1e-4:
            raise RuntimeError(f'{dataset} seed {seed} teacher {name} fails validation parity: {errors}')
        train_logits=predict_logits(model,getattr(bundle,f'{kind}_train'),config.batch_size)
        bank[name]={'train':logits_to_probabilities(train_logits,config.temperature),
                    'val':logits_to_probabilities(val_logits,config.temperature)}
        parity[name]={'saved':expected,'reproduced':current,'absolute_metric_errors':errors,'tolerance':1e-4}
        print(f'Validated frozen teacher {dataset} seed {seed}: {name}',flush=True)
        del model
    cache.parent.mkdir(parents=True,exist_ok=True)
    # np.savez_compressed writes to a file handle so no implicit extension is added.
    temporary=cache.with_suffix('.npz.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream,**{f'{name}_{part}':values for name,parts in bank.items() for part,values in parts.items()})
    temporary.replace(cache)
    atomic_json(cache_info,dict(identity=bank_identity,sha256=sha(cache),teacher_validation_parity=parity))
    return bank,parity


def one_fit(bundle,config,train_probabilities,val_probabilities,training_seed,epochs):
    import tensorflow as tf
    from makd_har.models import build_student
    from makd_har.reproducibility import set_global_seed
    from makd_har.training import train_distilled_student,predict_logits,classification_metrics
    tf.keras.backend.clear_session()
    set_global_seed(training_seed)
    model=build_student(tuple(bundle.raw_train.shape[1:]),len(bundle.class_names))
    started=time.perf_counter()
    model,history=train_distilled_student(model,bundle.raw_train,bundle.y_train,bundle.raw_val,bundle.y_val,
                                        train_probabilities,val_probabilities,seed=training_seed,epochs=epochs,
                                        batch_size=config.batch_size,learning_rate=config.learning_rate,
                                        patience=config.patience,temperature=config.temperature,
                                        hard_label_weight=config.hard_label_weight)
    metrics=classification_metrics(bundle.y_val,predict_logits(model,bundle.raw_val,config.batch_size))
    elapsed=time.perf_counter()-started
    return dict(validation_macro_f1=float(metrics['macro_f1']),validation_metrics=metrics,
                history=history,epochs_completed=len(history['train_loss']),elapsed_seconds=elapsed,
                status='complete',test_evaluated=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root',type=Path,default=HERE.parent)
    parser.add_argument('--dataset',choices=('uci','wisdm','all'),default='all')
    parser.add_argument('--verify-files-only',action='store_true')
    parser.add_argument('--preflight-only',action='store_true')
    parser.add_argument('--smoke',action='store_true',help='One epoch, first seed, one repetition; separate outputs; not research evidence')
    args=parser.parse_args()
    drive=args.drive_root.resolve()
    frozen=json.loads((HERE/'frozen-inputs.json').read_text())
    datasets=('uci','wisdm') if args.dataset=='all' else (args.dataset,)
    for dataset in datasets:
        verify_inputs(drive,dataset,frozen)
        print(f'Frozen input checks passed: {dataset}',flush=True)
    if args.verify_files_only:
        return
    sys.path.insert(0,str(HERE/'pipeline_snapshot'))
    import dataclasses
    import numpy as np
    import tensorflow as tf
    from makd_har.config import ExperimentConfig
    from makd_har.training import aggregate_teacher_probabilities
    import sklearn
    env=dict(python=sys.version,tensorflow=tf.__version__,keras=tf.keras.__version__,numpy=np.__version__,
             sklearn=sklearn.__version__,platform=platform.platform(),
             gpu_devices=[d.name for d in tf.config.list_physical_devices('GPU')])
    for dataset in datasets:
        source_info=frozen['datasets'][dataset]
        seeds=(11,) if args.smoke else (11,23,37)
        source=json.loads((drive/source_info['seed_records']['11']).read_text())
        known={f.name for f in dataclasses.fields(ExperimentConfig)}
        unknown=set(source['config'])-known
        if unknown:
            raise RuntimeError(f'Unsupported recorded config fields: {unknown}')
        config=ExperimentConfig(**source['config'])
        config.validate()
        bundle=load_bundle(drive,dataset,source,config)
        out=drive/'results/revision-2'/('smoke-coalition-noise' if args.smoke else 'coalition-noise')/dataset
        protocol=dict(version=VERSION,dataset=dataset,mode='smoke' if args.smoke else 'research',
                      outer_seeds=list(seeds),repetitions=1 if args.smoke else 5,
                      epochs=1 if args.smoke else config.coalition_epochs,config=source['config'],
                      frozen_inputs_sha256=sha(HERE/'frozen-inputs.json'),test_evaluated=False,
                      purpose='Within-bank student-optimization noise; no full Shapley uncertainty estimate')
        ensure_same(out/'experiment-protocol.json',protocol)
        # Session-specific environment is attached to each fit, not silently merged into one identity.
        environment_hash=digest(env)
        ensure_same(out/f'environment-{environment_hash[:16]}.json',env)
        all_results=[]
        for seed in seeds:
            source=json.loads((drive/source_info['seed_records'][str(seed)]).read_text())
            context=dict(protocol=protocol,outer_seed=seed,
                         source_record_sha256=sha(drive/source_info['seed_records'][str(seed)]),
                         source_files=source_info['source_files'],runtime_files=frozen['runtime_files'],
                         environment=env)
            bank,parity=frozen_probability_bank(drive,dataset,seed,source,bundle,config,out,context)
            atomic_json(out/f'seed-{seed}'/'preflight.json',dict(passed=True,teacher_validation_parity=parity,
                        identity=digest(context),test_evaluated=False))
            if args.preflight_only:
                continue
            train_full=aggregate_teacher_probabilities([bank[name]['train'] for name in source['teacher_order']])
            val_full=aggregate_teacher_probabilities([bank[name]['val'] for name in source['teacher_order']])
            for repetition in range(1,protocol['repetitions']+1):
                training_seed=seed*1000000+900000+repetition
                for coalition in ('empty','full'):
                    task=dict(context=context,repetition=repetition,coalition=coalition,training_seed=training_seed)
                    identity=digest(task)
                    path=out/f'seed-{seed}'/f'repetition-{repetition:02d}-{coalition}.json'
                    report=checked_resume(path,identity)
                    if report is None:
                        print(f'Fitting {dataset} bank {seed}, repetition {repetition}, {coalition}',flush=True)
                        report=one_fit(bundle,config,train_full if coalition=='full' else None,
                                       val_full if coalition=='full' else None,training_seed,protocol['epochs'])
                        report.update(dict(identity=identity,outer_seed=seed,repetition=repetition,
                                           coalition=coalition,training_seed=training_seed,
                                           environment_sha256=environment_hash,
                                           source_raw_shapley_values=source['shapley']['values']))
                        atomic_json(path,report)
                    else:
                        print(f'Resumed verified result: {dataset} {seed} {repetition} {coalition}',flush=True)
                    all_results.append(dict(path=str(path.relative_to(drive)),sha256=sha(path)))
        if not args.preflight_only:
            expected=len(seeds)*protocol['repetitions']*2
            assert len(all_results)==expected
            atomic_json(out/'completed.json',dict(status='complete',fits=expected,files=all_results,
                         protocol_sha256=digest(protocol),test_evaluated=False))
            print(f'COMPLETE {dataset}: {expected} fits; no test predictions evaluated.',flush=True)


if __name__=='__main__':
    main()
