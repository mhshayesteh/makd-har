"""Frozen CPU deployment audit. No teacher/student training or test-driven selection."""
from __future__ import annotations
import os
# A fresh subprocess sets these before importing any numerical/runtime library.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','TF_NUM_INTRAOP_THREADS','TF_NUM_INTEROP_THREADS'):
    os.environ[key]='1'
os.environ.update(CUDA_VISIBLE_DEVICES='-1',TF_ENABLE_ONEDNN_OPTS='0',TF_DETERMINISTIC_OPS='1')
import argparse
import hashlib
import json
import platform
import sys
import tempfile
import time
import uuid
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
SEEDS=[11,23,37,53,71,89,107,131,167,197]
VERSION='revision2-runtime-benchmark-v2'
SOURCE_TOL=1e-4
LOGIT_ATOL=1e-4
LOGIT_RTOL=1e-4

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)

def save_npz(path,*,compress=True,**arrays):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.with_suffix('.npz.tmp').open('wb') as f:(np.savez_compressed if compress else np.savez)(f,**arrays)
    path.with_suffix('.npz.tmp').replace(path)

def verify_files(root,items):
    for item in items:
        p=(root/item['path']).resolve();require(root.resolve() in p.parents,'Unsafe recorded path')
        require(p.is_file() and sha(p)==item['sha256'],f'Missing/changed artifact: {p}')

def records(root,paths):return [dict(path=p.relative_to(root).as_posix(),sha256=sha(p)) for p in paths]

def verify_inputs(drive):
    frozen=json.loads((HERE/'frozen-inputs.json').read_text())
    for rel,h in frozen['files'].items():
        p=(drive/rel).resolve();require(drive in p.parents and p.is_file() and sha(p)==h,f'Frozen input differs: {rel}')
    return frozen

def protocol(frozen):
    return dict(version=VERSION,frozen_inputs_sha256=digest(frozen),datasets=['uci','wisdm'],seeds=SEEDS,
        students='20 original proposed checkpoints; no retraining',precision='float32 TFLite; float64 compact SVM',
        source_metric_absolute_tolerance=SOURCE_TOL,conversion_logit_atol=LOGIT_ATOL,conversion_logit_rtol=LOGIT_RTOL,
        conversion_label_agreement='exact on validation and test; zero disagreements required',
        delegate='optimized builtin CPU kernels, default delegates explicitly disabled',threads=1,
        conversion='Keras SavedModel -> TFLITE_BUILTINS only; no quantization, custom ops or Select TF ops',
        svm='reuse both selected saved SVCs; one training-only WISDM linear SVC fit using original validation-selected linear C',
        batch_sizes=[1,128],rounds=3,batch1_warmups=30,batch1_repetitions=300,batch128_warmups=5,batch128_repetitions=30,
        timing_input='fixed validation windows, seed 20261019; identical subject/window indices for SVM and student',
        timing_scope='classifier call through output materialization and argmax/vote; preprocessing, feature extraction, loading, tensor allocation and tracing excluded',
        timing_order='all cases interleaved in deterministic random order each round; no mixed-session timing reuse',
        interpretation='descriptive CPU session; not physical edge device, energy, end-to-end or independent cohort evidence')

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);require(y.ndim==p.ndim==1 and len(y)==len(p)>0,'Bad labels')
    require(np.isin(y,range(6)).all() and np.isin(p,range(6)).all(),'Unknown class')
    cm=np.bincount(6*y.astype(int)+p.astype(int),minlength=36).reshape(6,6);tp=np.diag(cm).astype(float)
    denominator=cm.sum(0)+cm.sum(1);f=np.divide(2*tp,denominator,out=np.zeros(6),where=denominator>0)
    return dict(accuracy=float(tp.sum()/len(y)),macro_f1=float(f.mean()),per_class_f1=f.tolist(),confusion_matrix=cm.tolist())

def source_parity(current,expected):
    errors={k:abs(current[k]-float(expected[k])) for k in ('accuracy','macro_f1') if k in expected}
    require(errors and max(errors.values())<=SOURCE_TOL,f'Original source metric parity failed: {errors}')
    return dict(saved=expected,reproduced=current,errors=errors,tolerance=SOURCE_TOL)

def converted_parity(reference,converted):
    require(reference.shape==converted.shape and np.isfinite(reference).all() and np.isfinite(converted).all(),'Invalid logits')
    np.testing.assert_allclose(converted,reference,rtol=LOGIT_RTOL,atol=LOGIT_ATOL)
    disagreements=int(np.count_nonzero(reference.argmax(1)!=converted.argmax(1)))
    require(disagreements==0,'Conversion changed predicted labels; stop and diagnose without relaxing gates')
    return dict(label_disagreements=disagreements,max_absolute_logit_error=float(np.max(np.abs(reference-converted))),atol=LOGIT_ATOL,rtol=LOGIT_RTOL)

def linear_choice(source):
    choices=[r for r in source['methods']['svm']['tuning_results'] if r['params']['kernel']=='linear']
    require(len(choices)>0,'Original linear validation candidates absent')
    # Stable first candidate on exact ties, as in the original grid search.
    best=max(choices,key=lambda r:r['validation_macro_f1'])
    return best

def export_linear(model):
    require(model.kernel=='linear' and len(model.classes_)==6 and not model.break_ties,'Expected six-class linear SVC without break_ties')
    return dict(coef=np.array(model.coef_,dtype=np.float64),intercept=np.array(model.intercept_,dtype=np.float64),classes=np.array(model.classes_,dtype=np.int64))

def ovo_predict(x,state):
    values=np.asarray(x,dtype=np.float64)@state['coef'].T+state['intercept']
    n=len(state['classes']);require(values.shape[1]==n*(n-1)//2,'Bad OVO coefficient shape')
    votes=np.zeros((len(x),n),dtype=np.int64);k=0
    for i in range(n):
        for j in range(i+1,n):
            # LibSVM multiclass: positive -> class i; zero/negative -> class j.
            positive=values[:,k]>0;votes[:,i]+=positive;votes[:,j]+=~positive;k+=1
    # First class wins tied vote counts, matching SVC.predict(break_ties=False).
    return state['classes'][votes.argmax(axis=1)]

def timing_plan(n):
    require(n>=330,'Too few validation windows for prespecified timing corpus')
    order=np.random.default_rng(20261019).permutation(n)
    return dict(warm1=order[:30],sample1=order[30:330],warm128=order[:128],sample128=order[128:256])

def timing_summary(ns,batch):
    v=np.asarray(ns,dtype=np.float64);require(len(v)>1 and np.isfinite(v).all() and (v>0).all(),'Invalid timing samples')
    ms=v/1e6
    return dict(repetitions=len(v),batch_size=batch,mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),
        p95_ms=float(np.percentile(ms,95)),std_ms=float(ms.std(ddof=1)),min_ms=float(ms.min()),max_ms=float(ms.max()),
        samples_per_second=float(batch*1e9/v.mean()))

def canonical_threadpools(pools):
    # Discovery order varies across fresh processes. Preserve every library,
    # version, thread count and duplicate entry, while comparing a stable order.
    values=[{k:v for k,v in pool.items() if k!='filepath'} for pool in pools]
    require(all(p.get('num_threads')==1 for p in values),'A numerical library is not limited to one thread')
    return sorted(values,key=lambda p:json.dumps(p,sort_keys=True,separators=(',',':')))

def initialize_runtime():
    import tensorflow as tf
    import sklearn,joblib,threadpoolctl
    require(sklearn.__version__=='1.6.1','Install scikit-learn==1.6.1 before running; do not load saved SVCs with another version')
    tf.config.set_visible_devices([],'GPU');tf.config.threading.set_intra_op_parallelism_threads(1);tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism();tf.config.experimental.enable_tensor_float_32_execution(False)
    require(not tf.config.get_visible_devices('GPU'),'GPU visible during CPU benchmark')
    pool_guard=threadpoolctl.threadpool_limits(limits=1)
    cpuinfo=Path('/proc/cpuinfo').read_text() if Path('/proc/cpuinfo').exists() else platform.processor()
    cpu_lines=sorted(set(s.strip() for s in cpuinfo.splitlines() if s.startswith(('model name','vendor_id','cpu family','model\t','stepping','flags'))))
    env=dict(python=sys.version,tensorflow=tf.__version__,keras=tf.keras.__version__,numpy=np.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,
        platform=platform.platform(),host=platform.node(),cpu_identification=cpu_lines,logical_cpus=os.cpu_count(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        cpu_only=True,threads=1,tf_intra_threads=tf.config.threading.get_intra_op_parallelism_threads(),tf_inter_threads=tf.config.threading.get_inter_op_parallelism_threads(),
        deterministic_operations=True,tf32=False,onednn=False,threadpools=threadpoolctl.threadpool_info(),
        clock=dict(implementation=time.get_clock_info('perf_counter').implementation,resolution=time.get_clock_info('perf_counter').resolution))
    env['threadpools']=canonical_threadpools(env['threadpools'])
    return tf,joblib,env,pool_guard

class Lite:
    def __init__(self,tf,path,shape,batch):
        self.interpreter=tf.lite.Interpreter(model_path=str(path),num_threads=1,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES)
        inp=self.interpreter.get_input_details();require(len(inp)==1,'Expected one model input')
        self.interpreter.resize_tensor_input(inp[0]['index'],[batch,*shape],strict=True);self.interpreter.allocate_tensors()
        self.input=self.interpreter.get_input_details()[0];self.output=self.interpreter.get_output_details()[0]
        require(self.input['dtype']==np.float32 and self.output['dtype']==np.float32,'Expected float32 model I/O')
        self.ops=[v['op_name'] for v in self.interpreter._get_ops_details()]
        require(not any(v=='DELEGATE' or v.startswith(('Flex','CUSTOM')) for v in self.ops),'Unexpected delegate/custom op')
    def logits(self,x):
        self.interpreter.set_tensor(self.input['index'],x);self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output['index'])
    def predict(self,x):return self.logits(x).argmax(axis=1)

def batched_keras(tf,model,x):
    return np.concatenate([model(tf.convert_to_tensor(x[i:i+128]),training=False).numpy() for i in range(0,len(x),128)])

def batched_lite(tf,path,shape,x,batch_size=128):
    # Dynamic final batch is allocated outside calls; this phase is parity, not timing.
    outputs=[];engines={}
    for i in range(0,len(x),batch_size):
        a=np.ascontiguousarray(x[i:i+batch_size],dtype=np.float32);b=len(a)
        if b not in engines:engines[b]=Lite(tf,path,shape,b)
        outputs.append(engines[b].logits(a))
    return np.concatenate(outputs)

def source_record(drive,frozen,ds,seed):return json.loads((drive/frozen['seed_records'][ds][str(seed)]).read_text())

def checkpoint(drive,frozen,ds,seed):return drive/frozen['students'][ds][str(seed)]

def gate_context(frozen,spec,env):return dict(frozen_sha256=digest(frozen),specification_sha256=digest(spec),environment_sha256=digest(env))

def require_gate(root,name,context):
    p=root/name;require(p.is_file(),f'Run the preceding phase first: {name}')
    gate=json.loads(p.read_text());require(gate['status']=='complete' and gate['context']==context,f'Changed session identity: {name}')
    verify_files(root,gate['files']);return gate

def prepare(drive,root,frozen,context,tf,joblib):
    require(not (root/'preflight-completed.json').exists(),'Preflight already exists. Use its verified cache or start a fresh session; do not overwrite it.')
    sys.path.insert(0,str(drive/'revision_2/pipeline_snapshot'));sys.path.insert(0,str(drive/'revision_2'))
    from run_coalition_noise import load_bundle
    from makd_har.config import ExperimentConfig
    from sklearn.svm import SVC
    files=[];svm_fits=0
    for ds in ('uci','wisdm'):
        source=source_record(drive,frozen,ds,11);bundle=load_bundle(drive,ds,source,ExperimentConfig(**source['config']))
        folder=root/ds;folder.mkdir(parents=True,exist_ok=True)
        arrays={f'{kind}_{part}':np.asarray(getattr(bundle,f'{kind}_{part}')) for kind in ('raw','features','y','subjects') for part in ('val','test')}
        save_npz(folder/'data.npz',**arrays);atomic_json(folder/'data.json',dict(manifest=bundle.manifest(),context=context,cache_sha256=sha(folder/'data.npz')))
        files.extend([folder/'data.npz',folder/'data.json']);reports={};predictions={}
        selected=joblib.load(drive/frozen['svms'][ds]);params=source['methods']['svm']['selected_params']
        require(all(selected.get_params()[k]==v for k,v in params.items()),'Saved selected SVM parameters differ')
        require(not selected.break_ties,'Original SVM voting rule changed')
        models={'selected':selected}
        if ds=='wisdm':
            choice=linear_choice(source);linear=SVC(**choice['params'],cache_size=4096,random_state=2026)
            linear.fit(bundle.features_train,bundle.y_train);svm_fits+=1
            source_parity(metrics(bundle.y_val,linear.predict(bundle.features_val)),dict(macro_f1=choice['validation_macro_f1']))
            joblib.dump(linear,folder/'linear-alternative.joblib',compress=0);files.append(folder/'linear-alternative.joblib');models['linear-alternative']=linear
        for name,m in models.items():
            pred=m.predict(bundle.features_val);predictions[name]=pred
            expected=dict(macro_f1=source['methods']['svm']['validation_macro_f1']) if name=='selected' else dict(macro_f1=linear_choice(source)['validation_macro_f1'])
            reports[name]=dict(params={k:m.get_params()[k] for k in ('C','kernel','gamma','break_ties','decision_function_shape')},validation=source_parity(metrics(bundle.y_val,pred),expected),
                support_vectors=int(len(m.support_vectors_)),numeric_state_bytes=int(sum(getattr(m,k).nbytes for k in ('support_vectors_','dual_coef_','intercept_','classes_','n_support_','probA_','probB_'))),
                classifier_only=True,scaling_and_feature_extraction_excluded=True)
            if m.kernel=='linear':
                state=export_linear(m);cp=ovo_predict(bundle.features_val,state);np.testing.assert_array_equal(cp,pred)
                path=folder/f'{name}-compact.npz';save_npz(path,compress=False,**state);files.append(path)
                predictions[name+'-compact']=cp;reports[name]['compact_validation_label_disagreements']=0
                reports[name]['compact_numeric_bytes']=sum(v.nbytes for v in state.values());reports[name]['compact_artifact_bytes']=path.stat().st_size
        save_npz(folder/'svm-validation-predictions.npz',**predictions);atomic_json(folder/'svm-preflight.json',dict(context=context,models=reports,svm_fits=1 if ds=='wisdm' else 0))
        files.extend([folder/'svm-validation-predictions.npz',folder/'svm-preflight.json'])
        shape=list(bundle.raw_val.shape[1:])
        for seed in SEEDS:
            current=source_record(drive,frozen,ds,seed)
            require(current['dataset_manifest']==source['dataset_manifest'],'Split differs by seed')
            path=checkpoint(drive,frozen,ds,seed);model=tf.keras.models.load_model(path,compile=False,safe_mode=True)
            require(list(model.input_shape[1:])==shape and model.output_shape[-1]==6,'Model architecture shape differs')
            keras=batched_keras(tf,model,bundle.raw_val);source_check=source_parity(metrics(bundle.y_val,keras.argmax(1)),current['methods']['proposed']['validation'])
            taskdir=folder/f'seed-{seed}';taskdir.mkdir(parents=True,exist_ok=True);flat=taskdir/'student-float32.tflite'
            with tempfile.TemporaryDirectory(prefix='revision2-export-') as temp:
                model.export(temp,format='tf_saved_model',verbose=False,input_signature=[tf.TensorSpec([None,*shape],tf.float32,name='raw_window')])
                converter=tf.lite.TFLiteConverter.from_saved_model(temp);converter.optimizations=[]
                converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS];converter.allow_custom_ops=False
                flat.write_bytes(converter.convert())
            lite=batched_lite(tf,flat,shape,bundle.raw_val);conversion=converted_parity(keras,lite)
            lite1=batched_lite(tf,flat,shape,bundle.raw_val,batch_size=1);conversion1=converted_parity(keras,lite1)
            # Check the traced reconciliation path too, before any timing.
            traced=tf.function(lambda x:model(x,training=False),input_signature=[tf.TensorSpec([None,*shape],tf.float32)],autograph=False)
            trace=np.concatenate([traced(tf.convert_to_tensor(bundle.raw_val[i:i+128])).numpy() for i in range(0,len(bundle.raw_val),128)])
            trace_check=converted_parity(keras,trace)
            save_npz(taskdir/'validation.npz',keras_logits=keras,lite_logits=lite,lite_batch1_logits=lite1,traced_logits=trace)
            report=dict(context=context,source_model_sha256=sha(path),validation_source_parity=source_check,conversion=conversion,conversion_batch1=conversion1,traced=trace_check,
                tflite_sha256=sha(flat),keras_checkpoint_bytes=path.stat().st_size,tflite_artifact_bytes=flat.stat().st_size,
                keras_parameter_count=int(model.count_params()),keras_trainable_parameters=int(sum(np.prod(v.shape) for v in model.trainable_weights)),
                keras_numeric_weight_bytes=int(sum(v.numpy().nbytes for v in model.weights)),ops=Lite(tf,flat,shape,1).ops,
                architecture=model.get_config(),training_fits=0)
            atomic_json(taskdir/'preflight.json',report);files.extend([flat,taskdir/'validation.npz',taskdir/'preflight.json'])
            del model;tf.keras.backend.clear_session();print(f'PREFLIGHT {ds} seed {seed}: original validation, TFLite and traced predictions pass.',flush=True)
        del bundle
    atomic_json(root/'preflight-completed.json',dict(status='complete',context=context,students=20,svm_fits=svm_fits,neural_training_fits=0,test_evaluated=False,files=records(root,files)))
    print('PREFLIGHT COMPLETE: twenty conversions and SVM representations passed; one auxiliary linear SVC fit, zero neural training.',flush=True)

def load_arrays(root,ds):
    with np.load(root/ds/'data.npz',allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}

def svm_cases(drive,root,frozen,ds,joblib):
    selected=joblib.load(drive/frozen['svms'][ds]);cases=[('svm-selected',selected.predict,'features',dict(artifact_bytes=(drive/frozen['svms'][ds]).stat().st_size))]
    models={'selected':selected}
    if ds=='wisdm':
        models['linear-alternative']=joblib.load(root/ds/'linear-alternative.joblib')
        cases.append(('svm-linear-alternative',models['linear-alternative'].predict,'features',dict(artifact_bytes=(root/ds/'linear-alternative.joblib').stat().st_size)))
    for name,m in models.items():
        if m.kernel=='linear':
            with np.load(root/ds/f'{name}-compact.npz',allow_pickle=False) as z:state={k:z[k].copy() for k in z.files}
            cases.append((f'svm-{name}-compact',lambda x,s=state:ovo_predict(x,s),'features',dict(artifact_bytes=(root/ds/f'{name}-compact.npz').stat().st_size)))
    return cases

def research(drive,root,frozen,context,tf,joblib,smoke=False):
    require_gate(root,'preflight-completed.json',context)
    if not smoke:require_gate(root,'smoke-completed.json',context)
    marker=root/('smoke-completed.json' if smoke else 'completed.json')
    if marker.exists():
        require_gate(root,marker.name,context);print('Existing complete phase verified; timings will not be overwritten.',flush=True);return
    mode='smoke' if smoke else 'research';run=root/mode/str(uuid.uuid4());run.mkdir(parents=True)
    cases=[];files=[];metric_rows=[];data={};timing_inputs={}
    for ds in ('uci','wisdm'):
        a=load_arrays(root,ds);data[ds]=a;plan=timing_plan(len(a['y_val']));shape=list(a['raw_val'].shape[1:]);source=source_record(drive,frozen,ds,11)
        corpus={k:v.tolist() for k,v in plan.items()};atomic_json(run/f'{ds}-timing-corpus.json',dict(validation_indices=corpus,subject_ids={k:a['subjects_val'][v].tolist() for k,v in plan.items()},context=context));files.append(run/f'{ds}-timing-corpus.json')
        predicted={}
        for name,fn,kind,size in svm_cases(drive,root,frozen,ds,joblib):
            x=np.ascontiguousarray(a['features_test'],dtype=np.float64);pred=fn(x);predicted[name]=pred
            if name=='svm-selected':source_parity(metrics(a['y_test'],pred),source['methods']['svm']['test'])
            if name=='svm-selected-compact':np.testing.assert_array_equal(pred,predicted['svm-selected'])
            if name=='svm-linear-alternative-compact':
                original=joblib.load(root/ds/'linear-alternative.joblib');np.testing.assert_array_equal(pred,original.predict(x))
            metric_rows.append(dict(dataset=ds,condition=name,seed=None,partition='test',metrics=metrics(a['y_test'],pred),**size))
            cases.append(dict(key=f'{ds}/{name}',dataset=ds,condition=name,seed=None,fn=fn,kind='features',prepare=lambda x:np.ascontiguousarray(x,dtype=np.float64)))
        save_npz(run/f'{ds}-svm-test.npz',y_test=a['y_test'],subjects_test=a['subjects_test'],**predicted);files.append(run/f'{ds}-svm-test.npz')
        for seed in ([11] if smoke else SEEDS):
            model=tf.keras.models.load_model(checkpoint(drive,frozen,ds,seed),compile=False,safe_mode=True);flat=root/ds/f'seed-{seed}/student-float32.tflite'
            keras=batched_keras(tf,model,a['raw_test']);lite=batched_lite(tf,flat,shape,a['raw_test'])
            lite1=batched_lite(tf,flat,shape,a['raw_test'],batch_size=1);conversion1=converted_parity(keras,lite1)
            current=source_record(drive,frozen,ds,seed);check=source_parity(metrics(a['y_test'],keras.argmax(1)),current['methods']['proposed']['test']);conversion=converted_parity(keras,lite)
            traced=tf.function(lambda x,m=model:m(x,training=False),input_signature=[tf.TensorSpec([None,*shape],tf.float32)],autograph=False)
            tr=np.concatenate([traced(tf.convert_to_tensor(a['raw_test'][i:i+128])).numpy() for i in range(0,len(a['y_test']),128)]);trace_check=converted_parity(keras,tr)
            path=run/ds/f'seed-{seed}/test.npz';save_npz(path,y_test=a['y_test'],subjects_test=a['subjects_test'],keras_logits=keras,lite_logits=lite,lite_batch1_logits=lite1,traced_logits=tr);files.append(path)
            check_path=path.with_suffix('.json');atomic_json(check_path,dict(context=context,source_parity=check,conversion=conversion,conversion_batch1=conversion1,traced=trace_check));files.append(check_path)
            for name,logits in [('tflite',lite),('keras-eager',keras),('tf-function',tr)]:
                metric_rows.append(dict(dataset=ds,condition=name,seed=seed,partition='test',metrics=metrics(a['y_test'],logits.argmax(1))))
            engines={b:Lite(tf,flat,shape,b) for b in (1,128)}
            def lite_fn(x,e=engines):return e[len(x)].predict(x)
            cases.extend([
                dict(key=f'{ds}/tflite/{seed}',dataset=ds,condition='tflite',seed=seed,fn=lite_fn,kind='raw',prepare=lambda x:np.ascontiguousarray(x,dtype=np.float32)),
                dict(key=f'{ds}/keras-eager/{seed}',dataset=ds,condition='keras-eager',seed=seed,fn=lambda x,m=model:m(x,training=False).numpy().argmax(1),kind='raw',prepare=tf.convert_to_tensor),
                dict(key=f'{ds}/tf-function/{seed}',dataset=ds,condition='tf-function',seed=seed,fn=lambda x,f=traced:f(x).numpy().argmax(1),kind='raw',prepare=tf.convert_to_tensor)])
            print(f'TEST PARITY {ds} seed {seed}: original metrics and all runtime labels pass.',flush=True)
    # Materialize all timing inputs before starting clocks, and retain full raw samples.
    for case in cases:
        a=data[case['dataset']];plan=timing_plan(len(a['y_val']));x=a[case['kind']+'_val'];prep=case['prepare']
        if case['kind']=='raw':
            with np.load(root/case['dataset']/f'seed-{case["seed"]}/validation.npz',allow_pickle=False) as z:reference=z['keras_logits'].argmax(1)
        else:
            name=case['condition'][len('svm-'):]
            with np.load(root/case['dataset']/'svm-validation-predictions.npz',allow_pickle=False) as z:reference=z[name].copy()
        timing_inputs[case['key']]=dict(warm1=[prep(x[i:i+1]) for i in plan['warm1']],sample1=[prep(x[i:i+1]) for i in plan['sample1']],
            warm128=prep(x[plan['warm128']]),sample128=prep(x[plan['sample128']]),expected1=reference[plan['sample1']],expected128=reference[plan['sample128']])
    rounds=1 if smoke else 3;times=[];round_orders=[]
    for round_index in range(rounds):
        order=np.random.default_rng(20261019+round_index).permutation(len(cases));round_orders.append([cases[i]['key'] for i in order])
        for idx in order:
            c=cases[idx];inp=timing_inputs[c['key']];fn=c['fn'];samples={}
            for x in inp['warm1']:fn(x)
            count=10 if smoke else 300
            ns=[]
            for sample_index,x in enumerate(inp['sample1'][:count]):
                start=time.perf_counter_ns();result=fn(x);elapsed=time.perf_counter_ns()-start
                require(np.asarray(result).shape==(1,) and int(result[0])==int(inp['expected1'][sample_index]),'Timed batch-one label differs from verified validation predictions');ns.append(elapsed)
            samples['batch1_ns']=ns
            for _ in range(5):fn(inp['warm128'])
            ns=[]
            for _ in range(3 if smoke else 30):
                start=time.perf_counter_ns();result=fn(inp['sample128']);elapsed=time.perf_counter_ns()-start
                np.testing.assert_array_equal(result,inp['expected128']);ns.append(elapsed)
            samples['batch128_ns']=ns
            row=dict(key=c['key'],dataset=c['dataset'],condition=c['condition'],seed=c['seed'],round=round_index,**samples,
                batch1_summary=timing_summary(samples['batch1_ns'],1),batch128_summary=timing_summary(samples['batch128_ns'],128))
            times.append(row);print(f'TIMED {mode} round {round_index+1}: {c["key"]}',flush=True)
    report=run/'results.json';atomic_json(report,dict(context=context,mode=mode,timing_process_pid=os.getpid(),session_id=run.name,
        metrics=metric_rows,timings=times,round_orders=round_orders,case_keys=[c['key'] for c in cases],neural_training_fits=0,svm_fits_in_timing_phase=0))
    files.append(report);expected=11 if smoke else 65
    require(len(cases)==expected and len(metric_rows)==expected and len(times)==rounds*expected,'Incomplete condition set')
    atomic_json(marker,dict(status='complete',context=context,mode=mode,student_checkpoints=2 if smoke else 20,conditions=expected,timing_records=len(times),
        neural_training_fits=0,auxiliary_svm_fits=1,test_evaluated=True,results_file=report.relative_to(root).as_posix(),files=records(root,files)))
    print(f'COMPLETE {mode}: {expected} conditions, {rounds} timing rounds, zero neural training. Results: {root}',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--drive-root',type=Path,required=True);p.add_argument('--session',type=Path)
    g=p.add_mutually_exclusive_group();g.add_argument('--verify-only',action='store_true');g.add_argument('--prepare',action='store_true');g.add_argument('--smoke',action='store_true')
    args=p.parse_args();drive=args.drive_root.resolve();frozen=verify_inputs(drive);spec=protocol(frozen)
    print(f'Frozen inputs verified: {len(frozen["files"])}. Zero teacher/student training.',flush=True)
    if args.verify_only:return
    require(args.session is not None,'Pass a fresh session directory');root=args.session.resolve()
    require((drive/'results/revision-2/runtime-benchmark').resolve() in root.parents,'Session must be inside results/revision-2/runtime-benchmark/')
    root.mkdir(parents=True,exist_ok=True);tf,joblib,env,pool_guard=initialize_runtime();context=gate_context(frozen,spec,env)
    for name,value in [('protocol.json',spec),('environment.json',env)]:
        path=root/name
        if path.exists():require(json.loads(path.read_text())==value,f'Changed environment/protocol at {path}; start a fresh session')
        else:atomic_json(path,value)
    if args.prepare:prepare(drive,root,frozen,context,tf,joblib)
    else:research(drive,root,frozen,context,tf,joblib,smoke=args.smoke)

if __name__=='__main__':main()
