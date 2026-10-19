"""Independently verify the complete training-only pilot before writing analysis outputs."""
import argparse
import csv
import dataclasses
import hashlib
import itertools
import json
import math
import shutil
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

NAMES=['uniform','standardized','raw-tau-0p001','raw-tau-0p01','raw-tau-0p1']
SEEDS=[11,23,37]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def metrics(truth,pred):
    assert truth.ndim==pred.ndim==1 and len(truth)==len(pred)>0
    assert np.issubdtype(truth.dtype,np.integer) and np.issubdtype(pred.dtype,np.integer)
    assert np.isin(truth,range(6)).all() and np.isin(pred,range(6)).all()
    matrix=np.bincount(truth*6+pred,minlength=36).reshape(6,6)
    tp=np.diag(matrix).astype(float);support=matrix.sum(axis=1);predicted=matrix.sum(axis=0)
    f1=np.divide(2*tp,support+predicted,out=np.zeros(6),where=support+predicted>0)
    precision=np.divide(tp,predicted,out=np.zeros(6),where=predicted>0)
    recall=np.divide(tp,support,out=np.zeros(6),where=support>0)
    return dict(accuracy=float(tp.sum()/len(truth)),macro_f1=float(f1.mean()),macro_precision=float(precision.mean()),
                macro_recall=float(recall.mean()),per_class_f1=f1.tolist())


def independent_phi(utilities):
    result=[]
    for i in range(5):
        others=[j for j in range(5) if j!=i];value=0.
        for size in range(5):
            for coalition in itertools.combinations(others,size):
                value+=math.factorial(size)*math.factorial(4-size)/math.factorial(5)*(utilities[tuple(sorted((*coalition,i)))]-utilities[coalition])
        result.append(value)
    return np.array(result)


def independent_weights(phi):
    def softmax(z):
        exp=np.exp(z-z.max());return exp/exp.sum()
    z=np.zeros(5) if phi.std()<=1e-12 else (phi-phi.mean())/phi.std()
    return dict(uniform=np.ones(5)/5,standardized=softmax(z),**{name:softmax(phi/tau) for name,tau in zip(NAMES[2:],[.001,.01,.1])})


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--drive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();drive=args.drive.resolve();root=drive/'results/revision-2/weighting-pilot'
    if not (root/'completed.json').is_file():raise SystemExit('Weighting pilot incomplete; no analysis written.')
    package=drive/'revision_2/weighting_pilot';sys.path.insert(0,str(package))
    from run_pilot import verify_inputs,protocol,history_check
    from pilot_data import ROLES,validate_groups
    from makd_har.config import ExperimentConfig
    frozen=verify_inputs(drive);spec=protocol(frozen);env=json.loads((root/'environment.json').read_text())
    assert json.loads((root/'experiment-protocol.json').read_text())==spec
    assert env['deterministic_operations'] is True and env['tensor_float_32_execution'] is False
    completed=json.loads((root/'completed.json').read_text());gate=json.loads((root/'preflight-completed.json').read_text())
    for marker in (completed,gate):
        assert marker['specification_sha256']==digest(spec) and marker['environment_sha256']==digest(env)
    assert completed['status']=='complete' and completed['mode']=='research'
    assert (completed['fits'],completed['teacher_fits'],completed['student_fits'])==(252,30,222)
    assert completed['test_evaluated'] is False and completed['original_validation_evaluated'] is False
    assert gate['passed'] and len(gate['files'])==4
    checked=[];snapshots={root/'completed.json',root/'preflight-completed.json',root/'experiment-protocol.json',root/'environment.json'}
    def verify_list(items):
        for item in items:
            path=(drive/item['path']).resolve();assert root in path.parents and sha(path)==item['sha256']
            checked.append(dict(path=str(path),sha256=item['sha256']))
            if path.suffix=='.json':snapshots.add(path)
    verify_list(gate['files']);verify_list(completed['datasets'])
    assert {i['path'] for i in completed['datasets']}=={f'results/revision-2/weighting-pilot/{ds}/completed.json' for ds in ('uci','wisdm')}
    fit_rows=[];candidate_rows=[];value_rows=[];weight_rows=[];teacher_values=[];bank_rows=[];selection_records={};metric_count=0
    for dataset in ('uci','wisdm'):
        dsroot=root/dataset;marker=json.loads((dsroot/'completed.json').read_text())
        assert (marker['fits'],marker['teacher_fits'],marker['student_fits'])==(126,15,111)
        assert marker['status']=='complete' and marker['mode']=='research' and marker['dataset']==dataset
        assert marker['test_evaluated'] is False and marker['original_validation_evaluated'] is False
        assert marker['specification_sha256']==digest(spec) and marker['environment_sha256']==digest(env)
        assert len(marker['files'])==388 and len({r['path'] for r in marker['files']})==388
        expected_paths={f'results/revision-2/weighting-pilot/{dataset}/selection.json'}
        source=json.loads((drive/f"results/{'uci/production' if dataset=='uci' else 'wisdm/production-v2'}/seed-11.json").read_text())
        source_names=source['teacher_order']
        for seed in SEEDS:
            prefix=f'results/revision-2/weighting-pilot/{dataset}/seed-{seed}/'
            expected_paths|={prefix+'summary.json',prefix+'teacher-probabilities.json',prefix+'teacher-probabilities.npz'}
            stems=[f'teachers/{n}' for n in source_names]+[f'candidates/{n}' for n in NAMES]
            stems += ['coalitions/'+('-'.join(str(i) for i in range(5) if mask&(1<<i)) if mask else 'empty') for mask in range(32)]
            expected_paths|={prefix+stem+ext for stem in stems for ext in ('.json','.keras','.npz')}
        assert {item['path'] for item in marker['files']}==expected_paths
        verify_list(marker['files'])
        data_info=json.loads((dsroot/'inner-data.json').read_text())
        assert data_info['context']==dict(dataset=dataset,split=frozen['splits'][dataset],environment=env,
            frozen_manifest_sha256=sha(package/'frozen-inputs.json'))
        assert data_info['sha256']==sha(dsroot/'inner-data.npz') and data_info['scalers_fitted_on']=='fit role only'
        assert data_info['test_evaluated'] is False
        with np.load(dsroot/'inner-data.npz',allow_pickle=False) as cache:
            y=np.concatenate([cache[f'{r}_y'] for r in ROLES]);s=np.concatenate([cache[f'{r}_subjects'] for r in ROLES])
            validate_groups(y,s,frozen['splits'][dataset]);labels={r:cache[f'{r}_y'].copy() for r in ROLES}
            for role in ROLES:
                assert set(np.unique(cache[f'{role}_subjects']))==set(frozen['splits'][dataset]['groups'][role])
            assert np.abs(cache['fit_raw'].mean(axis=(0,1),dtype=np.float64)).max()<1e-6
            assert np.abs(cache['fit_features'].mean(axis=0,dtype=np.float64)).max()<1e-6
        for seed in SEEDS:
            seedroot=dsroot/f'seed-{seed}';summary=json.loads((seedroot/'summary.json').read_text());context=summary['context']
            assert context['dataset']==dataset and context['outer_seed']==seed and context['mode']=='research'
            # The older UCI source omitted protocol_version; the frozen constructor supplies its default.
            assert context['source_config']==json.loads(json.dumps(dataclasses.asdict(ExperimentConfig(**source['config']))))
            assert context['environment']==env and context['specification']==spec and context['data_sha256']==sha(dsroot/'inner-data.npz')
            assert summary['test_evaluated'] is False and summary['original_validation_evaluated'] is False and summary['smoke_values_not_research'] is False
            names=summary['teacher_order'];assert names==context['source_config']['teacher_names'] and len(names)==5
            utilities={};candidate_scores={};model_hashes={}
            fits=[seedroot/'teachers'/f'{name}.json' for name in names]
            coalitions=[tuple(i for i in range(5) if mask&(1<<i)) for mask in range(32)]
            fits += [seedroot/'coalitions'/f"{'-'.join(map(str,c)) if c else 'empty'}.json" for c in coalitions]
            fits += [seedroot/'candidates'/f'{name}.json' for name in NAMES]
            for path in fits:
                report=json.loads(path.read_text());task=report['task'];kind=task['kind']
                assert task['context']==context and task['epochs']==60 and report['identity']==digest(task) and report['status']=='complete'
                assert report['test_evaluated'] is False and report['original_validation_evaluated'] is False
                history_check(report,task)
                assert report['model_file']==path.with_suffix('.keras').name and report['predictions_file']==path.with_suffix('.npz').name
                for key in ('model','predictions'):assert sha(path.parent/report[key+'_file'])==report[key+'_sha256']
                role=None if kind=='teacher' else ('value' if kind=='coalition' else 'select')
                assert kind in ('teacher','coalition','candidate')
                expected_keys={'stop_y','stop_predictions'}|(set() if role is None else {'evaluation_y','evaluation_predictions'})
                with np.load(path.parent/report['predictions_file'],allow_pickle=False) as arrays:
                    assert set(arrays.files)==expected_keys
                    for label,r in [('stop','stop')]+([] if role is None else [('evaluation',role)]):
                        np.testing.assert_array_equal(arrays[label+'_y'],labels[r]);observed=metrics(arrays[label+'_y'],arrays[label+'_predictions'])
                        for key in observed:np.testing.assert_allclose(observed[key],report[label+'_metrics'][key],rtol=0,atol=1e-12)
                        metric_count+=1
                fit_rows.append(dict(dataset=dataset,outer_seed=seed,kind=kind,name=task['name'],epochs=report['epochs_completed'],
                    best_epoch=report['best_epoch'],elapsed_seconds=report['elapsed_seconds'],stop_macro_f1_pct=report['stop_metrics']['macro_f1']*100,
                    evaluation_role=role or '',evaluation_macro_f1_pct='' if role is None else report['evaluation_metrics']['macro_f1']*100))
                if kind=='teacher':
                    assert task['name'] in names and task['seed']==seed*100+names.index(task['name']);model_hashes[task['name']]=report['model_sha256']
                elif kind=='coalition':
                    assert task['seed']==seed*100000+17 and task['evaluation_role']=='value'
                    c=() if task['name']=='empty' else tuple(map(int,task['name'].split('-')))
                    assert task['teachers']==[names[i] for i in c] and task['weights']==([1/len(c)]*len(c) if c else None)
                    utilities[c]=report['evaluation_metrics']['macro_f1'];value_rows.append(dict(dataset=dataset,outer_seed=seed,coalition=task['name'],value_macro_f1_pct=utilities[c]*100))
                else:
                    assert task['seed']==seed*1000000+7 and task['evaluation_role']=='select' and task['teachers']==names
                    candidate_scores[task['name']]=report['evaluation_metrics']['macro_f1']
                    candidate_rows.append(dict(dataset=dataset,outer_seed=seed,candidate=task['name'],selection_macro_f1_pct=candidate_scores[task['name']]*100,
                        stop_macro_f1_pct=report['stop_metrics']['macro_f1']*100))
            phi=independent_phi(utilities);np.testing.assert_allclose(phi,summary['shapley']['values'],rtol=0,atol=1e-12)
            bank_info=json.loads((seedroot/'teacher-probabilities.json').read_text())
            assert bank_info['identity']==digest(dict(context=context,models=model_hashes))
            assert bank_info['sha256']==sha(seedroot/'teacher-probabilities.npz')
            assert bank_info['teacher_checkpoint_parity']=='Exact saved stop-label sequence for all five teachers'
            with np.load(seedroot/'teacher-probabilities.npz',allow_pickle=False) as bank:
                assert set(bank.files)=={f'{n}_{r}' for n in names for r in ('fit','stop')}
                for n in names:
                    for r in ('fit','stop'):
                        v=bank[f'{n}_{r}'];assert v.shape==(len(labels[r]),6) and np.isfinite(v).all() and (v>=0).all()
                        np.testing.assert_allclose(v.sum(axis=1),1,rtol=0,atol=1e-5)
            for path in seedroot.glob('coalitions/*.json'):
                assert json.loads(path.read_text())['task']['teacher_models_sha256']==digest(model_hashes)
            bank_rows.append(dict(dataset=dataset,outer_seed=seed,raw_phi_sd=float(phi.std()),
                raw_phi_sd_pp=float(phi.std()*100),empty_value_pct=utilities[()]*100,full_value_pct=utilities[(0,1,2,3,4)]*100))
            teacher_values.extend(dict(dataset=dataset,outer_seed=seed,teacher=n,raw_phi=float(v),raw_phi_pp=float(v*100)) for n,v in zip(names,phi))
            assert abs(sum(phi)-(utilities[(0,1,2,3,4)]-utilities[()]))<1e-10
            saved_utilities={tuple(map(int,k.split(','))) if k else ():v for k,v in summary['shapley']['coalition_values'].items()}
            assert saved_utilities==utilities
            expected_weights=independent_weights(phi)
            for name,w in expected_weights.items():
                np.testing.assert_allclose(w,summary['weights'][name],rtol=0,atol=1e-12)
                candidate=json.loads((seedroot/'candidates'/f'{name}.json').read_text())
                np.testing.assert_allclose(w,candidate['task']['weights'],rtol=0,atol=1e-12)
                assert candidate['task']['teacher_models_sha256']==digest(model_hashes)
                assert candidate['task']['valuation_game_sha256']==digest({','.join(map(str,k)):v for k,v in utilities.items()})
                for key in ('selection_metrics','stop_metrics'):
                    recorded=candidate['evaluation_metrics' if key=='selection_metrics' else 'stop_metrics']
                    for metric in recorded:np.testing.assert_allclose(summary['candidates'][name][key][metric],recorded[metric],rtol=0,atol=1e-12)
                diagnostics=summary['weight_diagnostics'][name]
                np.testing.assert_allclose(diagnostics['normalized_entropy'],-np.sum(w*np.log(np.maximum(w,1e-15)))/np.log(5),rtol=0,atol=1e-12)
                np.testing.assert_allclose(diagnostics['effective_teacher_count'],1/np.sum(w*w),rtol=0,atol=1e-12)
                weight_rows.append(dict(dataset=dataset,outer_seed=seed,candidate=name,**{f'weight_{n}':float(v) for n,v in zip(names,w)},
                    normalized_entropy=float(-np.sum(w*np.log(np.maximum(w,1e-300)))/np.log(5)),effective_teachers=float(1/np.sum(w*w))))
        selection=json.loads((dsroot/'selection.json').read_text());means={name:np.mean([r['selection_macro_f1_pct']/100 for r in candidate_rows if r['dataset']==dataset and r['candidate']==name]) for name in NAMES}
        chosen=max(NAMES,key=lambda n:(round(means[n],12),-NAMES.index(n)))
        assert selection['selected_candidate']==chosen and selection['test_evaluated'] is False and selection['original_validation_evaluated'] is False
        for name in NAMES:np.testing.assert_allclose(means[name],selection['selection_macro_f1_means'][name],rtol=0,atol=1e-12)
        selection_records[dataset]=selection
    assert len(fit_rows)==252 and len(candidate_rows)==30 and len(value_rows)==192 and len(weight_rows)==30 and metric_count==474
    paired=[];aggregate=[]
    for dataset in ('uci','wisdm'):
        uniform={r['outer_seed']:r['selection_macro_f1_pct'] for r in candidate_rows if r['dataset']==dataset and r['candidate']=='uniform'}
        for name in NAMES:
            rows=[r for r in candidate_rows if r['dataset']==dataset and r['candidate']==name]
            scores=np.array([r['selection_macro_f1_pct'] for r in rows]);differences=scores-np.array([uniform[r['outer_seed']] for r in rows])
            aggregate.append(dict(dataset=dataset,candidate=name,mean_selection_macro_f1_pct=float(scores.mean()),seed_sd_pp=float(scores.std(ddof=1)),
                mean_difference_vs_uniform_pp=float(differences.mean()),sd_difference_vs_uniform_pp=float(differences.std(ddof=1)),
                wins=int((differences>0).sum()),ties=int((differences==0).sum()),losses=int((differences<0).sum()),selected=selection_records[dataset]['selected_candidate']==name))
            if name!='uniform':
                paired.extend(dict(dataset=dataset,outer_seed=r['outer_seed'],candidate=name,difference_vs_uniform_pp=float(d)) for r,d in zip(rows,differences))
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('fits',fit_rows),('candidates',candidate_rows),('coalition-values',value_rows),('weights',weight_rows),
                      ('paired-differences',paired),('summary',aggregate),('teacher-values',teacher_values),('games',bank_rows)]:
        write_csv(output/f'weighting-pilot-{name}.csv',rows)
    lines=['# Training-only weighting pilot','','Verified all 252 fits (30 teachers and 222 students), frozen files, recorded output hashes, '
        'four-role subject exclusions, fit-only normalization, 474 metric reconstructions, six exact games, thirty weight vectors and both selections.',
        '','| Dataset | Candidate | Mean selection macro-F1 (%) | Seed SD (pp) | Selected |','|---|---|---:|---:|---|']
    for dataset in ('uci','wisdm'):
        for name in NAMES:
            scores=[r['selection_macro_f1_pct'] for r in candidate_rows if r['dataset']==dataset and r['candidate']==name]
            lines.append(f"| {dataset.upper()} | {name} | {np.mean(scores):.3f} | {np.std(scores,ddof=1):.3f} | {'Yes' if selection_records[dataset]['selected_candidate']==name else ''} |")
    lines+=['','These are development outcomes from three seeds on one fixed four-role subject partition per dataset. '
        'The selected score is optimistic for generalization. No original validation/test outcomes were evaluated. '
        'No significance claim or independent generalization improvement is established. Keep all five candidates and original main results.','',
        f"Analysis date: {datetime.now(ZoneInfo('Asia/Tehran')).date().isoformat()}. Manuscript integration pending.",'']
    lines+=['## Paired selection differences','','All differences below compare the same seed and fitting protocol. Units are macro-F1 percentage points; three seeds support descriptive summaries only.',
        '','| Dataset | Candidate | Mean difference vs uniform (pp) | Wins/ties/losses |','|---|---|---:|---:|']
    for r in aggregate:
        if r['candidate']!='uniform':lines.append(f"| {r['dataset'].upper()} | {r['candidate']} | {r['mean_difference_vs_uniform_pp']:+.3f} | {r['wins']}/{r['ties']}/{r['losses']} |")
    lines+=['','## Weight concentration and scale','','| Dataset | Candidate | Entropy range (0–1) | Effective teacher count range (1–5) |',
        '|---|---|---:|---:|']
    for dataset in ('uci','wisdm'):
        for name in NAMES:
            rows=[r for r in weight_rows if r['dataset']==dataset and r['candidate']==name]
            entropy=[r['normalized_entropy'] for r in rows];effective=[r['effective_teachers'] for r in rows]
            lines.append(f"| {dataset.upper()} | {name} | {min(entropy):.3f}–{max(entropy):.3f} | {min(effective):.3f}–{max(effective):.3f} |")
    lines+=['','Effective teacher count is 1/sum(weight squared): it is five for uniform weights and approaches one for a single dominant teacher. '
        'For SD(phi)>1e-12, softmax((phi − mean(phi))/SD(phi)) equals softmax(phi/SD(phi)). '
        'Thus the standardized rule behaves like raw softmax with a bank-specific temperature equal to raw Shapley SD; '
        'this algebra does not make that SD an uncertainty estimate. Raw temperature 0.1 is much smoother here, '
        'whereas 0.001 concentrates almost all weight on one teacher for several banks. Temperature choice matters; '
        'unstandardized weighting is not automatically an improvement.','',
        '## Interpretation for the revision','',
        'The frozen selection rule chooses uniform weights on UCI-HAR (74.609% mean selection macro-F1) and raw-Shapley '
        'temperature 0.1 on WISDM (79.935%). The WISDM selected candidate exceeds uniform by 2.123 pp on average '
        '(seed differences +1.582, +4.965 and −0.178 pp), winning in two of three seeds. It exceeds standardized '
        'weighting by 1.847 pp on average. On UCI-HAR, standardized weighting is 0.226 pp below uniform on average. '
        'These are numerical development differences, with no significance claim.','',
        'The inner fitting, checkpoint-stop, coalition-value and candidate-selection subjects are disjoint and contain '
        'all six classes. The fitting groups contain 8 UCI subjects (2640 windows) and 13 WISDM subjects (6191 windows). '
        'The original validation/test subjects and production checkpoints are excluded. The pilot therefore changes '
        'training population and development protocol: its scores cannot be compared directly with the original main test table. '
        'Candidate selection still creates optimism on the inner-selection set, and the three seeds share one subject split.','',
        'UCI candidate stop scores are approximately 97.5–98.9%, while selection scores are approximately 72.7–76.5%. '
        'WISDM candidate stop scores are approximately 66.0–73.4%, while selection scores are approximately 71.5–83.8%. '
        'Different subject groups can behave very differently; these observations do not isolate a causal source of the gaps '
        'and do not explain away the original WISDM validation–test gap. That original split/error audit remains pending.','',
        'Keep the original ten-seed main results and report this block as an additional training-only sensitivity/selection study. '
        'Record both pilot choices, retain all five candidates and qualify claims of a generally superior weighting rule. '
        'Adopting a different main weighting protocol would require consistently regenerated matched main comparisons; '
        'do not replace only a favorable row using these development results. No further major training is currently planned.','',
        '## Proposed manuscript text','',
        'To examine sensitivity to Shapley scaling, we compared uniform weights, standardized-Shapley softmax '
        '(weight temperature 1), and raw-Shapley softmax at temperatures 0.001, 0.01 and 0.1, with KD temperature '
        'fixed at 4. Using only original training subjects, we established separate subject groups for fitting, '
        'checkpoint stopping, coalition valuation and candidate selection, and trained fresh teacher banks for '
        'three seeds per dataset. All 32 coalition utilities were evaluated on valuation subjects after checkpoint '
        'selection on stopping subjects. The predefined mean-selection-macro-F1 rule selected uniform weighting '
        'on UCI-HAR (74.609%) and raw-Shapley temperature 0.1 on WISDM (79.935%). Corresponding standardized-rule '
        'means were 74.382% and 78.088%; the WISDM uniform mean was 77.812%. These exploratory development '
        'results demonstrate sensitivity to weight scaling and do not establish a significant or independent test '
        'advantage. We retain the original main test results and explicitly acknowledge the optimism of candidate '
        'selection and limitations of the original validation reuse.','',
        '## Proposed response content','',
        'We added the requested discussion of magnitude removal and evaluated unstandardized temperature alternatives. '
        'The training-only pilot separates fitting, checkpoint selection, coalition valuation and weighting selection '
        'by subject, retrains teachers without the held-out development subjects, and reports all five candidates '
        'across three seeds. We record dataset-specific pilot choices and avoid interpreting their selected scores '
        'as independent generalization evidence. The standardized transformation can sharpen small differences; '
        'the final revision must include this limitation and correct the Eq. 8 typo. These manuscript changes and '
        'final section/table references remain pending integration.','',
        f"Summed recorded fit training/validation/evaluation time: {sum(r['elapsed_seconds'] for r in fit_rows)/3600:.3f} hours; "
        'this excludes some setup/serialization and is not billed GPU time.','']
    (output/'weighting-pilot-report.md').write_text('\n'.join(lines),encoding='utf-8')
    plot_results(output,candidate_rows,weight_rows,selection_records)
    for name in ('run_pilot.py','pilot_data.py','test_pilot_guards.py','frozen-inputs.json','uci-split.json','wisdm-split.json'):snapshots.add(package/name)
    saved=[]
    for path in sorted(snapshots):
        destination=output/'evidence_snapshot/weighting-pilot'/path.relative_to(drive);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        saved.append(dict(source=str(path),snapshot=str(destination.relative_to(output)),sha256=sha(path)))
    audit=dict(fits_verified=252,models_verified=252,prediction_files_verified=252,frozen_inputs_verified=len(frozen['files']),
        metric_reconstructions=474,game_reconstructions=6,weight_vectors=30,checked_hashes=checked,snapshots=saved,
        script_sha256=sha(Path(__file__)),outputs={p.name:sha(p) for p in output.glob('weighting-pilot-*') if p.suffix in ('.csv','.md')})
    audit['outputs'].update({name:sha(output/name) for name in ('weighting-pilot.png','weighting-pilot.pdf')})
    (output/'weighting-pilot-verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(selection_records,indent=2))


def plot_results(output,candidates,weights,selections):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['pdf.fonttype']=42
    labels=['Uniform','Standardized','Raw T=0.001','Raw T=0.01','Raw T=0.1']
    fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
    colors=['#276493','#a65032','#397d58'];x=np.arange(5)
    for column,dataset in enumerate(('uci','wisdm')):
        for seed,color in zip(SEEDS,colors):
            scores=[next(r['selection_macro_f1_pct'] for r in candidates if r['dataset']==dataset and r['outer_seed']==seed and r['candidate']==name) for name in NAMES]
            effective=[next(r['effective_teachers'] for r in weights if r['dataset']==dataset and r['outer_seed']==seed and r['candidate']==name) for name in NAMES]
            axes[0,column].plot(x,scores,'o-',color=color,alpha=.8,label=f'Seed {seed}')
            axes[1,column].plot(x,effective,'o-',color=color,alpha=.8)
        selected=NAMES.index(selections[dataset]['selected_candidate'])
        axes[0,column].axvspan(selected-.22,selected+.22,color='#e7bc4b',alpha=.22)
        axes[0,column].set_title(('UCI-HAR' if dataset=='uci' else 'WISDM')+' — development selection')
        axes[0,column].set_ylabel('Selection macro-F1 (%)');axes[1,column].set_ylabel('Effective teacher count')
        axes[1,column].set_ylim(.8,5.2)
        for row in range(2):
            axes[row,column].set_xticks(x,labels,rotation=25,ha='right');axes[row,column].grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Training-only weighting pilot: three seeds, one subject partition per dataset\nShading marks the selected mean; no original validation/test evaluation',fontsize=12)
    fig.savefig(output/'weighting-pilot.png',dpi=180);fig.savefig(output/'weighting-pilot.pdf')
    plt.close(fig)


if __name__=='__main__':main()
