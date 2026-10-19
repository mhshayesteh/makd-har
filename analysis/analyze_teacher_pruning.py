"""Verify complete teacher-removal records and analyze the prespecified paired comparisons."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
import sys
from datetime import datetime,timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy import stats


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def paired(values):
    d=np.asarray(values,dtype=float);nonzero=d[d!=0]
    assert len(d)==10 and np.isfinite(d).all()
    if len(nonzero):
        ranks=stats.rankdata(np.abs(nonzero),method='average')
        wplus=float(ranks[nonzero>0].sum());total=float(ranks.sum())
        observed=min(wplus,total-wplus)
        null=np.array([np.dot(bits,ranks) for bits in itertools.product((0,1),repeat=len(nonzero))])
        p=float(np.mean(np.minimum(null,total-null)<=observed+1e-12))
        rank_biserial=(2*wplus-total)/total
    else:observed=0.;p=1.;rank_biserial=0.
    sd=float(d.std(ddof=1));half=float(stats.t.ppf(.975,9)*sd/math.sqrt(10))
    return dict(n=10,mean_difference_pp=float(d.mean()*100),median_difference_pp=float(np.median(d)*100),
        sd_difference_pp=sd*100,ci95_low_pp=float((d.mean()-half)*100),ci95_high_pp=float((d.mean()+half)*100),
        wilcoxon_statistic=observed,p_raw=p,rank_biserial=rank_biserial,
        wins=int((d>0).sum()),ties=int((d==0).sum()),losses=int((d<0).sum()))


def holm(values):
    values=np.asarray(values);order=np.argsort(values);result=np.empty(len(values))
    result[order]=np.minimum(1,np.maximum.accumulate(values[order]*np.arange(len(values),0,-1)))
    return result


def csv_file(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def metrics_from_labels(truth,predicted):
    """Independent six-class confusion-matrix reconstruction with zero-division set to zero."""
    matrix=np.bincount(np.asarray(truth,dtype=int)*6+np.asarray(predicted,dtype=int),minlength=36).reshape(6,6)
    tp=np.diag(matrix).astype(float);support=matrix.sum(axis=1);predicted_count=matrix.sum(axis=0)
    precision=np.divide(tp,predicted_count,out=np.zeros(6),where=predicted_count>0)
    recall=np.divide(tp,support,out=np.zeros(6),where=support>0)
    denom=support+predicted_count
    f1=np.divide(2*tp,denom,out=np.zeros(6),where=denom>0)
    return dict(accuracy=float(tp.sum()/matrix.sum()),macro_f1=float(f1.mean()),
        macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),per_class_f1=f1.tolist())


def verify_predictions(path,record,source,expected_labels):
    with np.load(path,allow_pickle=False) as archive:
        assert set(archive.files)=={'y_val','val_predictions','y_test','test_predictions'}
        for short,long in [('val','validation'),('test','test')]:
            truth=archive[f'y_{short}'];predicted=archive[f'{short}_predictions']
            assert truth.ndim==predicted.ndim==1 and len(truth)==len(predicted)==source['dataset_manifest'][long]['samples']
            assert np.issubdtype(truth.dtype,np.integer) and np.issubdtype(predicted.dtype,np.integer)
            assert np.isin(truth,range(6)).all() and np.isin(predicted,range(6)).all()
            counts={str(i):int((truth==i).sum()) for i in range(6)}
            assert counts==source['dataset_manifest'][long]['class_counts']
            if short not in expected_labels:expected_labels[short]=truth.copy()
            np.testing.assert_array_equal(truth,expected_labels[short])
            metrics=metrics_from_labels(truth,predicted)
            saved=record[f'{long}_metrics']
            for key in metrics:np.testing.assert_allclose(metrics[key],saved[key],rtol=0,atol=1e-12)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--drive',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();drive=args.drive.resolve();root=drive/'results/revision-2/teacher-pruning'
    if not (root/'completed.json').is_file():raise SystemExit('Teacher-removal research incomplete; no analysis written.')
    sys.path.insert(0,str(drive/'revision_2/teacher_pruning'))
    from run_teacher_pruning import verify_inputs,protocol,digest,checked_fit,retained_indices,paired_summary,SEEDS,validate_parity
    frozen=verify_inputs(drive);specification=protocol(frozen)
    assert json.loads((root/'experiment-protocol.json').read_text())==specification
    env=json.loads((root/'environment.json').read_text());assert env['deterministic_operations'] is True and env['tensor_float_32_execution'] is False
    completed=json.loads((root/'completed.json').read_text())
    assert completed['status']=='complete' and completed['mode']=='research' and completed['fits']==120
    assert completed['specification_sha256']==digest(specification) and completed['environment_sha256']==digest(env)
    gate=json.loads((root/'preflight-completed.json').read_text())
    assert gate['passed'] and gate['teacher_checks']==100 and gate['banks']==20 and len(gate['files'])==60
    assert gate['environment_sha256']==digest(env) and gate['specification_sha256']==digest(specification)
    snapshots={root/'completed.json',root/'preflight-completed.json',root/'environment.json',root/'experiment-protocol.json',
        drive/'revision_2/teacher_pruning/frozen-inputs.json',drive/'revision_2/teacher_pruning/run_teacher_pruning.py',
        drive/'revision_2/teacher_pruning/test_pruning_guards.py'}
    all_hashes=[]
    def verify_list(items):
        for item in items:
            path=(drive/item['path']).resolve();assert root in path.parents
            assert sha(path)==item['sha256'],f'Changed research artifact: {path}'
            all_hashes.append(dict(path=str(path),sha256=item['sha256']))
            if path.suffix=='.json':snapshots.add(path)
    verify_list(gate['files'])
    assert len(completed['datasets'])==2
    assert {item['path'].replace('\\','/') for item in completed['datasets']}=={
        f'results/revision-2/teacher-pruning/{dataset}/completed.json' for dataset in ('uci','wisdm')}
    verify_list(completed['datasets'])
    conditions=[];summaries=[]
    for dataset in ('uci','wisdm'):
        dsroot=root/dataset;marker=json.loads((dsroot/'completed.json').read_text())
        run_protocol=json.loads((dsroot/'run-protocol.json').read_text())
        assert run_protocol==dict(specification=specification,mode='research',dataset=dataset,outer_seeds=SEEDS,
                                  epochs=60,conditions=specification['conditions'])
        assert marker['status']=='complete' and marker['mode']=='research' and marker['fits']==60
        assert marker['dataset']==dataset and marker['test_evaluated'] is True
        assert marker['run_protocol_sha256']==digest(run_protocol) and marker['environment_sha256']==digest(env)
        expected=set()
        for seed in SEEDS:
            for name in specification['conditions']:
                for extension in ('json','keras','npz'):
                    expected.add(f'results/revision-2/teacher-pruning/{dataset}/seed-{seed}/{name}.{extension}')
            expected.add(f'results/revision-2/teacher-pruning/{dataset}/seed-{seed}/removal-comparison.json')
        actual=[item['path'].replace('\\','/') for item in marker['files']]
        assert len(actual)==190 and set(actual)==expected
        verify_list(marker['files']);snapshots.add(dsroot/'run-protocol.json')
        labels={}
        for seed in SEEDS:
            seed_dir=dsroot/f'seed-{seed}';source_path=drive/frozen['seed_records'][dataset][str(seed)]
            source=json.loads(source_path.read_text());snapshots.add(source_path)
            choice=frozen['choices'][dataset][str(seed)];raw=source['shapley']['values']
            assert choice['raw_shapley_values']==raw and choice['teacher_order']==source['teacher_order']
            assert choice['removed_index']==min(range(5),key=lambda i:(raw[i],i))
            preflight=json.loads((seed_dir/'preflight.json').read_text());context=preflight['context']
            assert preflight['passed'] and preflight['context_sha256']==digest(context)
            assert preflight['dataset_manifest']==source['dataset_manifest'] and preflight['test_evaluated'] is False
            assert context==dict(specification=specification,environment=env,dataset=dataset,outer_seed=seed,
                source_record_sha256=sha(source_path),config=source['config'],teacher_order=source['teacher_order'],frozen_choice=choice)
            bank=json.loads((seed_dir/'teacher-probabilities.json').read_text())
            assert bank['identity']==digest(context) and bank['sha256']==sha(seed_dir/'teacher-probabilities.npz')
            assert bank['teacher_validation_parity']==preflight['teacher_validation_parity']
            validate_parity(bank['teacher_validation_parity'],source)
            reports={}
            for condition in specification['conditions']:
                removed=None if condition=='full' else int(condition.split('-')[1]);retained=retained_indices(removed)
                task=dict(context=context,mode='research',condition=condition,removed_index=removed,
                    retained_indices=retained,uniform_weights=[1/len(retained)]*len(retained),training_seed=seed*1000000+7,epochs=60)
                path=seed_dir/f'{condition}.json';report=checked_fit(path,task);assert report is not None
                assert report['model_file']==f'{condition}.keras' and report['predictions_file']==f'{condition}.npz'
                verify_predictions(seed_dir/report['predictions_file'],report,source,labels)
                reports[condition]=report
                conditions.append(dict(dataset=dataset,outer_seed=seed,condition=condition,
                    removed_teacher='' if removed is None else source['teacher_order'][removed],
                    is_frozen_choice=removed==choice['removed_index'],training_seed=task['training_seed'],
                    validation_macro_f1_pct=report['validation_metrics']['macro_f1']*100,
                    test_macro_f1_pct=report['test_metrics']['macro_f1']*100,
                    test_accuracy_pct=report['test_metrics']['accuracy']*100,
                    epochs_completed=report['epochs_completed'],best_epoch=report['best_epoch'],elapsed_seconds=report['elapsed_seconds']))
            comparison=json.loads((seed_dir/'removal-comparison.json').read_text())
            computed=paired_summary(reports,choice['removed_index'])
            assert comparison==dict(context=context,run_protocol=run_protocol,removed_index=choice['removed_index'],
                removed_teacher=choice['removed_teacher'],raw_shapley_values=raw,comparisons=computed)
            row=dict(dataset=dataset,outer_seed=seed,removed_teacher=choice['removed_teacher'],
                selected_original_raw_phi_pp=raw[choice['removed_index']]*100)
            for partition,values in computed.items():
                for key,value in values.items():
                    if key!='all_five_removals':row[f'{partition}_{key}_pp' if 'minus' in key else f'{partition}_{key}_pct']=value*100
            summaries.append(row)
    tests=[]
    for dataset in ('uci','wisdm'):
        rows=[r for r in summaries if r['dataset']==dataset]
        for name in ('selected_minus_random','selected_minus_full','random_minus_full'):
            tests.append(dict(dataset=dataset,comparison=name,**paired([r[f'test_{name}_pp']/100 for r in rows])))
    for row,adjusted in zip(tests,holm([r['p_raw'] for r in tests])):row['p_holm_six_comparisons']=float(adjusted)
    assert len(conditions)==120 and len(summaries)==20 and len(tests)==6
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('conditions',conditions),('paired-seeds',summaries),('statistics',tests)]:
        csv_file(output/f'teacher-removal-{name}.csv',rows)
    snapshot=output/'evidence_snapshot/teacher-removal';snapshot_items=[]
    for path in sorted(snapshots):
        destination=snapshot/path.relative_to(drive);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,destination)
        snapshot_items.append(dict(path=str(path),sha256=sha(path),snapshot=str(destination.relative_to(output))))
    report=['# Teacher-removal study','','All 120 research fits, saved models, prediction hashes, metric reconstructions, '
        'source/split/teacher contexts and checkpoint histories passed verification. Original production results are preserved.',
        '','| Dataset | Comparison | Mean difference (pp) | Descriptive 95% interval (pp) | Wins/ties/losses | Holm p (six tests) |',
        '|---|---|---:|---:|---:|---:|']
    for r in tests:
        report.append(f"| {r['dataset'].upper()} | {r['comparison']} | {r['mean_difference_pp']:+.3f} | "
            f"[{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}] | {r['wins']}/{r['ties']}/{r['losses']} | {r['p_holm_six_comparisons']:.6f} |")
    report+=['','The removal choice was fixed from original raw validation Shapley values with earliest-index exact tie handling. '
        'All retained teachers have uniform weights. Expected random removal is the mean of all five separate removal outcomes, '
        'including the selected removal. Full is a matched current-runtime five-teacher student.','',
        'All six predeclared conditions are reported. Test labels and scores did not choose removal, checkpoints or seeds. '
        'These are ten seed-paired comparisons on a fixed subject split; the five removals are not independent repetitions. '
        'Exact signed-rank p-values enumerate all sign assignments of nonzero paired differences with midranks for ties; '
        'one Holm family covers six tests. Mean-difference t intervals are descriptive and unadjusted. The already-used '
        'test split provides post-review robustness evidence, not a newly untouched confirmation.','',
        'Neither selected-versus-random comparison is statistically significant after the predeclared correction. '
        'Attribution-guided pruning utility remains unproven. Preserve the result '
        'and narrow claims rather than selecting different seeds or a favorable removal after observing outcomes. '
        'The original UCI attribution uses a 20-epoch surrogate game and corrected WISDM a 60-epoch game. These findings '
        'do not estimate uncertainty for individual original Shapley values.','',
        f"Analysis date: {datetime.now(ZoneInfo('Asia/Tehran')).date().isoformat()} (Asia/Tehran). "
        'Manuscript/response integration remains pending.','']
    report+=['## Mean test outcomes','','Macro-F1 is reported as a percentage; differences above are percentage points. '
        'The random baseline is an expectation over five outcomes within each bank, followed by a mean across ten banks.',
        '','| Dataset | Lowest-Shapley removal | Expected random removal | Matched full five teachers |',
        '|---|---:|---:|---:|']
    for dataset in ('uci','wisdm'):
        subset=[r for r in summaries if r['dataset']==dataset]
        means=[np.mean([r[f'test_{name}_pct'] for r in subset]) for name in
               ('lowest_phi_removal','expected_random_removal','full_uniform')]
        report.append(f"| {dataset.upper()} | {means[0]:.3f} | {means[1]:.3f} | {means[2]:.3f} |")
    report+=['','## All removal conditions','','These are descriptive averages of the ten predeclared bank/seed outcomes. '
        'They do not select a replacement removal rule from test performance.',
        '','| Dataset | Removed teacher | Mean test macro-F1 (%) | Number selected by frozen rule |',
        '|---|---|---:|---:|']
    for dataset in ('uci','wisdm'):
        for index in range(5):
            subset=[r for r in conditions if r['dataset']==dataset and r['condition']==f'remove-{index}']
            report.append(f"| {dataset.upper()} | {subset[0]['removed_teacher']} | "
                f"{np.mean([r['test_macro_f1_pct'] for r in subset]):.3f} | "
                f"{sum(r['is_frozen_choice'] for r in subset)}/10 |")
    report+=['','## Interpretation and limits','',
        'The selected removal is below expected random removal by 0.149 pp on UCI-HAR and above it by 0.698 pp on WISDM. '
        'It wins in only 3/10 and 5/10 banks, respectively; both corrected p-values are 1.000. '
        'Selected removal differs from the matched full-teacher control by +0.041 and +0.245 pp, respectively. '
        'None of the six corrected comparisons is significant. This does not establish equivalence, absence of an effect, '
        'or that a teacher is harmful. The descriptive intervals include both positive and negative effects.','',
        'The frozen choice has a positive original raw Shapley value for seed 107 in both datasets. '
        'Thus "lowest" is not synonymous with "negative." Shapley averages marginal contributions over coalitions; '
        'it is not identical to the full-coalition removal effect. Here the attribution game also differs from the final '
        'four-teacher training condition. These are possible reasons for imperfect transfer, not effects isolated by this study.','',
        'The mean selected-minus-random validation differences are +0.217 pp (UCI-HAR) and +1.161 pp (WISDM), '
        'but those validation subjects also informed the original attribution choice and checkpoint selection. '
        'They are descriptive, reused-development outcomes and cannot establish generalization. '
        'Ten seeds characterize optimization/teacher-bank variability conditional on the fixed split; they do not '
        'supply ten independent subject samples. Individual Shapley-value uncertainty remains unmeasured.','',
        f"Summed saved fit elapsed time is {sum(r['elapsed_seconds'] for r in conditions)/3600:.3f} hours. "
        'This covers recorded training/validation work and excludes some setup/serialization; it is not billed GPU time.','',
        '## Proposed manuscript text','',
        'We assessed attribution-guided teacher removal using the ten original teacher banks for each dataset. '
        'Before training, we selected the teacher with the lowest original raw validation Shapley value, resolving exact '
        'ties by saved teacher order. For every bank, we trained all five leave-one-teacher-out students and a matched '
        'five-teacher control, with uniform retained-teacher weights and matched student training seeds and stopping rules. '
        'Expected uniform random removal was the mean of all five removal outcomes, including the selected one. '
        'Mean test macro-F1 was 91.018%, 91.167%, and 90.977% for selected removal, expected random removal, and full '
        'control on UCI-HAR; the corresponding WISDM values were 77.066%, 76.368%, and 76.820%. '
        'Selected-minus-random differences were -0.149 pp (descriptive 95% interval [-0.535, 0.236]) and +0.698 pp '
        '([-0.547, 1.944]). Neither comparison was significant under exact paired signed-rank testing with Holm '
        'correction across the six dataset/comparison tests (both adjusted p=1.000). These fixed-split, post-review '
        'robustness results do not establish a generalization benefit from attribution-guided pruning.','',
        '## Proposed response content','',
        'We added a controlled teacher-removal experiment covering all ten seeds on both datasets (120 student fits). '
        'The removal decision was frozen from original raw validation Shapley values. All five possible removals were '
        'trained so that random removal is represented by its exact expected outcome, and matched full-teacher controls '
        'were trained in the same runtime. We report every condition, paired differences, descriptive intervals and '
        'six-comparison Holm-adjusted tests. The results do not demonstrate a significant advantage over random removal. '
        'The final manuscript must therefore qualify pruning claims and explain the distinction between conditional '
        'coalition attribution and downstream teacher-removal utility. Section/table/page references remain to be added '
        'after manuscript integration and pagination.','']
    (output/'teacher-removal-report.md').write_text('\n'.join(report),encoding='utf-8')
    plot_results(output,summaries,tests)
    audit=dict(analysis_utc=datetime.now(timezone.utc).isoformat(),fits_verified=120,teacher_checks=100,
        models_verified=120,prediction_files_verified=120,prediction_metric_reconstructions=240,
        frozen_input_count=len(frozen['files']),analysis_script_sha256=sha(Path(__file__)),
        checked_hashes=all_hashes,snapshots=snapshot_items)
    audit['analysis_output_sha256']={p.name:sha(p) for p in output.glob('teacher-removal-*') if p.suffix in ('.csv','.md')}
    audit['analysis_output_sha256'].update({name:sha(output/name) for name in ('teacher-removal.png','teacher-removal.pdf')})
    (output/'teacher-removal-verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(tests,indent=2))


def plot_results(output,rows,tests):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype':42,'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    for ax,dataset in zip(axes,('uci','wisdm')):
        subset=[r for r in rows if r['dataset']==dataset];x=np.arange(10)
        values=[r['test_selected_minus_random_pp'] for r in subset]
        ax.scatter(x,values,color='#276493',s=42)
        ax.axhline(0,color='#777777',linestyle='--',linewidth=.8)
        average=next(r for r in tests if r['dataset']==dataset and r['comparison']=='selected_minus_random')
        ax.axhline(average['mean_difference_pp'],color='#c67525',linewidth=1,label='Mean paired difference')
        ax.set_xticks(x,[str(r['outer_seed']) for r in subset],rotation=45)
        ax.set_xlabel('Original teacher-bank seed');ax.set_ylabel('Selected − expected random removal\ntest macro-F1 (percentage points)')
        ax.set_title(dataset.upper());ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=9)
    fig.suptitle('Teacher removal: frozen lowest-Shapley choice versus uniform random removal')
    fig.savefig(output/'teacher-removal.png',dpi=180);fig.savefig(output/'teacher-removal.pdf')
    plt.close(fig)


if __name__=='__main__':main()
