"""Analyze only completed, hash-verified second-review noise experiments."""
import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from scipy.stats import t


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def digest(payload):
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--drive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--wisdm-results',type=Path,help='Explicit local folder override; preserve recorded canonical paths')
    args=parser.parse_args()
    args.drive=args.drive.resolve()
    rows=[];individual=[];pairs=[];evidence=set();verified=[];locations={}
    package=args.drive/'revision_2'
    frozen_path=package/'frozen-inputs.json'
    frozen=json.loads(frozen_path.read_text())
    evidence.add(frozen_path)
    for rel,expected_hash in frozen['runtime_files'].items():
        path=(package/rel).resolve()
        assert package in path.parents
        assert sha(path)==expected_hash,f'Frozen runtime changed: {path}'
        evidence.add(path)
        verified.append(dict(role='frozen_runtime',path=str(path),sha256=expected_hash))
    for dataset in ('uci','wisdm'):
        for rel,expected_hash in frozen['datasets'][dataset]['source_files'].items():
            path=(args.drive/rel).resolve()
            assert args.drive in path.parents
            assert sha(path)==expected_hash,f'Frozen source changed: {path}'
            verified.append(dict(role='frozen_source',path=str(path),sha256=expected_hash))
            if path.suffix=='.json':evidence.add(path)
    for dataset in ('uci','wisdm'):
        canonical=args.drive/'results/revision-2/coalition-noise'/dataset
        root=(args.wisdm_results if dataset=='wisdm' and args.wisdm_results else canonical).resolve()
        assert args.drive in root.parents
        locations[dataset]=dict(recorded_folder=str(canonical),actual_local_folder=str(root))
        marker=root/'completed.json'
        if not marker.exists():
            raise SystemExit(f'{dataset}: research noise experiment is not complete; no analysis written.')
        completed=json.loads(marker.read_text())
        assert completed['status']=='complete' and completed['fits']==30 and completed['test_evaluated'] is False
        expected={str((canonical/f'seed-{seed}'/f'repetition-{rep:02d}-{kind}.json').relative_to(args.drive)).replace('\\','/')
                  for seed in (11,23,37) for rep in range(1,6) for kind in ('empty','full')}
        actual={item['path'].replace('\\','/') for item in completed['files']}
        assert expected==actual and len(completed['files'])==30
        protocol=json.loads((root/'experiment-protocol.json').read_text())
        protocol_digest=hashlib.sha256(json.dumps(protocol,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        assert completed['protocol_sha256']==protocol_digest
        assert protocol['version']=='revision2-coalition-noise-v1'
        assert protocol['mode']=='research' and protocol['outer_seeds']==[11,23,37] and protocol['repetitions']==5
        assert protocol['dataset']==dataset and protocol['test_evaluated'] is False
        assert protocol['epochs']==(20 if dataset=='uci' else 60)
        assert protocol['frozen_inputs_sha256']==sha(frozen_path)
        evidence.update([marker,root/'experiment-protocol.json'])
        if dataset=='wisdm':
            startup_path=root/'deterministic-startup.json'
            startup=json.loads(startup_path.read_text())
            assert startup['deterministic_operations'] is True and startup['tensor_float_32_execution'] is False
            assert startup['original_runner_sha256']==sha(package/'run_coalition_noise.py')
            assert startup['startup_launcher_sha256']==sha(package/'run_wisdm_noise_deterministic.py')
            assert startup['frozen_inputs_sha256']==sha(frozen_path)
            evidence.update([startup_path,package/'run_wisdm_noise_deterministic.py'])
        for item in completed['files']:
            relative=Path(item['path'].replace('\\','/')).relative_to(canonical.relative_to(args.drive))
            path=(root/relative).resolve()
            assert root in path.parents
            assert sha(path)==item['sha256'],f'Changed result: {path}'
            evidence.add(path)
        for seed in (11,23,37):
            results={kind:[] for kind in ('empty','full')}
            source_path=args.drive/frozen['datasets'][dataset]['seed_records'][str(seed)]
            source=json.loads(source_path.read_text())
            assert source['config']==protocol['config']
            raw=np.asarray(source['shapley']['values'])*100
            preflight_path=root/f'seed-{seed}'/'preflight.json'
            bank_path=root/f'seed-{seed}'/'teacher-probabilities.json'
            preflight=json.loads(preflight_path.read_text())
            bank=json.loads(bank_path.read_text())
            assert preflight['passed'] is True and preflight['test_evaluated'] is False
            assert preflight['teacher_validation_parity']==bank['teacher_validation_parity']
            probability_path=root/f'seed-{seed}'/'teacher-probabilities.npz'
            assert sha(probability_path)==bank['sha256']
            verified.append(dict(role='teacher_probabilities',path=str(probability_path),sha256=bank['sha256']))
            evidence.update([source_path,preflight_path,bank_path])
            for name in source['teacher_order']:
                parity=preflight['teacher_validation_parity'][name]
                assert parity['saved']==source['teachers'][name]['validation']
                assert parity['tolerance']==1e-4
                for metric in ('accuracy','macro_f1'):
                    error=abs(parity['reproduced'][metric]-parity['saved'][metric])
                    assert error==parity['absolute_metric_errors'][metric] and error<=1e-4
            for rep in range(1,6):
                matched=[]
                for kind in ('empty','full'):
                    x=json.loads((root/f'seed-{seed}'/f'repetition-{rep:02d}-{kind}.json').read_text())
                    assert x['test_evaluated'] is False and x['status']=='complete'
                    assert x['repetition']==rep and x['coalition']==kind and x['outer_seed']==seed
                    assert 0<=x['validation_macro_f1']<=1
                    assert x['training_seed']==seed*1000000+900000+rep
                    env_path=root/f"environment-{x['environment_sha256'][:16]}.json"
                    env=json.loads(env_path.read_text())
                    assert digest(env)==x['environment_sha256']
                    evidence.add(env_path)
                    if dataset=='wisdm':
                        assert startup['tensorflow']==env['tensorflow'] and startup['keras']==env['keras']
                    context=dict(protocol=protocol,outer_seed=seed,source_record_sha256=sha(source_path),
                                 source_files=frozen['datasets'][dataset]['source_files'],
                                 runtime_files=frozen['runtime_files'],environment=env)
                    assert preflight['identity']==bank['identity']==digest(context)
                    task=dict(context=context,repetition=rep,coalition=kind,training_seed=x['training_seed'])
                    assert x['identity']==digest(task)
                    history=x['history']
                    assert len(history['train_loss'])==len(history['val_macro_f1'])==x['epochs_completed']
                    assert 1<=x['epochs_completed']<=protocol['epochs']
                    assert np.isfinite(history['train_loss']).all() and np.isfinite(history['val_macro_f1']).all()
                    assert x['validation_macro_f1']==x['validation_metrics']['macro_f1']
                    assert abs(x['validation_macro_f1']-max(history['val_macro_f1']))<1e-4
                    matched.append(x['training_seed'])
                    results[kind].append(x['validation_macro_f1']*100)
                    np.testing.assert_allclose(raw,np.asarray(x['source_raw_shapley_values'])*100,rtol=0,atol=0)
                    individual.append(dict(dataset=dataset,outer_seed=seed,repetition=rep,coalition=kind,
                                           training_seed=x['training_seed'],validation_macro_f1_pct=x['validation_macro_f1']*100,
                                           epochs_completed=x['epochs_completed'],elapsed_seconds=x['elapsed_seconds'],
                                           environment_sha256=x['environment_sha256']))
                assert matched[0]==matched[1]
                pairs.append(dict(dataset=dataset,outer_seed=seed,repetition=rep,
                                  full_minus_empty_pp=results['full'][-1]-results['empty'][-1]))
            empty,full=np.asarray(results['empty']),np.asarray(results['full'])
            differences=full-empty
            half=t.ppf(.975,4)*differences.std(ddof=1)/np.sqrt(5)
            rows.append(dict(dataset=dataset,outer_seed=seed,repetitions=5,
                             empty_mean_f1_pct=float(empty.mean()),empty_sd_pp=float(empty.std(ddof=1)),
                             empty_min_f1_pct=float(empty.min()),empty_max_f1_pct=float(empty.max()),
                             full_mean_f1_pct=float(full.mean()),full_sd_pp=float(full.std(ddof=1)),
                             full_min_f1_pct=float(full.min()),full_max_f1_pct=float(full.max()),
                             paired_full_minus_empty_mean_pp=float(differences.mean()),
                             paired_difference_sd_pp=float(differences.std(ddof=1)),
                             paired_mean_ci95_low_pp=float(differences.mean()-half),
                             paired_mean_ci95_high_pp=float(differences.mean()+half),
                             original_mean_absolute_raw_phi_pp=float(np.abs(raw).mean()),
                             original_max_absolute_raw_phi_pp=float(np.abs(raw).max())))
    args.output.mkdir(parents=True,exist_ok=True)
    for name,values in [('individual-fits',individual),('paired-differences',pairs)]:
        with (args.output/f'coalition-noise-{name}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    snapshot_root=args.output/'evidence_snapshot/coalition-noise'
    for path in sorted(evidence):
        destination=snapshot_root/path.relative_to(args.drive)
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,destination)
        verified.append(dict(role='snapshot',path=str(path),snapshot=str(destination.relative_to(args.output)),sha256=sha(path)))
    audit=dict(analysis_utc=datetime.now(timezone.utc).isoformat(),fits_verified=60,teacher_parity_checks=30,
               test_evaluated=False,locations=locations,files=verified,analysis_script_sha256=sha(Path(__file__)))
    with (args.output/'coalition-noise-summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    lines=['# Fixed teacher bank coalition noise','',
           'Five independent paired empty/full student runs per teacher bank. All values below are in macro-F1 percentage points. '
           'The empty/full diagnostic estimates optimization variability for these coalitions only; it does not estimate '
           'confidence intervals for all Shapley contributions. Paired t intervals are descriptive and unadjusted.','',
           '| Dataset | Bank seed | Empty SD | Full SD | Full minus empty mean | Original mean absolute Shapley |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['outer_seed']} | {r['empty_sd_pp']:.3f} | {r['full_sd_pp']:.3f} | "
                     f"{r['paired_full_minus_empty_mean_pp']:+.3f} | {r['original_mean_absolute_raw_phi_pp']:.3f} |")
    (args.output/'coalition-noise-report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    enrich_report(args.output,rows,individual,pairs,locations)
    audit['analysis_outputs_sha256']={path.name:sha(path) for path in args.output.glob('coalition-noise-*')
                                     if path.suffix in ('.csv','.md')}
    audit['analysis_outputs_sha256'].update({name:sha(args.output/name) for name in ('coalition-noise.png','coalition-noise.pdf')})
    (args.output/'coalition-noise-verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
    print('Verified 60 fits and 30 teacher parity checks; report and evidence snapshot written.')


def enrich_report(output,rows,individual,pairs,locations):
    report=output/'coalition-noise-report.md'
    lines=[report.read_text(encoding='utf-8'),'Analysis date: 18 October 2026. All 60 research fits passed completion, '
           'SHA-256, protocol, environment, source, context and fit identity checks. All thirty saved teachers '
           'passed the original 0.0001 validation accuracy and macro-F1 tolerance. Source data, six production '
           'records, thirty teacher checkpoints and frozen code hashes also agree. No test predictions were evaluated.',
           '', '## Utility means and observed ranges', '',
           'Means and ranges are macro-F1 percentages. Sample SDs and differences are percentage points (pp). '
           'Empty is supervised-only; full is uniform five-teacher distillation. UCI-HAR uses at most twenty '
           'epochs and WISDM sixty, with patience ten and best validation macro-F1 checkpoint selection.', '',
           '| Dataset | Bank | Empty mean [min, max] | Full mean [min, max] |',
           '|---|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset'].upper()} | {r['outer_seed']} | {r['empty_mean_f1_pct']:.3f} "
                     f"[{r['empty_min_f1_pct']:.3f}, {r['empty_max_f1_pct']:.3f}] | {r['full_mean_f1_pct']:.3f} "
                     f"[{r['full_min_f1_pct']:.3f}, {r['full_max_f1_pct']:.3f}] |")
    lines+=['','## Paired differences and descriptive uncertainty','',
            'The five empty/full pairs match student initialization and training seed. Intervals use t(4), '
            'are unadjusted and describe only repeated student training conditional on each bank and split. '
            'They do not describe variation over subjects or Shapley uncertainty.','',
            '| Dataset | Bank | Full minus empty mean (pp) | Difference SD (pp) | Descriptive 95% interval (pp) |',
            '|---|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['dataset'].upper()} | {r['outer_seed']} | {r['paired_full_minus_empty_mean_pp']:+.3f} | "
                     f"{r['paired_difference_sd_pp']:.3f} | [{r['paired_mean_ci95_low_pp']:+.3f}, {r['paired_mean_ci95_high_pp']:+.3f}] |")
    lines+=['','## Interpretation for Reviewer 1 comment 3','']
    for dataset in ('uci','wisdm'):
        subset=[r for r in rows if r['dataset']==dataset]
        sds=[r[f'{kind}_sd_pp'] for r in subset for kind in ('empty','full')]
        phi=[r['original_mean_absolute_raw_phi_pp'] for r in subset]
        lines.append(f"For {dataset.upper()}, within-bank empty/full utility SD ranges from {min(sds):.3f} to "
                     f"{max(sds):.3f} pp. The corresponding bank-level mean absolute original raw Shapley "
                     f"values range from {min(phi):.3f} to {max(phi):.3f} pp. This compares scales; it is not "
                     'a statistical signal-to-noise test for individual teacher contributions.')
        lines.append('')
    lines+=['Exact enumeration removes coalition-sampling approximation; it does not remove training variability. '
            'These repetitions quantify optimization variability for empty/full utilities conditional on the '
            'fixed teacher bank, data split, checkpoint rule and numerical execution settings. The thirty '
            'intermediate coalitions were not repeated. Consequently these results cannot establish confidence '
            'intervals, sign reliability or rank stability for the full Shapley vector. A negative original '
            'value alone does not establish that a teacher is harmful.','',
            'Do not treat the fifteen repetitions per dataset as fifteen independent teacher banks or subject '
            'cohorts. The earlier ten-seed rank analysis remains separate. This diagnostic does not establish '
            'test-set improvement or select a new teacher-weighting rule.','',
            '## Runtime and recovery provenance','',
            'Both datasets record TensorFlow 2.21.0, Keras 3.13.2, Python 3.13.15, NumPy 2.1.3, '
            'scikit-learn 1.6.1 and a GPU. The generic environment record does not identify the GPU model; '
            'the recovery session UI previously reported an L4. UCI-HAR uses the original startup. WISDM '
            'uses deterministic startup and disabled TensorFloat-32 before teacher inference and student '
            'fitting, as recorded in deterministic-startup.json. All fifteen WISDM saved teachers reproduced '
            'validation accuracy and macro-F1 exactly. No tolerance was relaxed. Execution precision differs '
            'between datasets; the difference in noise magnitudes cannot be attributed to the datasets alone.','',
            f"The completed WISDM folder is locally named `{locations['wisdm']['actual_local_folder']}`. "
            'The analysis explicitly maps its canonical `coalition-noise/wisdm/` completion paths to that '
            '`wisdm (1)` folder and verifies every original result hash. Neither folder names nor completion '
            'records were changed.','',
            'Selected JSON records and small code files are preserved under '
            '`evidence_snapshot/coalition-noise/`. Large datasets, teacher checkpoints and probability arrays '
            'remain on Google Drive. `coalition-noise-verification.json` records checked paths and hashes. '
            'The per-fit CSV records training seeds, epochs and elapsed time.','',
            '## Next planned experiment','',
            'Next is the predeclared UCI-HAR 20-versus-60 epoch coalition-budget sensitivity study '
            '(Reviewer 1 comment 5), followed by teacher removal (comment 2). Preserve planned seed counts '
            'and null outcomes. Full repeated-game Shapley intervals would require a separate frozen protocol '
            'if strong attribution reliability claims remain; the current revision direction is to qualify '
            'those claims explicitly.','',
            '## Draft manuscript text for integration','',
            'We separated between-bank variation from student-optimization variation by fixing three saved '
            'teacher banks per dataset and training the empty (supervised-only) and full (uniform five-teacher '
            'distillation) coalitions with five independent paired student seeds. We retained the original '
            'coalition budgets, checkpoint criterion and data split, and evaluated validation utility only. '
            'Within-bank variability is reported alongside the original raw Shapley magnitudes. Exact '
            'enumeration therefore gives exact attribution of the realized training-dependent utility table; '
            'it does not guarantee stable teacher importance. Because only the empty and full coalitions '
            'were repeated, this diagnostic does not provide confidence intervals for individual Shapley '
            'values or teacher ranks.','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(11,7.4),constrained_layout=True)
    colors=['#276493','#cd7c2f','#64884f']
    for column,dataset in enumerate(('uci','wisdm')):
        subset=[r for r in rows if r['dataset']==dataset];x=np.arange(3)
        for offset,key,label,color in zip((-.23,0,.23),
                ('empty_sd_pp','full_sd_pp','original_mean_absolute_raw_phi_pp'),
                ('Empty utility SD','Full utility SD','Mean |original raw Shapley|'),colors):
            axes[0,column].bar(x+offset,[r[key] for r in subset],width=.22,label=label,color=color)
        axes[0,column].set_title('UCI-HAR (20-epoch limit)' if dataset=='uci' else 'WISDM (60-epoch limit)')
        axes[0,column].set_ylabel('Macro-F1 percentage points')
        axes[0,column].set_xticks(x,['11','23','37']);axes[0,column].set_xlabel('Frozen teacher-bank seed')
        axes[0,column].grid(axis='y',alpha=.2);axes[0,column].set_axisbelow(True)
        axes[0,column].set_ylim(0,max(r[key] for r in subset for key in
            ('empty_sd_pp','full_sd_pp','original_mean_absolute_raw_phi_pp'))*1.5)
        for i,r in enumerate(subset):
            values=[p['full_minus_empty_pp'] for p in pairs if p['dataset']==dataset and p['outer_seed']==r['outer_seed']]
            axes[1,column].scatter(i+np.linspace(-.12,.12,5),values,color=colors[0],s=26,alpha=.8)
            mean=r['paired_full_minus_empty_mean_pp']
            axes[1,column].errorbar(i,mean,yerr=[[mean-r['paired_mean_ci95_low_pp']],
                [r['paired_mean_ci95_high_pp']-mean]],fmt='D',color='#222222',capsize=5,markersize=5)
        axes[1,column].axhline(0,color='#555555',linewidth=.8,linestyle='--')
        axes[1,column].set_xticks(x,['11','23','37']);axes[1,column].set_xlabel('Frozen teacher-bank seed')
        axes[1,column].set_ylabel('Full − empty macro-F1 (pp)')
        axes[1,column].set_title('Paired differences; mean and descriptive 95% CI',fontsize=10)
        axes[1,column].grid(axis='y',alpha=.2)
    axes[0,0].legend(fontsize=8,loc='upper left')
    fig.suptitle('Fixed-bank coalition training variability\nScale comparison does not estimate individual Shapley uncertainty',fontsize=13)
    fig.savefig(output/'coalition-noise.png',dpi=180);fig.savefig(output/'coalition-noise.pdf')
    plt.close(fig)


if __name__=='__main__':main()
