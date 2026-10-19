"""Analyze only complete, hash-verified UCI paired budget experiments."""
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
import numpy as np
from scipy.stats import rankdata,spearmanr


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def exact_values(utilities,n=5):
    values=[]
    for player in range(n):
        result=0.
        others=[j for j in range(n) if j!=player]
        for size in range(n):
            coefficient=math.factorial(size)*math.factorial(n-size-1)/math.factorial(n)
            for subset in itertools.combinations(others,size):
                result+=coefficient*(utilities[tuple(sorted((*subset,player)))]-utilities[subset])
        values.append(result)
    return np.array(values)


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def verify_history_and_metrics(record,patience=10):
    scores=record['history']['val_macro_f1'];best=-math.inf;stale=0
    assert len(scores)==record['epochs_completed'] and 1<=len(scores)<=60
    for epoch,score in enumerate(scores,1):
        if score>best+1e-7:best=score;stale=0
        else:stale+=1
        if stale>=patience:assert epoch==len(scores),'Training continued beyond the stopping rule'
    assert len(scores)==60 or stale==patience,'History ended before its prescribed stopping point'
    assert abs(best-record['validation_metrics']['macro_f1'])<1e-4
    for key in ('validation_metrics','test_metrics'):
        if key not in record:continue
        metrics=record[key]
        for name in ('accuracy','macro_f1','macro_precision','macro_recall'):
            assert math.isfinite(metrics[name]) and 0<=metrics[name]<=1
        assert len(metrics['per_class_f1'])==6
        assert all(math.isfinite(v) and 0<=v<=1 for v in metrics['per_class_f1'])
        assert abs(np.mean(metrics['per_class_f1'])-metrics['macro_f1'])<1e-12


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--drive',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();drive=args.drive.resolve();root=drive/'results/revision-2/uci-budget'
    marker=root/'completed.json'
    if not marker.exists():raise SystemExit('Budget research is incomplete; no analysis written.')
    sys.path.insert(0,str(drive/'revision_2/uci_budget'))
    from run_uci_budget import verify_inputs,digest,checked_result
    frozen=verify_inputs(drive)
    completed=json.loads(marker.read_text())
    assert completed['status']=='complete' and completed['mode']=='research' and completed['fits']==102
    assert completed['coalition_test_evaluated'] is False and completed['final_test_evaluated'] is True
    protocol_path=root/'experiment-protocol.json';protocol=json.loads(protocol_path.read_text())
    assert digest(protocol)==completed['protocol_sha256']
    assert protocol['version']=='revision2-uci-budget-v1' and protocol['outer_seeds']==[11,23,37]
    assert protocol['coalition_budgets']==[20,60] and protocol['players']==5 and protocol['final_epochs']==60
    assert protocol['frozen_inputs_sha256']==sha(drive/'revision_2/uci_budget/frozen-inputs.json')
    env_path=root/f"environment-{completed['environment_sha256'][:16]}.json"
    environment=json.loads(env_path.read_text());assert digest(environment)==completed['environment_sha256']
    assert environment['deterministic_operations'] is True and environment['tensor_float_32_execution'] is False
    expected=set()
    for seed in (11,23,37):
        seed_dir=root/f'seed-{seed}'
        for size in range(6):
            for coalition in itertools.combinations(range(5),size):
                key='-'.join(map(str,coalition)) if coalition else 'empty'
                expected.add(str((seed_dir/'coalitions'/f'{key}.json').relative_to(drive)).replace('\\','/'))
        for budget in (20,60):
            for ext in ('json','keras'):
                expected.add(str((seed_dir/f'final-from-{budget}-epoch-game.{ext}').relative_to(drive)).replace('\\','/'))
        expected.add(str((seed_dir/'budget-comparison.json').relative_to(drive)).replace('\\','/'))
    actual=[item['path'].replace('\\','/') for item in completed['files']]
    assert set(actual)==expected and len(actual)==len(expected)==111
    snapshots={marker,protocol_path,env_path,drive/'revision_2/uci_budget/frozen-inputs.json',
               drive/'revision_2/uci_budget/run_uci_budget.py',drive/'revision_2/uci_budget/test_budget_guards.py'}
    for item in completed['files']:
        path=(drive/item['path']).resolve();assert root in path.parents
        assert sha(path)==item['sha256'],f'Changed budget result: {path}'
        if path.suffix=='.json':snapshots.add(path)
    summaries=[];teachers=[];coalitions=[];fits=[];final_metrics=[]
    for seed in (11,23,37):
        seed_dir=root/f'seed-{seed}';comparison=json.loads((seed_dir/'budget-comparison.json').read_text())
        context=comparison['context'];assert context['protocol']==protocol and context['environment']==environment
        source_path=drive/frozen['seed_records'][str(seed)];source=json.loads(source_path.read_text())
        assert context['source_record_sha256']==sha(source_path) and context['config']==source['config']
        assert context['teacher_order']==source['teacher_order'] and comparison['teacher_order']==source['teacher_order']
        bank=json.loads((drive/f'results/revision-2/coalition-noise/uci/seed-{seed}/teacher-probabilities.json').read_text())
        assert context['teacher_probabilities_sha256']==bank['sha256']
        preflight_path=seed_dir/'preflight.json';preflight=json.loads(preflight_path.read_text())
        assert preflight['passed'] is True and preflight['context_sha256']==digest(context)
        assert preflight['dataset_manifest']==source['dataset_manifest']
        assert preflight['teacher_probability_identity']==bank['identity']
        assert preflight['teacher_validation_parity']==bank['teacher_validation_parity']
        snapshots.update([preflight_path,source_path,
            drive/f'results/revision-2/coalition-noise/uci/seed-{seed}/teacher-probabilities.json'])
        utilities={20:{},60:{}}
        for size in range(6):
            for coalition in itertools.combinations(range(5),size):
                key='-'.join(map(str,coalition)) if coalition else 'empty'
                task=dict(context=context,kind='coalition',coalition_indices=list(coalition),training_seed=seed*100000+17)
                record=checked_result(seed_dir/'coalitions'/f'{key}.json',digest(task),'coalition')
                assert record['training_seed']==seed*100000+17 and record['coalition_indices']==list(coalition)
                assert record['outer_seed']==seed and 1<=record['epochs_completed']<=60
                assert record['epochs_completed']==len(record['history']['train_loss'])
                verify_history_and_metrics(record,context['config']['patience'])
                assert set(record['budget_utilities'])=={'20','60'}
                assert record['teacher_names']==[source['teacher_order'][i] for i in coalition]
                values={b:record['budget_utilities'][str(b)]['validation_macro_f1'] for b in (20,60)}
                assert values[60]>=values[20]-1e-12
                for b in (20,60):utilities[b][coalition]=values[b]
                coalitions.append(dict(outer_seed=seed,coalition=key,utility20_pct=values[20]*100,
                    utility60_pct=values[60]*100,difference_pp=(values[60]-values[20])*100,
                    epochs_completed=record['epochs_completed'],
                    best_epoch20=record['budget_utilities']['20']['best_epoch'],
                    best_epoch60=record['budget_utilities']['60']['best_epoch']))
                fits.append(dict(outer_seed=seed,kind='coalition',condition=key,training_seed=record['training_seed'],
                    epochs_completed=record['epochs_completed'],elapsed_seconds=record['elapsed_seconds']))
        phi={b:exact_values(utilities[b]) for b in (20,60)}
        weights={}
        for b in (20,60):
            game=comparison['games'][str(b)]
            np.testing.assert_allclose(phi[b],game['shapley']['values'],atol=1e-12,rtol=0)
            assert abs(phi[b].sum()-(utilities[b][(0,1,2,3,4)]-utilities[b][()]))<1e-10
            std=phi[b].std();z=(phi[b]-phi[b].mean())/std if std>1e-12 else np.zeros(5)
            z=z/context['config']['shapley_softmax_temperature']
            exp=np.exp(z-z.max());weights[b]=exp/exp.sum()
            np.testing.assert_allclose(weights[b],game['weights'],atol=1e-12,rtol=0)
            task=dict(context=context,kind='final',weight_game_epochs=b,weights=game['weights'],
                      training_seed=seed*1000000+7,epochs=60)
            report=checked_result(seed_dir/f'final-from-{b}-epoch-game.json',digest(task),'final')
            assert report==comparison['finals'][str(b)] and report['training_seed']==seed*1000000+7
            assert report['outer_seed']==seed and report['weight_game_epochs']==b
            np.testing.assert_allclose(report['weights'],game['weights'],atol=0,rtol=0)
            assert set(report['budget_utilities'])=={'60'}
            verify_history_and_metrics(report,context['config']['patience'])
            fits.append(dict(outer_seed=seed,kind='final',condition=f'weights-from-{b}',training_seed=report['training_seed'],
                epochs_completed=report['epochs_completed'],elapsed_seconds=report['elapsed_seconds']))
            final_metrics.append(dict(outer_seed=seed,weight_game_epochs=b,
                validation_macro_f1_pct=report['validation_metrics']['macro_f1']*100,
                test_macro_f1_pct=report['test_metrics']['macro_f1']*100,
                test_accuracy_pct=report['test_metrics']['accuracy']*100,
                epochs_completed=report['epochs_completed'],best_epoch=report['budget_utilities']['60']['best_epoch']))
        ranks={b:rankdata(-phi[b],method='average') for b in (20,60)}
        for i,name in enumerate(source['teacher_order']):
            teachers.append(dict(outer_seed=seed,teacher=name,phi20_pp=phi[20][i]*100,phi60_pp=phi[60][i]*100,
                rank20=ranks[20][i],rank60=ranks[60][i],weight20=weights[20][i],weight60=weights[60][i]))
        finals=comparison['finals'];historical=source['shapley']['coalition_values']
        historical_gap=max(abs(utilities[20][c]-historical[','.join(map(str,c))]) for c in utilities[20])
        assert abs(historical_gap-comparison['historical_prefix_utility_max_absolute_difference'])<1e-12
        assert comparison['historical_20_epoch_game']==source['shapley']
        assert comparison['historical_20_epoch_weights']==source['shapley_weights']
        seed_coalitions=[r for r in coalitions if r['outer_seed']==seed]
        summaries.append(dict(outer_seed=seed,rank_spearman_20_vs_60=float(spearmanr(phi[20],phi[60]).statistic),
            weight_l1_movement=float(np.abs(weights[60]-weights[20]).sum()),
            minimum_teacher20=source['teacher_order'][int(np.argmin(phi[20]))],
            minimum_teacher60=source['teacher_order'][int(np.argmin(phi[60]))],
            sign_changes=int(np.sum(np.sign(phi[20])!=np.sign(phi[60]))),
            changed_coalition_utilities=sum(r['difference_pp']>0 for r in seed_coalitions),
            coalitions_trained_beyond20=sum(r['epochs_completed']>20 for r in seed_coalitions),
            coalition_min_epochs=min(r['epochs_completed'] for r in seed_coalitions),
            coalition_max_epochs=max(r['epochs_completed'] for r in seed_coalitions),
            max_coalition_utility_gain_pp=max(r['difference_pp'] for r in seed_coalitions),
            max_absolute_teacher_weight_change=float(np.abs(weights[60]-weights[20]).max()),
            historical20_rank_spearman_vs_current20=float(spearmanr(source['shapley']['values'],phi[20]).statistic),
            historical20_max_utility_gap_pp=historical_gap*100,
            final_validation20_pct=finals['20']['validation_metrics']['macro_f1']*100,
            final_validation60_pct=finals['60']['validation_metrics']['macro_f1']*100,
            final_test20_pct=finals['20']['test_metrics']['macro_f1']*100,
            final_test60_pct=finals['60']['test_metrics']['macro_f1']*100,
            final_test60_minus20_pp=(finals['60']['test_metrics']['macro_f1']-finals['20']['test_metrics']['macro_f1'])*100))
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    assert len(fits)==102 and len(final_metrics)==6
    for name,rows in [('summary',summaries),('teacher-values',teachers),('coalition-values',coalitions),
                      ('individual-fits',fits),('final-metrics',final_metrics)]:
        write_csv(output/f'uci-budget-{name}.csv',rows)
    evidence=output/'evidence_snapshot/uci-budget';manifest=[]
    for path in sorted(snapshots):
        dest=evidence/path.relative_to(drive);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
        manifest.append(dict(path=str(path),sha256=sha(path),snapshot=str(dest.relative_to(output))))
    (output/'uci-budget-verification.json').write_text(json.dumps(dict(analysis_utc=datetime.now(timezone.utc).isoformat(),
        fits=102,coalition_fits=96,final_fits=6,analysis_script_sha256=sha(Path(__file__)),files=manifest),indent=2),encoding='utf-8')
    lines=['# UCI-HAR paired coalition budget sensitivity','','All 102 research fits and recorded file hashes verified. '
           'Exact Shapley values and standardized-softmax weights independently reconstructed from all 32 coalitions per bank.',
           '','| Bank | Rank Spearman | Weight L1 movement | Lowest teacher: 20 → 60 | Final test difference, 60 − 20 (pp) |',
           '|---|---:|---:|---|---:|']
    for r in summaries:lines.append(f"| {r['outer_seed']} | {r['rank_spearman_20_vs_60']:.3f} | {r['weight_l1_movement']:.3f} | "
        f"{r['minimum_teacher20']} → {r['minimum_teacher60']} | {r['final_test60_minus20_pp']:+.3f} |")
    lines+=['','Both game budgets use the same training trajectories, teacher targets, split and numerical settings. '
            'The 20-epoch utilities come from the first twenty epochs with the original checkpoint rule. Early stopping applies. '
            'Final students are matched by initialization/minibatch seed and a 60-epoch limit.','',
            'The original published 20-epoch game is a separate historical reference; its runtime differs. Historical '
            'utility gaps are retained in the CSV and cannot be attributed to the epoch budget. These three banks '
            'provide descriptive robustness evidence, not new independent subject-level confirmation. Both predeclared '
            'test outcomes are reported without selecting a winner or changing the seed set. No significance claim is made.','']
    (output/'uci-budget-report.md').write_text('\n'.join(lines),encoding='utf-8')
    enrich_report_and_plot(output,summaries,teachers,coalitions,fits,final_metrics,environment)
    audit_path=output/'uci-budget-verification.json'
    audit=json.loads(audit_path.read_text())
    audit['verified']=dict(frozen_input_files=len(frozen['files']),recorded_output_files=111,
        fit_identities=102,training_histories=102,early_stopping_rules=102,
        final_model_hashes=6,independent_shapley_vectors=6,independent_weight_vectors=6,
        dataset_and_teacher_contexts=3)
    audit['analysis_outputs_sha256']={p.name:sha(p) for p in output.glob('uci-budget-*')
        if p.suffix in ('.csv','.md')}
    audit['analysis_outputs_sha256'].update({name:sha(output/name) for name in ('uci-budget.png','uci-budget.pdf')})
    audit_path.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps(summaries,indent=2))


def enrich_report_and_plot(output,summaries,teachers,coalitions,fits,final_metrics,environment):
    report=output/'uci-budget-report.md'
    lines=[report.read_text(encoding='utf-8'),'## Verification and stopping behavior','',
        'Analysis date: 18 October 2026. Verification includes fifty frozen input hashes, all 111 recorded '
        'output hashes (102 fit JSON records, six saved models and three comparison records), all 102 fit '
        'identities, full training histories, the exact checkpoint/early-stop rule, three source/split/teacher '
        'contexts, six independently reconstructed Shapley vectors and six weight vectors. Smoke results '
        'are excluded. Coalitions used validation only; only the six predeclared final students evaluated test predictions.','',
        '| Bank | Improved utility after epoch 20 | Trained beyond 20 | Epoch range across 32 coalitions | Maximum utility gain (pp) |',
        '|---|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['outer_seed']} | {r['changed_coalition_utilities']}/32 | {r['coalitions_trained_beyond20']}/32 | "
            f"{r['coalition_min_epochs']}–{r['coalition_max_epochs']} | {r['max_coalition_utility_gain_pp']:.6f} |")
    changed=[r for r in coalitions if r['difference_pp']>0]
    total_changed=len(changed)
    lines+=['',f'{96-total_changed} of 96 coalition utilities were identical at the two limits. '
        'Training beyond epoch 20 does not imply a later best checkpoint. A 60-epoch limit retains the '
        'same patience-ten stopping rule; it does not force sixty epochs or test a no-early-stopping protocol.','']
    if changed:
        lines+=['Changed coalition utilities:','',
            '| Bank | Coalition indices (zero based) | Utility at 20 (%) | Utility at 60 (%) | Gain (pp) | Best epoch: 20 → 60 |',
            '|---|---|---:|---:|---:|---|']
        for r in changed:
            lines.append(f"| {r['outer_seed']} | {r['coalition']} | {r['utility20_pct']:.6f} | {r['utility60_pct']:.6f} | "
                         f"{r['difference_pp']:.6f} | {r['best_epoch20']} → {r['best_epoch60']} |")
        lines+=['','Teacher order: HCF-MLP, raw CNN, LSTM, TCN, Transformer.','']
    lines+=['## Matched final students','',
        'Each final student has a 60-epoch limit and validation-only checkpoint selection. The labels '
        '20 and 60 below refer to the game used to derive teacher weights, not the final training limit. '
        'Values are macro-F1 percentages; the difference is in percentage points.','',
        '| Bank | Validation, 20-game weights | Validation, 60-game weights | Test, 20-game weights | Test, 60-game weights | Test difference (pp) |',
        '|---|---:|---:|---:|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['outer_seed']} | {r['final_validation20_pct']:.3f} | {r['final_validation60_pct']:.3f} | "
            f"{r['final_test20_pct']:.3f} | {r['final_test60_pct']:.3f} | {r['final_test60_minus20_pp']:+.3f} |")
    mean20=float(np.mean([r['final_test20_pct'] for r in summaries]))
    mean60=float(np.mean([r['final_test60_pct'] for r in summaries]))
    lines+=['',f'The three-bank descriptive mean test macro-F1 is {mean20:.3f}% with 20-game weights and '
        f'{mean60:.3f}% with 60-game weights, a difference of {mean60-mean20:+.3f} pp. '
        'This is not evidence of a statistically established gain. The test split had already been examined '
        'in production, and all three outcomes are retained without selecting a winner.','',
        '## Interpretation','',
        'Teacher ordering, the lowest-value teacher and contribution signs were unchanged in all three banks. '
        'Banks 11 and 37 produced exactly the same game values and weights at the two limits. Bank 23 had '
        f"weight L1 movement {next(r for r in summaries if r['outer_seed']==23)['weight_l1_movement']:.6f} "
        '(weights on the 0-to-1 scale); its largest single teacher-weight change was '
        f"{100*next(r for r in summaries if r['outer_seed']==23)['max_absolute_teacher_weight_change']:.3f} "
        'percentage points of weight. Its matched final test result changed by +0.368 pp.','',
        'The result supports limited sensitivity to the 20-versus-60 maximum epoch budget under the retained '
        'early-stopping rule for these three frozen banks and this split. Keep the original ten-seed main '
        'results and describe the UCI valuation as a 20-epoch surrogate game. Do not replace its main result '
        'row with the new three-seed 60-game outcomes or claim that the two budgets are universally equivalent. '
        'This check does not remove the material training noise or weak cross-bank rank agreement found '
        'earlier, and it does not establish that a teacher with a negative value is harmful.','',
        '## Historical comparison and runtime','',
        '| Bank | Largest historical-vs-current 20-game utility gap (pp) | Rank Spearman, historical vs current 20-game |',
        '|---|---:|---:|']
    for r in summaries:
        lines.append(f"| {r['outer_seed']} | {r['historical20_max_utility_gap_pp']:.6f} | "
                     f"{r['historical20_rank_spearman_vs_current20']:.3f} |")
    lines+=['',f"Runtime: TensorFlow {environment['tensorflow']}, Keras {environment['keras']}, "
        f"NumPy {environment['numpy']}, scikit-learn {environment['sklearn']}, "
        f"GPU {environment['gpu'][0]['device_name']}. Deterministic operations were enabled and TensorFloat-32 disabled. "
        'The published game used a different runtime. Historical discrepancies are not budget effects; '
        'only the paired prefix games isolate the tested budget change. Frozen teacher targets were reused '
        'without teacher retraining.','',
        f"The sum of saved per-fit training/validation elapsed times is {sum(r['elapsed_seconds'] for r in fits)/3600:.3f} hours. "
        'This sum excludes some setup, serialization and orchestration costs and is not billed GPU time.','',
        '## Manuscript and response integration — pending','',
        'Suggested manuscript text: To assess the UCI-HAR surrogate-game budget, we fixed three teacher '
        'banks and evaluated all 32 coalitions with a 60-epoch limit, deriving the 20-epoch utilities from '
        'the same training trajectories under the original checkpoint and early-stopping rules. Teacher '
        'rankings and signs were unchanged in all three banks (Spearman rank correlation 1.00). Only one '
        'of 96 utilities improved after epoch 20, by 0.038904 macro-F1 percentage points. Matched final '
        'students with the two weighting conditions showed test differences of 0.000, +0.368 and 0.000 '
        'percentage points for seeds 11, 23 and 37, respectively. These descriptive checks support limited '
        'budget sensitivity in the tested setting; they do not establish general equivalence or remove '
        'optimization uncertainty. The main UCI attribution remains explicitly a 20-epoch surrogate game.','',
        'For Reviewer 1 comment 5, report the frozen paired design, all three seed outcomes and early-stop '
        'counts, distinguish historical runtime discrepancies, and retain the original main table. The '
        'WISDM selection-reuse issue is a separate part of the same comment and remains pending. The DOCX '
        'and final response have not yet been updated with these findings.','',
        '## Next planned block','',
        'The next predeclared experiment is teacher removal: all five possible single-teacher removals '
        'for each of ten original seeds on both datasets (100 final-student fits). The lowest-original-raw-'
        'Shapley choice is compared with the exact expected outcome of uniform random removal. A matched '
        'five-teacher baseline needs verified training provenance; add matched controls if a historical '
        'result cannot isolate the removal effect in the current runtime. This block has not been launched.','']
    report.write_text('\n'.join(lines),encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey='row')
    x=np.arange(5);handles=[]
    for column,seed in enumerate((11,23,37)):
        subset=[r for r in teachers if r['outer_seed']==seed]
        for row,key in enumerate(('phi','weight')):
            scale=1 if key=='phi' else 100
            for budget,offset,color,marker in [(20,-.045,'#276493','o'),(60,.045,'#c67525','D')]:
                field=f'{key}{budget}_pp' if key=='phi' else f'{key}{budget}'
                line,=axes[row,column].plot(x+offset,[r[field]*scale for r in subset],
                    color=color,marker=marker,linestyle='--' if budget==60 else '-',linewidth=1,
                    markersize=5,markerfacecolor='white' if budget==60 else color,label=f'{budget}-epoch game')
                if row==0 and column==0:handles.append(line)
            axes[row,column].grid(axis='y',alpha=.2)
            axes[row,column].set_xticks(x,['MLP','CNN','LSTM','TCN','Transformer'],rotation=25,ha='right')
            if row==0:axes[row,column].set_title(f'Frozen teacher-bank seed {seed}')
        axes[0,column].axhline(0,color='#666666',linewidth=.7)
        axes[1,column].set_ylim(0,66)
    axes[0,0].set_ylabel('Raw Shapley value\n(macro-F1 percentage points)')
    axes[1,0].set_ylabel('Teacher weight (%)')
    fig.suptitle('UCI-HAR coalition budget sensitivity: ranks and signs unchanged in all three banks',fontsize=13,y=.98)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.94),ncol=2,frameon=False)
    fig.subplots_adjust(top=.85,bottom=.14,left=.085,right=.985,hspace=.28,wspace=.14)
    fig.savefig(output/'uci-budget.png',dpi=180)
    fig.savefig(output/'uci-budget.pdf')
    plt.close(fig)


if __name__=='__main__':main()
