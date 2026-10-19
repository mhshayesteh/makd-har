"""Reproducible second-review analysis of immutable Google Drive records."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import shutil
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SEEDS = (11, 23, 37, 53, 71, 89, 107, 131, 167, 197)
TEACHERS = ('hcf_mlp', 'raw_cnn', 'lstm', 'tcn', 'transformer')
DISPLAY = ('HCF MLP', 'CNN', 'BiLSTM', 'TCN', 'Transformer')
DATASETS = {'uci': 'results/uci/production', 'wisdm': 'results/wisdm/production-v2'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def holm(values):
    p = np.asarray(values, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    out[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return out.tolist()


def paired(a, b):
    d = np.asarray(a) - np.asarray(b)
    # Exhaustive sign permutations handle ties and zeros explicitly for n=10.
    nonzero = d[d != 0]
    if len(nonzero):
        ranks = stats.rankdata(np.abs(nonzero))
        pos = ranks[nonzero > 0].sum()
        observed = min(pos, ranks.sum() - pos)
        null = np.array([np.dot(bits, ranks) for bits in itertools.product((0, 1), repeat=len(nonzero))])
        p = float(np.mean(np.minimum(null, ranks.sum() - null) <= observed + 1e-12))
    else:
        observed, p = 0., 1.
    sd = float(d.std(ddof=1))
    half = float(stats.t.ppf(.975, len(d) - 1) * sd / math.sqrt(len(d)))
    return dict(n=len(d), mean_difference_pp=float(d.mean() * 100),
                ci95_low_pp=float((d.mean() - half) * 100),
                ci95_high_pp=float((d.mean() + half) * 100),
                effect_size_dz=float(d.mean() / sd) if sd else None,
                wilcoxon_statistic=float(observed), p_raw=p,
                wins=int((d > 0).sum()), ties=int((d == 0).sum()),
                losses=int((d < 0).sum()))


def exact_shapley(coalition_values, count):
    result = np.zeros(count)
    for teacher in range(count):
        others = [i for i in range(count) if i != teacher]
        for size in range(count):
            factor = math.factorial(size) * math.factorial(count-size-1) / math.factorial(count)
            for subset in itertools.combinations(others, size):
                key = ','.join(map(str, sorted(subset)))
                enlarged = ','.join(map(str, sorted((*subset, teacher))))
                result[teacher] += factor * (coalition_values[enlarged] - coalition_values[key])
    return result


def kendall_w(ranks):
    m, n = ranks.shape
    s = float(np.sum((ranks.sum(axis=0) - m*(n+1)/2)**2))
    correction = 0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        correction += int(np.sum(counts**3-counts))
    denominator = m*m*(n**3-n) - m*correction
    return 12*s/denominator if denominator else 0.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--drive', type=Path, required=True)
    parser.add_argument('--paper', type=Path, required=True)
    args = parser.parse_args()
    out = args.paper/'results_analysis/revision-2'
    evidence = out/'evidence_snapshot'
    out.mkdir(parents=True, exist_ok=True)
    manifest, summaries, teacher_rows, raw_rows, rank_pairs = [], [], [], [], []
    weights_rows, original_tests, student_tests, metric_rows, split_rows = [], [], [], [], []
    validations = []

    def snapshot(source, relative):
        target = evidence/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        h = sha(source)
        if target.exists() and sha(target) != h:
            raise RuntimeError(f'Evidence changed since freeze: {source}')
        if not target.exists():
            shutil.copyfile(source, target)
        assert sha(target) == h
        manifest.append(dict(source=str(source), snapshot=str(target.relative_to(out)),
                             bytes=source.stat().st_size, sha256=h))
        return target

    for ds, relative in DATASETS.items():
        base = args.drive/relative
        records = []
        for seed in SEEDS:
            p = snapshot(base/f'seed-{seed}.json', Path(ds)/f'seed-{seed}.json')
            x = json.loads(p.read_text(encoding='utf-8'))
            assert x['model_seed'] == seed
            assert tuple(x['teacher_order']) == TEACHERS
            assert len(x['shapley']['coalition_values']) == 32
            phi = np.asarray(x['shapley']['values'])
            assert np.all(np.isfinite(phi))
            reconstructed = exact_shapley(x['shapley']['coalition_values'], 5)
            np.testing.assert_allclose(phi, reconstructed, rtol=0, atol=1e-12)
            efficiency_error = abs(phi.sum() - (x['shapley']['coalition_values']['0,1,2,3,4'] - x['shapley']['coalition_values']['']))
            assert efficiency_error < 1e-12
            w = np.asarray(x['shapley_weights'])
            assert np.all(w >= 0) and abs(w.sum()-1) < 1e-12
            z = (phi-phi.mean())/phi.std() if phi.std() > 1e-12 else np.zeros(5)
            e = np.exp(z/float(x['config']['shapley_softmax_temperature'])-np.max(z/float(x['config']['shapley_softmax_temperature'])))
            np.testing.assert_allclose(w, e/e.sum(), atol=1e-10, rtol=0)
            if records:
                assert x['config'] == records[0]['config']
                assert x['dataset_manifest'] == records[0]['dataset_manifest']
            records.append(x)
            ranks = stats.rankdata(-phi)
            for j,name in enumerate(TEACHERS):
                raw_rows.append(dict(dataset=ds, seed=seed, teacher=name,
                                     raw_shapley=float(phi[j]), raw_shapley_pp=float(phi[j]*100),
                                     rank_highest_is_1=float(ranks[j]), weight=float(w[j])))
            for name,m in x['methods'].items():
                if 'test' in m:
                    metric_rows.append(dict(dataset=ds, seed=seed, method=name,
                                            validation_macro_f1=m.get('validation',{}).get('macro_f1',m.get('validation_macro_f1')),
                                            test_macro_f1=m['test']['macro_f1'], test_accuracy=m['test']['accuracy']))
            weights_rows.append(dict(dataset=ds, seed=seed, raw_phi_sd_pp=float(phi.std()*100),
                                     normalized_entropy=float(-np.sum(w*np.log(w))/np.log(5)),
                                     effective_teacher_count=float(1/np.sum(w*w)),
                                     full_minus_empty_pp=float(phi.sum()*100),
                                     efficiency_residual=float(efficiency_error)))
        for filename in ('experiment-config.json','dataset-manifest.json','significance-tests.json','summary.csv'):
            if (base/filename).exists():
                snapshot(base/filename, Path(ds)/filename)
        phi = np.array([x['shapley']['values'] for x in records])
        ranks = np.array([stats.rankdata(-row) for row in phi])
        w = np.array([x['shapley_weights'] for x in records])
        corr = np.corrcoef(ranks)
        for a,b in itertools.combinations(range(10),2):
            rank_pairs.append(dict(dataset=ds, seed_a=SEEDS[a], seed_b=SEEDS[b],
                                   spearman_rho=float(corr[a,b])))
        pairs = corr[np.triu_indices(10,1)]
        summaries.append(dict(dataset=ds, seed_count=10, kendall_w=kendall_w(ranks),
                              mean_pairwise_spearman=float(pairs.mean()),
                              median_pairwise_spearman=float(np.median(pairs)),
                              min_pairwise_spearman=float(pairs.min()), max_pairwise_spearman=float(pairs.max()),
                              mean_full_minus_empty_pp=float(phi.sum(axis=1).mean()*100),
                              negative_teacher_seed_pairs=int((phi<0).sum())))
        for j,name in enumerate(TEACHERS):
            teacher_rows.append(dict(dataset=ds, teacher=name, n=10,
                                     raw_phi_mean_pp=float(phi[:,j].mean()*100),
                                     raw_phi_sd_pp=float(phi[:,j].std(ddof=1)*100),
                                     raw_phi_median_pp=float(np.median(phi[:,j])*100),
                                     raw_phi_min_pp=float(phi[:,j].min()*100), raw_phi_max_pp=float(phi[:,j].max()*100),
                                     negative_seeds=int((phi[:,j]<0).sum()),
                                     positive_seeds=int((phi[:,j]>0).sum()),
                                     rank_mean=float(ranks[:,j].mean()), rank_sd=float(ranks[:,j].std(ddof=1)),
                                     rank_min=float(ranks[:,j].min()), rank_max=float(ranks[:,j].max()),
                                     highest_rank_count=int((ranks[:,j]==1).sum()),
                                     lowest_rank_count=int((ranks[:,j]==5).sum()),
                                     weight_mean=float(w[:,j].mean()), weight_sd=float(w[:,j].std(ddof=1))))
        tests = []
        for reference in ('cnn','lstm','slfn','supervised_student','svm','uniform_teachers'):
            a = [x['methods']['proposed']['test']['macro_f1'] for x in records]
            b = [x['methods'][reference]['test']['macro_f1'] for x in records]
            tests.append(dict(dataset=ds, comparison=f'proposed vs {reference}',
                              correction_family='original six proposed comparisons within dataset',
                              design='student seeds versus one deterministic reference' if reference=='svm' else 'matched model seeds on one fixed split',
                              **paired(a,b)))
        for row, adjusted in zip(tests,holm([t['p_raw'] for t in tests])):
            row['p_holm'] = adjusted
        saved = json.loads((base/'significance-tests.json').read_text())
        for row in tests:
            old = next(z for z in saved if z['comparison']==row['comparison'])
            for key in ('p_raw','p_holm','wilcoxon_statistic'):
                assert abs(row[key]-old[key]) < 1e-12, (ds,row['comparison'],key,row[key],old[key])
        original_tests.extend(tests)
        additional = []
        for a,b in [('proposed','uniform_teachers'),('proposed','supervised_student'),('uniform_teachers','supervised_student')]:
            additional.append(dict(dataset=ds,comparison=f'{a} vs {b}',
                                   correction_family='additional three student comparisons within dataset',
                                   **paired([x['methods'][a]['test']['macro_f1'] for x in records],
                                            [x['methods'][b]['test']['macro_f1'] for x in records])))
        for row,adjusted in zip(additional,holm([t['p_raw'] for t in additional])):
            row['p_holm'] = adjusted
        student_tests.extend(additional)
        for partition in ('train','validation','test'):
            m=records[0]['dataset_manifest'][partition]
            split_rows.append(dict(dataset=ds, partition=partition, windows=m['samples'],
                                   subjects=' '.join(map(str,m['subjects'])),class_counts=json.dumps(m['class_counts'],sort_keys=True)))
        validations.append(dict(dataset=ds, ten_expected_seeds=True, configs_consistent=True,
                                manifest_consistent=True, all_32_coalitions=True,
                                exact_shapley_reconstruction=True, original_pvalues_reproduced=True))
        fig,axes=plt.subplots(1,2,figsize=(10.6,4.9),layout='constrained')
        limit=float(np.max(np.abs(phi))*100)
        im=axes[0].imshow(phi*100,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
        axes[0].set(title=f'{ds.upper()} raw Shapley values',xlabel='Teacher',ylabel='Model seed')
        axes[0].set_xticks(range(5),DISPLAY,rotation=35,ha='right');axes[0].set_yticks(range(10),SEEDS)
        fig.colorbar(im,ax=axes[0],label='Validation macro-F1 contribution (percentage points)')
        im2=axes[1].imshow(ranks,cmap='viridis_r',vmin=1,vmax=5,aspect='auto')
        axes[1].set(title='Within-seed teacher ranks',xlabel='Teacher',ylabel='Model seed')
        axes[1].set_xticks(range(5),DISPLAY,rotation=35,ha='right');axes[1].set_yticks(range(10),SEEDS)
        for i in range(10):
            for j in range(5):axes[1].text(j,i,f'{ranks[i,j]:g}',ha='center',va='center',color='white' if ranks[i,j]>3 else 'black')
        fig.colorbar(im2,ax=axes[1],ticks=range(1,6),label='Rank (1 = largest raw value)')
        fig.savefig(out/f'{ds}-shapley-stability.png',dpi=220)
        fig.savefig(out/f'{ds}-shapley-stability.pdf')
        plt.close(fig)

    teacher_metrics = []
    for ds,relative in DATASETS.items():
        records=[json.loads((evidence/ds/f'seed-{seed}.json').read_text()) for seed in SEEDS]
        for teacher in TEACHERS:
            row=dict(dataset=ds,teacher=teacher,n=10)
            for partition in ('validation','test'):
                values=np.array([x['teachers'][teacher][partition]['macro_f1'] for x in records])*100
                row[f'{partition}_f1_mean_pct']=float(values.mean())
                row[f'{partition}_f1_sd_pp']=float(values.std(ddof=1))
            row['validation_minus_test_pp']=row['validation_f1_mean_pct']-row['test_f1_mean_pct']
            teacher_metrics.append(row)
    for p in (args.paper/'01_Revision_1_Latest_v10').glob('*'):
        if p.is_file():
            manifest.append(dict(source=str(p),snapshot=None,bytes=p.stat().st_size,sha256=sha(p)))
    benchmark=args.drive/'results/svm-student-comparison-v1/benchmark-all.json'
    snapshot(benchmark,Path('controlled-benchmark')/benchmark.name)
    for p in sorted((args.drive/'revised_pipeline').rglob('*.py')):
        if '__pycache__' not in p.parts:
            manifest.append(dict(source=str(p),snapshot=None,bytes=p.stat().st_size,sha256=sha(p)))
    (out/'source-manifest.json').write_text(json.dumps(dict(
        date='2026-10-17',baseline='v10 submitted revision 1, confirmed by author in chat',
        limitations='Current hashes establish this snapshot; historical code/checkpoint identity is not independently proven.',
        files=manifest,validation=validations),indent=2),encoding='utf-8')
    for name,rows in [('raw-shapley-and-ranks.csv',raw_rows),('shapley-summary.csv',teacher_rows),
                      ('rank-agreement.csv',rank_pairs),('rank-agreement-summary.csv',summaries),
                      ('weight-scale-diagnostics.csv',weights_rows),('original-statistics-reproduced.csv',original_tests),
                      ('additional-student-comparisons.csv',student_tests),('method-metrics-by-seed.csv',metric_rows),
                      ('teacher-performance.csv',teacher_metrics),('split-summary.csv',split_rows)]:
        write_csv(out/name,rows)

    lines=['# Second review analysis of existing production results','',
           'Analysis date 17 October 2026. Submitted revision 1 baseline v10 was confirmed by the author. '
           'The twenty seed records are frozen in evidence_snapshot with source hashes in source-manifest.json. '
           'No new training was performed for this report.','',
           '## Main findings','',
           'Teacher rankings vary substantially across the recorded seeds. Exact enumeration is verified, '
           'but it does not remove optimization noise in coalition utilities. Cross-seed variation mixes '
           'changes in teacher banks with student training randomness; it does not answer the fixed-bank noise question.','']
    for s in summaries:
        lines += [f"### {s['dataset'].upper()}",'',
                  f"Kendall concordance W is {s['kendall_w']:.3f}; mean pairwise seed Spearman correlation is "
                  f"{s['mean_pairwise_spearman']:.3f} (range {s['min_pairwise_spearman']:.3f} to {s['max_pairwise_spearman']:.3f}). "
                  f"There are {s['negative_teacher_seed_pairs']}/50 negative teacher-seed values. "
                  f"Mean full-minus-empty coalition validation utility is {s['mean_full_minus_empty_pp']:+.3f} percentage points.",'',
                  '| Teacher | Raw value mean ± SD in pp | Range in pp | Negative seeds | Rank mean |',
                  '|---|---:|---:|---:|---:|']
        for r in teacher_rows:
            if r['dataset']==s['dataset']:
                lines.append(f"| {r['teacher']} | {r['raw_phi_mean_pp']:+.3f} ± {r['raw_phi_sd_pp']:.3f} | "
                             f"{r['raw_phi_min_pp']:+.3f} to {r['raw_phi_max_pp']:+.3f} | {r['negative_seeds']}/10 | {r['rank_mean']:.2f} |")
        lines += ['',f"![Raw values and ranks]({s['dataset']}-shapley-stability.png)",'']
    lines += ['## Matched student comparisons','',
              'Original p-values and Holm correction across six proposed-versus-reference comparisons per dataset '
              'were independently reproduced. The table below uses that original correction, so it remains aligned '
              'with the submitted analysis. Intervals are descriptive paired t intervals, not multiplicity-adjusted.','',
              '| Dataset | Comparison | Mean difference in pp | 95% interval in pp | Original Holm p |',
              '|---|---|---:|---:|---:|']
    for r in original_tests:
        if r['comparison'] in ('proposed vs uniform_teachers','proposed vs supervised_student'):
            lines.append(f"| {r['dataset']} | {r['comparison']} | {r['mean_difference_pp']:+.3f} | "
                         f"[{r['ci95_low_pp']:+.3f}, {r['ci95_high_pp']:+.3f}] | {r['p_holm']:.4f} |")
    lines += ['', 'The additional three-student family is provided separately in additional-student-comparisons.csv. '
              'It is an explicitly additional analysis and must not replace the original family without explanation. '
              'All intervals and tests describe model-seed variability on fixed subject partitions. '
              'The repeated SVM values are one deterministic reference, not independent retraining trials.','',
              '## Teacher performance for the main manuscript','',
              '| Dataset | Teacher | Validation F1 mean ± SD in percent | Test F1 mean ± SD in percent |',
              '|---|---|---:|---:|']
    for r in teacher_metrics:
        lines.append(f"| {r['dataset']} | {r['teacher']} | {r['validation_f1_mean_pct']:.2f} ± {r['validation_f1_sd_pp']:.2f} | "
                     f"{r['test_f1_mean_pct']:.2f} ± {r['test_f1_sd_pp']:.2f} |")
    lines += ['', 'Teacher-ensemble performance still requires inference from saved teacher models. '
              'A coalition-distilled student is not a teacher ensemble.','',
              '## Evidence and next steps','',
              '- All twenty records contain the expected seed, teacher order and 32 coalition utilities. '
              'Raw Shapley values were independently reconstructed to absolute tolerance 1e-12; efficiency identities and stored softmax weights agree.',
              '- Saved original p-values and corrected p-values reproduce to absolute tolerance 1e-12.',
              '- The corrected WISDM protocol is used throughout. Earlier OneDrive WISDM summaries are excluded.',
              '- Standardized softmax discards the absolute raw-value scale. Entropy and raw standard deviation are recorded per seed; '
              'nonuniform weights alone do not demonstrate a reliable contribution signal.',
              '- First new run: fixed-bank empty/full coalition noise, using seeds 11, 23 and 37 and five paired repetitions. '
              'Teacher validation parity and data-manifest checks must pass before those runs.',
              '- Pruning, new budget/weighting sensitivity, deployment timing, and teacher-ensemble results remain pending.',
              '', 'Statistical method reference: [SciPy Wilcoxon documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wilcoxon.html). '
              'The implementation explicitly enumerates all sign assignments for the ten paired differences, handling ties and zeros.', '']
    (out/'existing-results-report.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps(dict(rank_summaries=summaries,student_comparisons=[r for r in original_tests if 'uniform' in r['comparison'] or 'supervised' in r['comparison']],
                         output=str(out)),indent=2))


if __name__=='__main__':
    main()
