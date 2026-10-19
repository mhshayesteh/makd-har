"""Verify current loaders reproduce recorded data partitions without training."""
import argparse
import json
import sys
import zipfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--drive',type=Path,required=True)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    sys.path.insert(0,str(args.drive/'revision_2/pipeline_snapshot'))
    from makd_har.data import load_uci_har,load_wisdm_v1
    cache=args.cache.resolve()
    cache.mkdir(parents=True,exist_ok=True)
    if not list(cache.rglob('train/X_train.txt')):
        with zipfile.ZipFile(args.drive/'resources/UCI HAR Dataset.zip') as z:
            for member in z.infolist():
                resolved=(cache/member.filename).resolve()
                if resolved!=cache and cache not in resolved.parents:
                    raise RuntimeError('Unsafe archive path')
            z.extractall(cache)
    candidates=list(cache.rglob('train/X_train.txt'))
    assert len(candidates)==1
    report={}
    for dataset,loader,source in [
        ('uci',lambda:load_uci_har(candidates[0].parent.parent,validation_fraction=.2,split_seed=2026),'results/uci/production'),
        ('wisdm',lambda:load_wisdm_v1(args.drive/'resources/WISDM_ar_v1.1_raw.txt',validation_fraction=.2,test_fraction=.2,split_seed=2026),'results/wisdm/production-v2')]:
        bundle=loader()
        manifest=bundle.manifest()
        expected=json.loads((args.drive/source/'seed-11.json').read_text())['dataset_manifest']
        if manifest!=expected:
            raise RuntimeError(f'{dataset} data does not reproduce production manifest')
        report[dataset]={'passed':True,'manifest':manifest,'test_predictions_evaluated':False}
        print(f'{dataset}: original subjects, sample counts, class counts and preprocessing manifest reproduced.',flush=True)
        del bundle
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()
