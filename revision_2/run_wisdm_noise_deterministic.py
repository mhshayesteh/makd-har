"""Initialize deterministic operations before frozen-teacher validation."""
import argparse
import hashlib
import json
import runpy
import sys
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--drive-root', type=Path, required=True)
    parser.add_argument('--dataset', choices=('wisdm',), default='wisdm')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    original = here / 'run_coalition_noise.py'
    frozen = json.loads((here / 'frozen-inputs.json').read_text())
    if file_hash(original) != frozen['runtime_files']['run_coalition_noise.py']:
        raise RuntimeError('The original frozen runner has changed.')
    import tensorflow as tf
    tf.keras.utils.set_random_seed(2026)
    tf.config.experimental.enable_op_determinism()
    tf.config.experimental.enable_tensor_float_32_execution(False)
    startup = dict(
        correction='deterministic startup before frozen-teacher inference',
        initial_seed=2026, deterministic_operations=True,
        tensor_float_32_execution=False,
        tensorflow=tf.__version__, keras=tf.keras.__version__,
        original_runner_sha256=file_hash(original),
        startup_launcher_sha256=file_hash(__file__),
        frozen_inputs_sha256=file_hash(here / 'frozen-inputs.json'),
        note='Student training still resets each prescribed repetition seed. No tolerance, source weights, data, or epoch budget is changed.',
    )
    record = args.drive_root / 'results/revision-2/coalition-noise/wisdm/deterministic-startup.json'
    record.parent.mkdir(parents=True, exist_ok=True)
    if record.exists() and json.loads(record.read_text()) != startup:
        raise RuntimeError('A different startup record already exists; do not mix configurations.')
    record.write_text(json.dumps(startup, indent=2, sort_keys=True), encoding='utf-8')
    print('Deterministic operations and full float32 precision enabled before teacher validation.', flush=True)
    sys.argv = [str(original), '--drive-root', str(args.drive_root), '--dataset', 'wisdm']
    if args.preflight_only:
        sys.argv.append('--preflight-only')
    runpy.run_path(str(original), run_name='__main__')


if __name__ == '__main__':
    main()
