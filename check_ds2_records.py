"""Compare each permitted DS2 recording with TV2's saved seed-42 predictions."""
import csv
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent_tools import ALLOWED_RECORD_IDS, CLASSES, execute_tool

ROOT = Path(__file__).resolve().parent
MODEL_SHA256 = '7d19389f652832c31e94cd6f1b168dd64e1d27df68fa573efbb4768cd5ca20cf'


def main():
    model_path = ROOT / 'models/tv2/exports/cnn_rr_fp32.onnx'
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != MODEL_SHA256:
        raise ValueError('ONNX khác mô hình TV2 đã dùng tạo bảng tham chiếu.')
    reference_path = ROOT / 'reports/tv2/ds2_deployed_record_counts.csv'
    with reference_path.open(encoding='utf-8-sig', newline='') as stream:
        references = {row['record_id']: row for row in csv.DictReader(stream)}
    if set(references) != set(ALLOWED_RECORD_IDS):
        raise ValueError('Danh sách bảng tham chiếu không khớp 21 bản DS2.')

    results = []
    for record_id in ALLOWED_RECORD_IDS:
        try:
            actual = execute_tool('classify_record', {'record_id': record_id})
            expected = references[record_id]
            passed = (
                actual['record_id'] == record_id
                and actual['dataset_split'] == 'DS2'
                and actual['mode'] == 'onnx'
                and actual['model_version'] == '20261008_094922'
                and actual['processed_count'] == int(expected['processed_count'])
                and actual['class_counts'] == {label: int(expected[label]) for label in CLASSES}
                and actual['unclassified_count'] == 0
            )
            row = {'record_id': record_id, 'passed': passed, 'api_result': actual}
            if not passed:
                row['expected_counts'] = {label: int(expected[label]) for label in CLASSES}
                row['expected_processed_count'] = int(expected['processed_count'])
        except Exception as error:
            row = {'record_id': record_id, 'passed': False, 'error_type': type(error).__name__}
            if isinstance(error, ValueError):
                row['reason'] = str(error)
        results.append(row)
        print(f"{'PASS' if row['passed'] else 'FAIL'}: record {record_id}", flush=True)

    output = ROOT / 'outputs'
    output.mkdir(exist_ok=True)
    passed = sum(row['passed'] for row in results)
    report = {
        'checked_at': datetime.now(timezone.utc).isoformat(),
        'model_sha256': MODEL_SHA256,
        'passed_count': passed,
        'total_count': len(results),
        'all_passed': passed == len(results),
        'scope': 'Per-record predicted counts via API versus historical TV2 CNN+RR seed42. This is not per-beat probability equality or a new training run.',
        'records': results,
    }
    path = output / f'ds2_records_check_{uuid.uuid4().hex}.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'KET QUA: {passed}/{len(results)} ban DS2 khop bang TV2.')
    print(f'Da luu: {path}')
    if not report['all_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
