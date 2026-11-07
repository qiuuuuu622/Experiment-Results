# SPDX-License-Identifier: Apache-2.0
"""Representative top-k regression; full historical matrix remains archived."""
import argparse

from .common import check_vllm, device_info, save
from .verify_topk import check_case, check_ragged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    check_vllm()
    report = dict(status='running', physical_gpu=3, device=device_info(), checks=[])
    try:
        for rows, width in [(1, 4096), (8, 65536), (32, 262144), (513, 16384)]:
            for k in (512, 2048):
                for distribution in ('random', 'equal_nonzero', 'tie_bin_overflow'):
                    for backend in ('streaming', 'cluster'):
                        report['checks'].append(
                            check_case(rows, width, k, distribution, backend)
                        )
                        save(args.output, report)
        report['ragged'] = check_ragged()
        report['status'] = 'passed'
    except BaseException as exc:
        report.update(status='failed', error=repr(exc))
        raise
    finally:
        save(args.output, report)
    print(f"Passed {len(report['checks'])} plus {len(report['ragged'])} ragged")


if __name__ == '__main__':
    main()
