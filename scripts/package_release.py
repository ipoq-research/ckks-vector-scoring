"""Build a source-only release from an explicit allowlist and record checks."""
# SPDX-License-Identifier: Apache-2.0
import argparse
import ast
import hashlib
import json
import posixpath
import re
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ENGINE = '''client_native.cpp client_retained.cpp fresh_backend.cu modadd.c
native_client.py fresh_backend.py packed_backend.py raw_backend.py profile_client.py
seal_reproduction.py reproduce.py profile_phases.py build_native.py build_phase.py
build_fresh.py benchmark_packed.py benchmark_import.py verify_import.py verify_crt52.py
requirements.txt'''.split()
VENDOR = '''cpu_features_source.json cpu_features_source.tar.gz hexl_source.json
hexl_source.tar.gz matched_source.json pocketfft_LICENSE.md pocketfft_hdronly.h
pocketfft_source.json seal_matched_source.tar.gz'''.split()
TOP = '''README.md LICENSE NOTICE LICENSING.md THIRD_PARTY_NOTICES.md SECURITY.md
CONTRIBUTING.md CITATION.cff AUTHORS.md .gitignore run_experiment.py verify_receipt.py'''.split()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selection():
    names = TOP + ['engine/' + n for n in ENGINE] + ['engine/vendor/' + n for n in VENDOR]
    # First-party release-only directories. Generated Python caches are excluded.
    for folder, extensions in [('docs', {'.md'}), ('scripts', {'.py', '.sh'}),
                               ('tests', {'.py'}), ('licenses', {'.txt'})]:
        names.extend(str(p.relative_to(ROOT)) for p in (ROOT / folder).iterdir() if p.is_file() and p.suffix in extensions)
    names += ['evidence/historical/' + n + '.json' for n in ['packed_gpu', 'import_validation_gpu', 'crt52_gpu']]
    names += ['evidence/audit/' + n + '.json' for n in ['original_source_hashes', 'provenance', 'packed_cpu',
              'import_validation_cpu', 'crt52_cpu', 'cpu_receipt', 'evidence_check', 'package_checks']]
    require(len(names) == len(set(names)), 'Duplicate release selection')
    return sorted(names)


def validate(names):
    findings = []
    text_count = python_count = link_count = 0
    patterns = {
        'private_key': r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
        'github_token': r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})\b',
        'aws_access_key': r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
        'openai_key': r'\bsk-(?:proj-)?[A-Za-z0-9_-]{30,}\b',
        'credential_url': r'https?://[^\s/@:]+:[^\s/@]+@',
        'workspace_path': '/' + r'(?:workspace|mnt/data|root|home)/',
    }
    for name in names:
        if name == 'evidence/audit/package_checks.json':
            continue
        path = ROOT / name
        require(path.is_file() and not path.is_symlink(), 'Missing or symlinked file: ' + name)
        require(path.suffix not in {'.so', '.a', '.o', '.key', '.pem'}, 'Excluded file type')
        if path.suffix == '.py':
            ast.parse(path.read_text(), filename=name)
            python_count += 1
        if name.startswith('engine/vendor/'):
            continue
        source = path.read_text()
        text_count += 1
        for label, pattern in patterns.items():
            if re.search(pattern, source):
                findings.append({'path': name, 'pattern': label})
        if path.suffix == '.md':
            for href in re.findall(r'\[[^\]]+\]\(([^)]+)\)', source):
                if '://' in href or href.startswith('#'):
                    continue
                target = (path.parent / href.split('#', 1)[0]).resolve()
                require(target.is_relative_to(ROOT), 'Link escapes release: ' + name)
                require(str(target.relative_to(ROOT)) in names, 'Unpackaged link target: ' + href)
                link_count += 1
    require(not findings, 'Sensitive-pattern matches (review before packaging): ' + json.dumps(findings))
    archives = {}
    for label in ['matched', 'hexl', 'cpu_features']:
        meta = json.loads((ROOT / 'engine/vendor' / (label + '_source.json')).read_text())
        archive = ROOT / 'engine/vendor' / meta['archive']
        require(digest(archive) == meta['sha256'], 'Dependency digest mismatch')
        with tarfile.open(archive) as stream:
            members = stream.getmembers()
            for member in members:
                require(not member.name.startswith('/') and '..' not in Path(member.name).parts, 'Unsafe archive path')
                require(not member.isdev() and not member.isfifo(), 'Special file in archive')
                if member.issym() or member.islnk():
                    target = posixpath.normpath(posixpath.join(posixpath.dirname(member.name) if member.issym() else '', member.linkname))
                    require(not target.startswith('/') and '..' not in Path(target).parts, 'Unsafe archive link')
        archives[meta['archive']] = {'sha256': digest(archive), 'members': len(members), 'path_checks_passed': True}
    pocket = json.loads((ROOT / 'engine/vendor/pocketfft_source.json').read_text())
    for entry in pocket['files']:
        require(digest(ROOT / 'engine/vendor' / entry['file']) == entry['sha256'], 'PocketFFT digest mismatch')
    return {'passed': True, 'release': '0.1.0', 'review_scope': 'Packaging and limited pattern checks, not a security certification.',
            'selected_files_including_manifest': len(names) + 1, 'first_party_and_notice_text_files_scanned': text_count,
            'python_files_parsed': python_count, 'local_markdown_links_checked': link_count,
            'sensitive_pattern_findings': findings, 'dependency_archives': archives,
            'pocketfft_digests_verified': True, 'compiled_binaries_included': False,
            'exclusions': ['Expanded dependencies and builds', 'Private keys and credentials', 'Cloud administration and billing',
                           'Presenter controls and browser state', 'Python environments', 'Unrelated research']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT.parent / 'IPOQ_FHE_Research_v0.1.0.zip')
    args = parser.parse_args()
    names = selection()
    checks = validate(names)
    (ROOT / 'evidence/audit/package_checks.json').write_text(json.dumps(checks, indent=2) + '\n')
    lines = [digest(ROOT / name) + '  ' + name for name in names]
    (ROOT / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in sorted(names + ['SHA256SUMS']):
            info = zipfile.ZipInfo('ipoq-fhe-scoring-research/' + name, date_time=(2026, 9, 21, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o100755 if name.endswith('.sh') else 0o100644) << 16
            archive.writestr(info, (ROOT / name).read_bytes())
    with zipfile.ZipFile(args.output) as archive:
        require(archive.testzip() is None, 'ZIP CRC check failed')
    checksum = digest(args.output)
    args.output.with_suffix(args.output.suffix + '.sha256').write_text(checksum + '  ' + args.output.name + '\n')
    print(json.dumps({'path': str(args.output), 'bytes': args.output.stat().st_size, 'sha256': checksum, **checks}, indent=2))


if __name__ == '__main__':
    main()
