# Copyright 2026 IPOQ, LLC
# SPDX-License-Identifier: Apache-2.0
"""Build against the SEAL submodule pinned by SEAL-Python v4.4.0."""
from pathlib import Path
import json,os,shutil,subprocess,tarfile,hashlib,sys
ROOT=Path(__file__).resolve().parent

def main():
    (ROOT/'results').mkdir(exist_ok=True)
    vendor=ROOT/'vendor';meta=json.loads((vendor/'matched_source.json').read_text())
    pocket=json.loads((vendor/'pocketfft_source.json').read_text())
    for file in pocket['files']:
        if hashlib.sha256((vendor/file['file']).read_bytes()).hexdigest()!=file['sha256']:
            raise ValueError('PocketFFT dependency checksum mismatch')
    source=vendor/meta['source_dir'];archive=vendor/meta['archive']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=meta['sha256']:raise ValueError('Source checksum mismatch')
    if not source.exists():
        with tarfile.open(archive) as f:f.extractall(vendor,filter='data')
    cmake=os.environ.get('FHE_CMAKE') or shutil.which('cmake')
    if not cmake:raise RuntimeError('Install cmake or set FHE_CMAKE to its executable')
    hexl='--hexl' in sys.argv
    prefix=vendor/'accelerated_install'
    if hexl:
        for label,extra in [('cpu_features',['-DBUILD_PIC=ON','-DBUILD_TESTING=OFF','-DBUILD_EXECUTABLE=OFF']),
                             ('hexl',['-DHEXL_TESTING=OFF','-DHEXL_BENCHMARK=OFF','-DHEXL_SHARED_LIB=OFF','-DHEXL_DEBUG=OFF'])]:
            info=json.loads((vendor/(label+'_source.json')).read_text());arc=vendor/info['archive'];src=vendor/info['source_dir']
            if hashlib.sha256(arc.read_bytes()).hexdigest()!=info['sha256']:raise ValueError('Dependency checksum mismatch')
            if not src.exists():
                with tarfile.open(arc) as f:f.extractall(vendor,filter='data')
            out=vendor/(label+'_build')
            subprocess.run([cmake,'-S',str(src),'-B',str(out),'-DCMAKE_BUILD_TYPE=Release',
                '-DCMAKE_POSITION_INDEPENDENT_CODE=ON','-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
                '-DCMAKE_INSTALL_PREFIX='+str(prefix),'-DCMAKE_PREFIX_PATH='+str(prefix),*extra],check=True)
            subprocess.run([cmake,'--build',str(out),'-j','4'],check=True)
            subprocess.run([cmake,'--install',str(out)],check=True)
    build=vendor/('hexl_seal_build' if hexl else 'matched_build')
    subprocess.run([cmake,'-S',str(source),'-B',str(build),'-DCMAKE_BUILD_TYPE=Release',
                    '-DSEAL_USE_MSGSL=OFF','-DSEAL_USE_ZLIB=OFF','-DSEAL_USE_ZSTD=OFF',
                    '-DSEAL_BUILD_DEPS=OFF','-DCMAKE_POSITION_INDEPENDENT_CODE=ON',
                    '-DSEAL_USE_INTEL_HEXL='+('ON' if hexl else 'OFF'),'-DCMAKE_PREFIX_PATH='+str(prefix)],check=True)
    subprocess.run([cmake,'--build',str(build),'-j','4'],check=True)
    command=['c++','-O3','-march=native','-std=c++17','-fPIC','-shared','-fvisibility=hidden','-Wl,-Bsymbolic',
             '-I'+str(source/'native'/'src'),'-I'+str(build/'native'/'src'),
             *(['-I'+str(prefix/'include')] if hexl else []),
             str(ROOT/'client_native.cpp'),str(next((build/'lib').glob('libseal-*.a'))),
             *([str(p) for p in (prefix/'lib').glob('libhexl.a')]+[str(p) for p in (prefix/'lib').glob('libcpu_features.a')] if hexl else []),
             '-pthread','-o',str(ROOT/('client_native_hexl.so' if hexl else 'client_native.so'))]
    # Public C entry points are annotated explicitly in the source.
    subprocess.run(command,check=True)
    (ROOT/'results'/('build_hexl.json' if hexl else 'build.json')).write_text(json.dumps({'seal_source':meta,'hexl':hexl,'bridge_command':command,
        'compiler':subprocess.check_output(['c++','--version'],text=True).splitlines()[0]},indent=2)+'\n')

if __name__=='__main__':main()
