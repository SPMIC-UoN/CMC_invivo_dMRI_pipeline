# src/dmri_pipeline/nodes/reg2std.py

# script adapted from the F99 registration script from xtract (original written by Saad Jbabdi)

import argparse
import re, sys, os
import subprocess

FSLDIR = os.getenv('FSLDIR')
FSLbin = os.path.join(FSLDIR, 'bin')

# some useful functions
def errchk(errflag):
    if errflag:
        print("Exit without doing anything..")
        quit()

def imgtest(fname):
    r = subprocess.run([f'{os.path.join(FSLbin, "imtest")} {fname}'], capture_output=True, text=True, shell=True)
    return int(r.stdout)

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

class Usage(Exception):
    def __init__(self, msg):
        self.msg = msg

parser = MyParser(prog='reg2std',
                  description='Runs registrations to standard space')

required = parser.add_argument_group('Required arguments')
optional = parser.add_argument_group('Optional arguments')

required.add_argument("-fa", metavar='<path>', required=True, help="Path to FA map")
required.add_argument("-atl", metavar='<path>', required=True, help="Target standard space")
required.add_argument("-out", metavar='<path>', required=True, help="Output directory")
optional.add_argument("-config", metavar='<path>', required=False, help="Config file (default is $FSLDIR/data/xtract_data/standard/F99/config)")
argsa = parser.parse_args()

# start
fa = argsa.fa
atl = argsa.atl
out = argsa.out
config = argsa.config

errflag=0
# basic data checks
if imgtest(fa) == 0:
    print(f'Error: {fa} not found')
    errflag = 1

if imgtest(atl) == 0:
    print(f'Error: {atl} not found')
    errflag = 1
errchk(errflag)

if config is not None:
    if os.path.exists(config):
        print(f'Error: {atl} not found')
        errflag = 1
else:
    config = os.path.join(os.getenv('FSLDIR'), 'data', 'xtract_data', 'standard', 'F99', 'config')
errchk(errflag)

if os.path.isdir(out):
    print('Warning: output directory already exists. May overwrite existing files')
else:
    os.makedirs(out)

# start processing
print('Initial affine registration')
r = subprocess.run([os.path.join(FSLbin, 'flirt'), "-in", fa, "-ref", atl, "-omat", os.path.join(out, 'stdreg_anat_to_std.mat'),
                "-out", os.path.join(out, 'stdreg_anat_to_std')])

print('FNIRT with custom config file')
r = subprocess.run([os.path.join(FSLbin, 'fnirt'), f'--in={fa}', f'--ref={atl}', f'--aff={os.path.join(out, "stdreg_anat_to_std.mat")}',
                f'--iout={os.path.join(out, "stdreg_anat_to_std_nonlin")}', f'--cout={os.path.join(out, "stdreg_anat_to_std_warp")}', f'--config={config}'])
r = subprocess.run([os.path.join(FSLbin, 'invwarp'), '-w', os.path.join(out, "stdreg_anat_to_std_warp"), '-r', fa, '-o', os.path.join(out, "stdreg_std_to_anat_warp")])

print('Check registration')
r = subprocess.run([os.path.join(FSLbin, 'applywarp'), '-i', atl, '-o', os.path.join(out, "std_to_anat_nonlin"), '-r', fa,
                '-w', os.path.join(out, "stdreg_std_to_anat_warp")])
r = subprocess.run([os.path.join(FSLbin, 'applywarp'), '-i', fa, '-o', os.path.join(out, "anat_to_std_nonlin"), '-r', atl,
                '-w', os.path.join(out, "stdreg_anat_to_std_warp")])

print('Done!')
quit()
