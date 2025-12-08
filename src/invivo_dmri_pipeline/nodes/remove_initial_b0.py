# src/dmri_pipeline/nodes/remove_initial_b0.py

import argparse, os, sys, subprocess, re
import numpy as np

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message + '\n')
        self.print_help()
        sys.exit(2)

def strip_ext(name):
    name=os.path.basename(name)
    if name.endswith('.nii.gz'): name=name[:-7]
    elif name.endswith('.nii'): name=name[:-4]
    return name

parser = MyParser(prog='remove_initial_b0')
required = parser.add_argument_group('Required arguments')
required.add_argument("-indat", metavar='<list>', nargs='+', required=True, help="Space-separated list of input .nii.gz files")
required.add_argument("-outdir", metavar='<dir>', required=True, help="Output directory")
optional = parser.add_argument_group('Optional arguments')
optional.add_argument("--b0range", type=float, default=0, help="Max b-value to be considered a b=0 (default: 0)")
optional.add_argument("--outname", metavar='<name or list>', nargs='+', help="Output base name(s) without extension; one name if single input or list matching -indat")
args = parser.parse_args()

print(f"\n--- Remove initial b0 --- ")

# robustly collect and sort inputs by a numeric token in the *stem* (no extension);
# if no digits are found, fall back to lexicographic order
nii_files = [os.path.abspath(f) for f in args.indat]

def _stem(p):
    b = os.path.basename(p)
    if b.endswith('.nii.gz'):
        return b[:-7]

    if b.endswith('.nii'):
        return b[:-4]

    return b

def _num_key(p):
    s = _stem(p)
    toks = [t for t in s.split('_') if t.isdigit()]
    if toks:
        return (0, int(toks[0]), s)

    m = re.search(r'(\d+)', s)
    if m:
        return (0, int(m.group(1)), s)

    return (1, s.lower())

nii_files.sort(key=_num_key)
outdir = os.path.abspath(args.outdir)

if not os.path.exists(outdir): os.makedirs(outdir)

if args.outname is None:
    out_bases=[os.path.basename(f).replace('.nii.gz','').replace('.nii','') for f in nii_files]
else:
    if len(args.outname)==1 and len(nii_files)==1:
        out_bases=[strip_ext(args.outname[0])]
    elif len(args.outname)==len(nii_files):
        out_bases=[strip_ext(n) for n in args.outname]
    else:
        sys.exit("Error: --outname must be a single name when one input is provided, or a list with the same length as -indat.")

for nii_file,out_base in zip(nii_files,out_bases):
    base=os.path.basename(nii_file).replace('.nii.gz','').replace('.nii','')
    bval_file=nii_file.replace('.nii.gz','.bval').replace('.nii','.bval')
    bvec_file=nii_file.replace('.nii.gz','.bvec').replace('.nii','.bvec')

    if not os.path.exists(bval_file) or not os.path.exists(bvec_file):
        raise FileNotFoundError(f"Missing .bval or .bvec for {nii_file}")

    print(f'\nProcessing {base} → {out_base}...')
    bvals=np.loadtxt(bval_file)

    if bvals.ndim>1: bvals=bvals.flatten()

    if not (bvals[0] <= args.b0range and bvals[1] <= args.b0range):
        sys.exit(f"Error: First two bvals for {base} must both be ≤ {args.b0range}. Aborting.")

    out_nii=os.path.join(outdir,out_base+'.nii.gz')

    print('  Removing first volume with fslroi...')
    subprocess.run(['fslroi', nii_file, out_nii, '1', '-1'], check=True)

    print('  Updating bval and bvec...')
    bvecs=np.loadtxt(bvec_file)

    if bvecs.ndim==1: bvecs=bvecs.reshape(1,-1)

    bvals=bvals[1:]
    bvecs=bvecs[:,1:]

    if np.all(np.equal(np.mod(bvals,1),0)):
        fmt_bval='%d'
        bvals=bvals.astype(int)
    else:
        fmt_bval='%.8f'

    out_bval=os.path.join(outdir,out_base+'.bval')
    out_bvec=os.path.join(outdir,out_base+'.bvec')
    np.savetxt(out_bval,bvals[np.newaxis],fmt=fmt_bval)
    np.savetxt(out_bvec,bvecs,fmt='%.8f')

    print(f'  Saved: {os.path.basename(out_nii)}, {os.path.basename(out_bval)}, {os.path.basename(out_bvec)}')

print(f"\n--- Done! --- ")
