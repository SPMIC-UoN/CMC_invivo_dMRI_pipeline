# src/invivo_dmri_pipeline/nodes/bvals_round.py

import argparse, sys
import numpy as np

class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

parser = MyParser(prog='bvals_round')

required = parser.add_argument_group('Required arguments')
required.add_argument("-in", metavar='<path>', dest='bval_path_in', required=True, help='Input bvals file')
required.add_argument("-out", metavar='<path>', dest='bval_path_out', required=True, help='Output bvals file')
required.add_argument("-blist", metavar='<list>', dest='round_vec', type=lambda s: [int(item) for item in s.split(',')], required=True, help='Comma separated b-values vector')
required.add_argument("-tol", metavar='<int>', dest='bval_tolerance', type=int, required=True, help='Tolerance')

argsa = parser.parse_args()

# Extract parsed arguments
bval_path_in = argsa.bval_path_in
bval_path_out = argsa.bval_path_out
round_vec = argsa.round_vec
bval_tolerance = argsa.bval_tolerance

# Load b-vals
bvals = np.loadtxt(bval_path_in, dtype=int)
new_bvals = bvals.copy()

# Loop through each bval and round to closest value in round_vec within tolerance
for i in range(len(bvals)):
    for j in range(len(round_vec)):
        if abs(bvals[i] - round_vec[j]) < bval_tolerance:
            new_bvals[i] = round_vec[j]

# Write the new bvals to the output file
with open(bval_path_out, 'w') as f:
    f.write(' '.join(map(str, new_bvals)) + '\n')
