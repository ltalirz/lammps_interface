#!/usr/bin/env python

import sys

if len(sys.argv) != 2:
    print('usage: %s [pdb file]'%(sys.argv[0]))
    sys.exit()

file_stream = open(sys.argv[1], 'r')
# pdb file, burn 

atoms = []
for j in file_stream.readlines():
    if j.startswith("REMARK"):
        continue
    if j.startswith("ATOM"):
        l = j.strip().split()
        atoms.append((int(l[1]), l[2], float(l[5]), float(l[6]), float(l[7]), l[10]))

for a in atoms:
    line1 = "(%i, {'element':'%s',\n"%(a[0],a[5])
    nspaces = line1.index("'elem")
    line1 += " "*nspaces
    line1 += "'special_flag':'%s',\n"%a[1]
    line1 += " "*nspaces
    line1 += "'cartesian_coordinates':np.array([%f,%f,%f])\n"%(a[2], a[3], a[4])
    line1 += " "*nspaces
    line1 += "}\n" + " "*(nspaces-2) + "),"
    print(line1)
