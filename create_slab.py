#!/usr/bin/env python
"""
main.py

the program starts here.

"""

import sys,os
import math
import re
import numpy as np
import networkx as nx
import ForceFields
import itertools
import operator
from structure_data import from_CIF, write_CIF, clean
from structure_data import write_RASPA_CIF, write_RASPA_sim_files, MDMC_config
from structure_data import SlabGraph
from CIFIO import CIF
from ccdc import CCDC_BOND_ORDERS
from datetime import datetime
from InputHandler import Options
from copy import deepcopy
import Molecules
if sys.version_info < (3,0):
    input = raw_input

try:
    from ase.io import *
    from ase.build import *
except:
    print("Error ASE not found! ASE import failed, cannot build initial slab guess with ASE!")
    sys.exit()

try:
    import pymatgen
    import pymatgen.io.cif as pic                                                   
    from pymatgen.core.surface import Slab, SlabGenerator,generate_all_slabs, Structure, PeriodicSite

    # Needed for custom get all unique millers
    from functools import reduce
    from pymatgen.util.coord_utils import in_coord_list
    try:
        # New Py>=3.5 import
        from math import gcd
    except ImportError:
        # Deprecated import from Py3.5 onwards.
        from fractions import gcd
except:
    print("Error Pymatgen not found! Pymatgen import failed, cannot build initial slab guess with Pymatgen!")
    sys.exit()

class LammpsSimulation(object):
    def __init__(self, options):
        self.name = clean(options.cif_file)
        self.special_commands = []
        self.options = options
        self.molecules = []
        self.subgraphs = []
        self.molecule_types = {}
        self.unique_atom_types = {}
        self.atom_ff_type = {}
        self.unique_bond_types = {}
        self.bond_ff_type = {}
        self.unique_angle_types = {}
        self.angle_ff_type = {}
        self.unique_dihedral_types = {}
        self.dihedral_ff_type = {}
        self.unique_improper_types = {}
        self.improper_ff_type = {}
        self.unique_pair_types = {}
        self.pair_in_data = True
        self.separate_molecule_types = True
        self.framework = True # Flag if a framework exists in the simulation.
        self.supercell = (1, 1, 1) # keep track of supercell size
        self.type_molecules = {}
        self.no_molecule_pair = True  # ensure that h-bonding will not occur between molecules of the same type
        self.fix_shake = {}
        self.fix_rigid = {}

    def set_MDMC_config(self, MDMC_config):
        self.MDMC_config = MDMC_config

    def unique_atoms(self, g):
        """Computes the number of unique atoms in the structure"""
        count = len(self.unique_atom_types) 
        fwk_nodes = sorted(g.nodes())
        molecule_nodes = []
        # check if this is the main graph
        if g == self.graph:
            for k in sorted(self.molecule_types.keys()):
                nds = []
                for m in self.molecule_types[k]:
                
                    jnodes = sorted(self.subgraphs[m].nodes())
                    nds += jnodes

                    for n in jnodes:
                        del fwk_nodes[fwk_nodes.index(n)]
                molecule_nodes.append(nds)
            molecule_nodes.append(fwk_nodes)

        # determine if the graph is the main structure, or a molecule template
        # this is *probably* not the best way to do it.
        moltemplate = ("Molecules" in "%s"%g.__class__)
        mainstructr = ("structure_data" in "%s"%g.__class__)
        if (moltemplate and mainstructr):
            print("ERROR: there is some confusion about class assignment with "+
                  "MolecularGraphs.  You should probably contact one of the developers.")
            sys.exit()

        for node, data in g.nodes_iter(data=True):
            if self.separate_molecule_types and molecule_nodes and mainstructr:
                molid = [j for j,mol in enumerate(molecule_nodes) if node in mol]
                molid = molid[0]
            elif moltemplate:
                # random keyboard mashing. Just need to separate this from other atom types in the
                # system. This is important when defining separating the Molecule's atom types
                # from other atom types in the framework that would otherwise be identical.
                # This allows for the construction of the molecule group for this template.
                molid = 23523523
            else:
                molid = 0
            # add factor for h_bond donors
            if data['force_field_type'] is None:
                if data['h_bond_donor']:
                    # add neighbors to signify type of hbond donor
                    label = (data['element'], data['h_bond_donor'], molid, tuple(sorted([g.node[j]['element'] for j in g.neighbors(node)])))
                else:
                    label = (data['element'], data['h_bond_donor'], molid)
            else:
                if data['h_bond_donor']:
                    # add neighbors to signify type of hbond donor
                    label = (data['force_field_type'], data['h_bond_donor'], molid, tuple(sorted([g.node[j]['element'] for j in g.neighbors(node)])))
                else:
                    label = (data['force_field_type'], data['h_bond_donor'], molid)

            try:
                type = self.atom_ff_type[label]
            except KeyError:
                count += 1
                type = count
                self.atom_ff_type[label] = type  
                self.unique_atom_types[type] = (node, data)
                if not moltemplate:
                    self.type_molecules[type] = molid
            data['ff_type_index'] = type

    def unique_bonds(self, g):
        """Computes the number of unique bonds in the structure"""
        count = len(self.unique_bond_types) 
        for n1, n2, data in g.edges_iter2(data=True):
            btype = "%s"%data['potential']

            try:
                type = self.bond_ff_type[btype]

            except KeyError:
                try: 
                    if data['potential'].special_flag == 'shake':
                        self.fix_shake.setdefault('bonds', []).append(count+1)
                except AttributeError:
                    pass
                count += 1
                type = count
                self.bond_ff_type[btype] = type

                self.unique_bond_types[type] = (n1, n2, data) 

            data['ff_type_index'] = type
    
    def unique_angles(self, g):
        count = len(self.unique_angle_types) 
        for b, data in g.nodes_iter(data=True):
            # compute and store angle terms
            try:
                ang_data = data['angles']
            
                for (a, c), val in ang_data.items():
                    atype = "%s"%val['potential']
                    try:
                        type = self.angle_ff_type[atype]

                    except KeyError:
                        count += 1
                        try: 
                            if val['potential'].special_flag == 'shake':
                                self.fix_shake.setdefault('angles', []).append(count)
                        except AttributeError:
                            pass
                        type = count
                        self.angle_ff_type[atype] = type
                        self.unique_angle_types[type] = (a, b, c, val) 
                    val['ff_type_index'] = type
                    # update original dictionary
                    data['angles'][(a, c)] = val
            except KeyError:
                # no angle associated with this node.
                pass

    def unique_dihedrals(self, g):
        count = len(self.unique_dihedral_types)
        dihedral_type = {}
        for b, c, data in g.edges_iter2(data=True):
            try:
                dihed_data = data['dihedrals']
                for (a, d), val in dihed_data.items():
                    dtype = "%s"%val['potential']
                    try:
                        type = dihedral_type[dtype]
                    except KeyError:
                        count += 1 
                        type = count
                        dihedral_type[dtype] = type
                        self.unique_dihedral_types[type] = (a, b, c, d, val)
                    val['ff_type_index'] = type
                    # update original dictionary
                    data['dihedrals'][(a,d)] = val
            except KeyError:
                # no dihedrals associated with this edge
                pass

    def unique_impropers(self, g):
        count = len(self.unique_improper_types) 
        
        for b, data in g.nodes_iter(data=True):
            try:
                rem = []
                imp_data = data['impropers']
                for (a, c, d), val in imp_data.items():
                    if val['potential'] is not None:
                        itype = "%s"%val['potential']
                        try:
                            type = self.improper_ff_type[itype]
                        except KeyError:
                            count += 1
                            type = count
                            self.improper_ff_type[itype] = type
                            self.unique_improper_types[type] = (a, b, c, d, val)

                        val['ff_type_index'] = type
                    else:
                        rem.append((a,c,d))

                for m in rem:
                    data['impropers'].pop(m)

            except KeyError:
                # no improper terms associated with this atom
                pass

    def unique_pair_terms(self):
        pot_names = []
        nodes_list = sorted(self.unique_atom_types.keys())
        electro_neg_atoms = ["N", "O", "F"]
        for n, data in self.graph.nodes_iter(data=True):
            if data['h_bond_donor']:
                pot_names.append('h_bonding')
            if data['tabulated_potential']:
                pot_names.append('table')
            pot_names.append(data['pair_potential'].name)
        # mix yourself

        table_str = ""
        if len(list(set(pot_names))) > 1 or (any(['buck' in i for i in list(set(pot_names))])):
            self.pair_in_data = False
            for (i, j) in itertools.combinations_with_replacement(nodes_list, 2):
                (n1, i_data), (n2, j_data) = self.unique_atom_types[i], self.unique_atom_types[j]
                mol1 = self.type_molecules[i]
                mol2 = self.type_molecules[j]
                # test to see if h-bonding to occur between molecules
                pairwise_test = ((mol1 != mol2 and self.no_molecule_pair) or (not self.no_molecule_pair))
                if i_data['tabulated_potential'] and j_data['tabulated_potential']:
                    table_pot = deepcopy(i_data)
                    table_str += table_pot['table_function'](i_data,j_data, table_pot)
                    table_pot['table_potential'].filename = "table." + self.name
                    self.unique_pair_types[(i, j, 'table')] = table_pot

                if (i_data['h_bond_donor'] and j_data['element'] in electro_neg_atoms and pairwise_test and not j_data['h_bond_donor']):
                    hdata = deepcopy(i_data)
                    hdata['h_bond_potential'] = hdata['h_bond_function'](n2, self.graph, flipped=False)
                    hdata['tabulated_potential'] = False
                    self.unique_pair_types[(i,j,'hb')] = hdata
                if (j_data['h_bond_donor'] and i_data['element'] in electro_neg_atoms and pairwise_test and not i_data['h_bond_donor']):
                    hdata = deepcopy(j_data)
                    hdata['tabulated_potential'] = False
                    hdata['h_bond_potential'] = hdata['h_bond_function'](n1, self.graph, flipped=True)
                    self.unique_pair_types[(i,j,'hb')] = hdata 
                # mix Lorentz-Berthelot rules
                pair_data = deepcopy(i_data)
                if 'buck' in i_data['pair_potential'].name and 'buck' in j_data['pair_potential'].name:
                    eps1 = i_data['pair_potential'].eps 
                    eps2 = j_data['pair_potential'].eps 
                    sig1 = i_data['pair_potential'].sig 
                    sig2 = j_data['pair_potential'].sig 
                    eps = np.sqrt(eps1*eps2)
                    Rv = (sig1 + sig2)
                    Rho = Rv/12.0
                    A = 1.84e5 * eps
                    C=2.25*(Rv)**6*eps

                    pair_data['pair_potential'].A = A 
                    pair_data['pair_potential'].rho = Rho
                    pair_data['pair_potential'].C = C
                    pair_data['tabulated_potential'] = False
                    # assuming i_data has the same pair_potential name as j_data
                    self.unique_pair_types[(i,j, i_data['pair_potential'].name)] = pair_data
                elif 'lj' in i_data['pair_potential'].name and 'lj' in j_data['pair_potential'].name:

                    pair_data['pair_potential'].eps = np.sqrt(i_data['pair_potential'].eps*j_data['pair_potential'].eps)
                    pair_data['pair_potential'].sig = (i_data['pair_potential'].sig + j_data['pair_potential'].sig)/2.
                    pair_data['tabulated_potential'] = False
                    self.unique_pair_types[(i,j, i_data['pair_potential'].name)] = pair_data

        # can be mixed by lammps
        else:
            for b in sorted(list(self.unique_atom_types.keys())):
                data = self.unique_atom_types[b][1]
                pot = data['pair_potential']
                self.unique_pair_types[b] = data

        if (table_str):
            f = open('table.'+self.name, 'w')
            f.writelines(table_str)
            f.close()
        return

    def define_styles(self):
        # should be more robust, some of the styles require multiple parameters specified on these lines
        self.kspace_style = "ewald %f"%(0.000001)
        bonds = set([j['potential'].name for n1, n2, j in list(self.unique_bond_types.values())])
        if len(list(bonds)) > 1:
            self.bond_style = "hybrid %s"%" ".join(list(bonds))
        elif len(list(bonds)) == 1:
            self.bond_style = "%s"%list(bonds)[0]
            for n1, n2, b in list(self.unique_bond_types.values()):
                b['potential'].reduced = True
        else:
            self.bond_style = ""
        angles = set([j['potential'].name for a,b,c,j in list(self.unique_angle_types.values())])
        if len(list(angles)) > 1:
            self.angle_style = "hybrid %s"%" ".join(list(angles))
        elif len(list(angles)) == 1:
            self.angle_style = "%s"%list(angles)[0]
            for a,b,c,ang in list(self.unique_angle_types.values()):
                ang['potential'].reduced = True
                if (ang['potential'].name == "class2"):
                    ang['potential'].bb.reduced=True
                    ang['potential'].ba.reduced=True
        else:
            self.angle_style = ""

        dihedrals = set([j['potential'].name for a,b,c,d,j in list(self.unique_dihedral_types.values())])
        if len(list(dihedrals)) > 1:
            self.dihedral_style = "hybrid %s"%" ".join(list(dihedrals))
        elif len(list(dihedrals)) == 1:
            self.dihedral_style = "%s"%list(dihedrals)[0]
            for a,b,c,d, di in list(self.unique_dihedral_types.values()):
                di['potential'].reduced = True
                if (di['potential'].name == "class2"):
                    di['potential'].mbt.reduced=True
                    di['potential'].ebt.reduced=True
                    di['potential'].at.reduced=True
                    di['potential'].aat.reduced=True
                    di['potential'].bb13.reduced=True
        else:
            self.dihedral_style = ""

        impropers = set([j['potential'].name for a,b,c,d,j in list(self.unique_improper_types.values())])
        if len(list(impropers)) > 1:
            self.improper_style = "hybrid %s"%" ".join(list(impropers))
        elif len(list(impropers)) == 1:
            self.improper_style = "%s"%list(impropers)[0]
            for a,b,c,d,i in list(self.unique_improper_types.values()):
                i['potential'].reduced = True
                if (i['potential'].name == "class2"):
                    i['potential'].aa.reduced=True
        else:
            self.improper_style = "" 
        pairs = set(["%r"%(j['pair_potential']) for j in list(self.unique_pair_types.values())]) | \
                set(["%r"%(j['h_bond_potential']) for j in list(self.unique_pair_types.values()) if j['h_bond_potential'] is not None]) | \
                set(["%r"%(j['table_potential']) for j in list(self.unique_pair_types.values()) if j['tabulated_potential']]) 
        if len(list(pairs)) > 1:
            self.pair_style = "hybrid/overlay %s"%(" ".join(list(pairs)))
        else:
            self.pair_style = "%s"%list(pairs)[0]
            for p in list(self.unique_pair_types.values()):
                p['pair_potential'].reduced = True

    def set_graph(self, graph):
        self.graph = graph

        try:
            if(not self.options.force_field == "UFF") and (not self.options.force_field == "Dreiding") and \
                    (not self.options.force_field == "UFF4MOF"):
                self.graph.find_metal_sbus = True # true for BTW_FF and Dubbeldam
            if (self.options.force_field == "Dubbeldam"):
                self.graph.find_organic_sbus = True

            self.graph.compute_topology_information(self.cell, self.options.tol, self.options.neighbour_size) 
        except AttributeError:
            # no cell set yet 
            pass

    def set_cell(self, cell):
        self.cell = cell
        try:
            self.graph.compute_topology_information(self.cell, self.options.tol, self.options.neighbour_size)
        except AttributeError:
            # no graph set yet
            pass

    def split_graph(self):

        self.compute_molecules()
        if (self.molecules): 
            print("Molecules found in the framework, separating.")
            molid=0
            for molecule in self.molecules:
                molid += 1
                sg = self.cut_molecule(molecule)
                sg.molecule_id = molid
                # unwrap coordinates
                sg.unwrap_node_coordinates(self.cell)
                self.subgraphs.append(sg)
        type = 0
        temp_types = {}
        for i, j in itertools.combinations(range(len(self.subgraphs)), 2):
            if self.subgraphs[i].number_of_nodes() != self.subgraphs[j].number_of_nodes():
                continue
            
            #TODO(pboyd): For complex 'floppy' molecules, a rigid 3D clique detection
            # algorithm won't work very well. Inchi or smiles comparison may be better,
            # but that would require using openbabel. I'm trying to keep this
            # code as independent of non-standard python libraries as possible.
            matched = self.subgraphs[i] | self.subgraphs[j]
            if (len(matched) == self.subgraphs[i].number_of_nodes()):
                if i not in list(temp_types.keys()) and j not in list(temp_types.keys()):
                    type += 1
                    temp_types[i] = type
                    temp_types[j] = type
                    self.molecule_types.setdefault(type, []).append(i)
                    self.molecule_types[type].append(j)
                else:
                    try:
                        type = temp_types[i]
                        temp_types[j] = type
                    except KeyError:
                        type = temp_types[j]
                        temp_types[i] = type
                    if i not in self.molecule_types[type]:
                        self.molecule_types[type].append(i)
                    if j not in self.molecule_types[type]:
                        self.molecule_types[type].append(j)
        unassigned = set(range(len(self.subgraphs))) - set(list(temp_types.keys()))
        for j in list(unassigned):
            type += 1
            self.molecule_types[type] = [j]

    def assign_force_fields(self):
        
        attr = {'graph':self.graph, 'cutoff':self.options.cutoff, 'h_bonding':self.options.h_bonding,
                'keep_metal_geometry':self.options.fix_metal, 'bondtype':self.options.dreid_bond_type,
                'eps_scale_factor':self.options.eps_scale_factor}
        param = getattr(ForceFields, self.options.force_field)(**attr)

        self.special_commands += param.special_commands()

        # apply different force fields.
        for mtype in list(self.molecule_types.keys()):
            # prompt for ForceField?
            rep = self.subgraphs[self.molecule_types[mtype][0]]
            #response = input("Would you like to apply a new force field to molecule type %i with atoms (%s)? [y/n]: "%
            #        (mtype, ", ".join([rep.node[j]['element'] for j in rep.nodes()])))
            #ff = self.options.force_field
            #if response.lower() in ['y','yes']:
            #    ff = input("Please enter the name of the force field: ")
            #elif response.lower() in ['n', 'no']:
            #    pass 
            #else:
            #    print("Unrecognized command: %s"%response)

            ff = self.options.mol_ff
            if ff is None:
                ff = self.options.force_field
                atoms = ", ".join([rep.node[j]['element'] for j in rep.nodes()])
                print("WARNING: Molecule %s with atoms (%s) will be using the %s force field as no "%(mtype,atoms,ff)+
                      " value was set for molecules. To prevent this warning "+
                      "set --molecule-ff=[some force field] on the command line.")
            h_bonding = False
            if (ff == "Dreiding"):
                hbonding = input("Would you like this molecule type to have hydrogen donor potentials? [y/n]: ")
                if hbonding.lower() in ['y', 'yes']:
                    h_bonding = True
                elif hbonding.lower() in ['n', 'no']:
                    h_bonding = False
                else:
                    print("Unrecognized command: %s"%hbonding)
                    sys.exit()
            for m in self.molecule_types[mtype]:
                # Water check
                # currently only works if the cif file contains water particles without dummy atoms.
                ngraph = self.subgraphs[m]
                self.assign_molecule_ids(ngraph)
                mff = ff
                if ff[-5:] == "Water":
                    self.add_water_model(ngraph, ff) 
                    mff = mff[:-6] # remove _Water from end of name
                if ff[-3:] == "CO2":
                    self.add_co2_model(ngraph, ff)
                # TODO(pboyd): should have an eps_scale_factor for molecules??
                p = getattr(ForceFields, mff)(graph=self.subgraphs[m], 
                                         cutoff=self.options.cutoff, 
                                         h_bonding=h_bonding,
                                         eps_scale_factor=1.)
                self.special_commands += p.special_commands()

    def assign_molecule_ids(self, graph):
        for node in graph.nodes():
            graph.node[node]['molid'] = graph.molecule_id

    def molecule_template(self, mol):
        """ Construct a molecule template for
        reading and insertions in a LAMMPS simulation.

        This combines two classes which have
        been separated conceptually - ForceField and
        Molecules.
        For some molecules, the force field is implicit
        within the structure (e.g. TIP5P_Water molecule
        must be used with the TIP5P ForceField).
        But one can imagine cases where this is not true
        (alkanes? CO2?).

        """
        # no error checking here, it is assumed that the user
        # knows which force field to pair with which molecule
        # I'm not sure what would happen if there were a mismatch
        # but hopefully error-checking elsewhere in the code
        # will catch these things.
        molecule = getattr(Molecules, mol)()
        if self.options.mol_ff is None:
            mol_ff = self.options.force_field

        elif self.options.mol_ff.endswith("_Water"):
            # parse if _Water is at the end to get the force
            # fields for various water models.
            mol_ff = mol[:-6]
        else:
            # just take the general force field used on the
            # framework
            mol_ff = self.options.mol_ff
        #TODO(pboyd): Check how h-bonding is handeled at this level
        ff = getattr(ForceFields, mol_ff)(graph=molecule,
                                     cutoff=self.options.cutoff)
        
        # add the unique potentials to the unique_dictionaries.
        self.unique_atoms(molecule)
        self.unique_bonds(molecule)
        self.unique_angles(molecule)
        self.unique_dihedrals(molecule)
        self.unique_impropers(molecule)
        # somehow update atom, bond, angle, dihedral, improper etc. types to 
        # include atomic species that don't exist yet..
        self.template_molecule = molecule
        template_file = "%s.molecule"%molecule.__class__.__name__
        file = open(template_file, 'w')
        file.writelines(molecule.str(atom_types=self.atom_ff_type))
        file.close()
        print('Molecule template file written as %s'%template_file)

    def add_co2_model(self, ngraph, ff):
        size = ngraph.number_of_nodes()
        if size < 3 or size > 3:
            print("Error: cannot assign %s "%(ff) +
                  "to molecule of size %i, with "%(size)+
                  "atoms (%s)"%(", ".join([ngraph.node[kk]['element'] for
                                           kk in ngraph.nodes()])))
            print("If this is a CO2 molecule with pre-existing "+
                    "dummy atoms for a particular force field, "+
                    "please remove them and re-run this code.")
            sys.exit()
        for node in ngraph.nodes():
            if ngraph.node[node]['element'] == "C":
                catom = ngraph.node[node]
            elif ngraph.node[node]['element'] == "O":
                try:
                    oatom1
                    o2id = node
                    oatom2 = ngraph.node[node]
                except NameError:
                    o1id = node
                    oatom1 = ngraph.node[node]

        co2 = getattr(Molecules, ff)()
        co2.approximate_positions(C_pos  = catom['cartesian_coordinates'],
                                  O_pos1 = oatom1['cartesian_coordinates'],
                                  O_pos2 = oatom2['cartesian_coordinates'])

        # update the co2 atoms in the graph with the force field molecule
        mol_c = deepcopy(co2.node[1])
        mol_o1 = deepcopy(co2.node[2])
        mol_o2 = deepcopy(co2.node[3])
        # hackjob - get rid of the angle data on the carbon, so that
        # the framework indexed values for each oxygen remain with the carbon atom.
        mol_c.pop('angles')
        catom.update(mol_c)
        oatom1.update(mol_o1)
        oatom2.update(mol_o2)
        #for node in ngraph.nodes():
        #    #data = deepcopy(ngraph.node[node]) # doesn't work - some of the data is 
        #                                        # specific to the molecule in the
        #                                        # framework.

        #    if data['element'] == "C":
        #        cid = node 
        #        ngraph.node[node] = co2.node[1].copy()
        #    elif data['element'] == "O":
        #        try:
        #            otm1
        #            ngraph.node[node] = co2.node[3].copy()
        #        except NameError:
        #            otm1 = node
        #            ngraph.node[node] = co2.node[2].copy()

    def add_water_model(self, ngraph, ff):
        size = ngraph.number_of_nodes()
        if size < 3 or size > 3:
            print("Error: cannot assign %s "%(ff) +
                  "to molecule of size %i, with "%(size)+
                  "atoms (%s)"%(", ".join([ngraph.node[kk]['element'] for
                                           kk in ngraph.nodes()])))
            print("If this is a water molecule with pre-existing "+
                    "dummy atoms for a particular force field, "+
                    "please remove them and re-run this code.")
            sys.exit()
        for node in ngraph.nodes():
            if ngraph.node[node]['element'] == "O":
                oid = node
                oatom = ngraph.node[node]
            elif ngraph.node[node]['element'] == "H":
                try:
                    hatom1
                    h2id = node
                    hatom2 = ngraph.node[node]
                except NameError:
                    h1id = node
                    hatom1 = ngraph.node[node]

        h2o = getattr(Molecules, ff)()
        h2o.approximate_positions(O_pos  = oatom['cartesian_coordinates'],
                                  H_pos1 = hatom1['cartesian_coordinates'],
                                  H_pos2 = hatom2['cartesian_coordinates'])

        # update the water atoms in the graph with the force field molecule
        mol_o = deepcopy(h2o.node[1])
        mol_h1 = deepcopy(h2o.node[2])
        mol_h2 = deepcopy(h2o.node[3])
        # hackjob - get rid of the angle data on the carbon, so that
        # the framework indexed values for each oxygen remain with the carbon atom.
        try:
            mol_o.pop('angles')
        except KeyError:
            pass

        oatom.update(mol_o)
        hatom1.update(mol_h1)
        hatom2.update(mol_h2)
        # update the water atoms in the graph with the force field molecule 
        #for node in ngraph.nodes():
        #    data = deepcopy(ngraph.node[node])
        #    if data['element'] == "O":
        #        oid = node 
        #        ngraph.node[node] = h2o.node[1].copy()
        #    elif data['element'] == "H":
        #        try:
        #            htm1
        #            ngraph.node[node] = h2o.node[3].copy()
        #        except NameError:
        #            htm1 = node
        #            ngraph.node[node] = h2o.node[2].copy()

        # add dummy particles
        for dx in h2o.nodes():
            if dx > 3:
                self.increment_graph_sizes()
                os = ngraph.original_size
                ngraph.add_node(os, **h2o.node[dx])
                ngraph.add_edge(oid, os, order=1.,
                                weight=1.,
                                length=h2o.Rdum,
                                symflag='1_555',
                                )
                ngraph.sorted_edge_dict.update({(oid, os): (oid, os)})
                ngraph.sorted_edge_dict.update({(os, oid): (oid, os)})
        # compute new angles between dummy atoms
        ngraph.compute_angles()


    def increment_graph_sizes(self, inc=1):
        self.graph.original_size += inc
        for mtype in list(self.molecule_types.keys()):
            for m in self.molecule_types[mtype]:
                graph = self.subgraphs[m]
                graph.original_size += 1

    def compute_simulation_size(self):
        
        if self.options.orthogonalize:
            if not (np.allclose(self.cell.alpha, 90., atol=1) and np.allclose(self.cell.beta, 90., atol=1) and\
                    np.allclose(self.cell.gamma, 90., atol=1)):

                print("WARNING: Orthogonalization of simulation cell requested. This can "+
                      "make simulation sizes incredibly large. I hope you know, what you "+
                      "are doing!")
                transformation_matrix = self.cell.orthogonal_transformation()
                self.graph.redefine_lattice(transformation_matrix, self.cell)
        supercell = self.cell.minimum_supercell(self.options.cutoff)
        if np.any(np.array(supercell) > 1):
            print("WARNING: unit cell is not large enough to"
                  +" support a non-bonded cutoff of %.2f Angstroms."%self.options.cutoff) 

        if(self.options.replication is not None):
            supercell = tuple(map(int, re.split('x| |, |,',self.options.replication)))
            if(len(supercell) != 3):
                if(supercell[0] < 1 or supercell[1] < 1 or supercell[2] < 1):
                    print("Incorrect supercell requested: %s\n"%(supercell))
                    print("Use <ixjxk> format")
                    print("Exiting...")
                    sys.exit()
        self.supercell=supercell
        if np.any(np.array(supercell) > 1):
            print("Re-sizing to a %i x %i x %i supercell. "%(supercell))
            
            #TODO(pboyd): apply to subgraphs as well, if requested.
            self.graph.build_supercell(supercell, self.cell)
            molcount = 0
            if self.subgraphs:
                molcount = max([g.molecule_id for g in self.subgraphs])
            
            for mtype in list(self.molecule_types.keys()):
                # prompt for replication of this molecule in the supercell.
                rep = self.subgraphs[self.molecule_types[mtype][0]]
                if(self.options.auto_mol_rep is None):
                    response = input("Would you like to replicate molceule %i with atoms (%s) in the supercell? [y/n]: "%
                            (mtype, ", ".join([rep.node[j]['element'] for j in rep.nodes()])))
                elif(self.options.auto_mol_rep is True):
                    response = 'y'
                else:
                    repsonse = 'n'
                    
                if response in ['y', 'Y', 'yes']:
                    for m in self.molecule_types[mtype]:
                        self.subgraphs[m].build_supercell(supercell, self.cell, track_molecule=True, molecule_len=molcount)
            self.cell.update_supercell(supercell)

    def merge_graphs(self):
        for mgraph in self.subgraphs:
            self.graph += mgraph
        for node in self.graph.nodes():
            data=self.graph.node[node]
        if sorted(self.graph.nodes()) != [i+1 for i in range(len(self.graph.nodes()))]:
            print("Re-labelling atom indices.")
            reorder_dic = {i:j+1 for i, j in zip(sorted(self.graph.nodes()), range(len(self.graph.nodes())))}
            self.graph.reorder_labels(reorder_dic)
            for mgraph in self.subgraphs:
                mgraph.reorder_labels(reorder_dic)

    def write_lammps_files(self):
        self.unique_atoms(self.graph)
        self.unique_bonds(self.graph)
        self.unique_angles(self.graph)
        self.unique_dihedrals(self.graph)
        self.unique_impropers(self.graph)
        if self.options.insert_molecule:
            self.molecule_template(self.options.insert_molecule)
        self.unique_pair_terms()
        self.define_styles()

        data_str = self.construct_data_file() 
        datafile = open("data.%s"%self.name, 'w')
        datafile.writelines(data_str)
        datafile.close()

        inp_str = self.construct_input_file()
        inpfile = open("in.%s"%self.name, 'w')
        inpfile.writelines(inp_str)
        inpfile.close()
        print("files created!")

    def construct_data_file(self):
    
        t = datetime.today()
        string = "Created on %s\n\n"%t.strftime("%a %b %d %H:%M:%S %Y %Z")
    
        if(len(self.unique_atom_types.keys()) > 0):
            string += "%12i atoms\n"%(nx.number_of_nodes(self.graph))
        if(len(self.unique_bond_types.keys()) > 0):
            string += "%12i bonds\n"%(nx.number_of_edges(self.graph))
        if(len(self.unique_angle_types.keys()) > 0):
            string += "%12i angles\n"%(self.graph.count_angles())
        if(len(self.unique_dihedral_types.keys()) > 0):
            string += "%12i dihedrals\n"%(self.graph.count_dihedrals())
        if (len(self.unique_improper_types.keys()) > 0):
            string += "%12i impropers\n"%(self.graph.count_impropers())
    
        if(len(self.unique_atom_types.keys()) > 0):
            string += "\n%12i atom types\n"%(len(self.unique_atom_types.keys()))
        if(len(self.unique_bond_types.keys()) > 0):
            string += "%12i bond types\n"%(len(self.unique_bond_types.keys()))
        if(len(self.unique_angle_types.keys()) > 0):
            string += "%12i angle types\n"%(len(self.unique_angle_types.keys()))
        if(len(self.unique_dihedral_types.keys()) > 0):
            string += "%12i dihedral types\n"%(len(self.unique_dihedral_types.keys()))
        if (len(self.unique_improper_types.keys()) > 0):
            string += "%12i improper types\n"%(len(self.unique_improper_types.keys()))
    
        string += "%19.6f %10.6f %s %s\n"%(0., self.cell.lx, "xlo", "xhi")
        string += "%19.6f %10.6f %s %s\n"%(0., self.cell.ly, "ylo", "yhi")
        string += "%19.6f %10.6f %s %s\n"%(0., self.cell.lz, "zlo", "zhi")
        # currently the only reason to eliminate the skew from a simulation is
        # if pxrd is requested. Better make sure that the xy, xz, and yz are 0 or near 0!
        if self.options.pxrd:
            if not (np.allclose(np.array([self.cell.xy, self.cell.xz, self.cell.yz]), 0.0)):
                print("WARNING: the cell is not orthogonal! with xy = %8.4f, xz = %8.4f and yz = %8.4f"%(self.cell.xy,
                                            self.cell.xz, self.cell.yz))
                print("Making simulation input anyway, but proceed with caution!")
            string += "# %19.6f %10.6f %10.6f %s %s %s\n"%(self.cell.xy, self.cell.xz, self.cell.yz, "xy", "xz", "yz")
        else:
            string += "%19.6f %10.6f %10.6f %s %s %s\n"%(self.cell.xy, 
                                                         self.cell.xz, 
                                                         self.cell.yz, 
                                                         "xy", "xz", "yz")
    
        # Let's track the forcefield potentials that haven't been calc'd or user specified
        no_bond = []
        no_angle = []
        no_dihedral = []
        no_improper = []
        
        # this should be non-zero, but just in case..
        if(len(self.unique_atom_types.keys()) > 0):
            string += "\nMasses\n\n"
            for key in sorted(self.unique_atom_types.keys()):
                unq_atom = self.unique_atom_types[key][1] 
                mass, type = unq_atom['mass'], unq_atom['force_field_type']
                string += "%5i %15.9f # %s\n"%(key, mass, type)
    
        if(len(self.unique_bond_types.keys()) > 0):
            string += "\nBond Coeffs\n\n"
            for key in sorted(self.unique_bond_types.keys()):
                n1, n2, bond = self.unique_bond_types[key]
                atom1, atom2 = self.graph.node[n1], self.graph.node[n2]
                if bond['potential'] is None:
                    no_bond.append("%5i : %s %s"%(key, 
                                                  atom1['force_field_type'], 
                                                  atom2['force_field_type']))
                else:
                    ff1, ff2 = (atom1['force_field_type'], 
                                atom2['force_field_type'])
    
                    string += "%5i %s "%(key, bond['potential'])
                    string += "# %s %s\n"%(ff1, ff2)
    
        class2angle = False
        if(len(self.unique_angle_types.keys()) > 0):
            string += "\nAngle Coeffs\n\n"
            for key in sorted(self.unique_angle_types.keys()):
                a, b, c, angle = self.unique_angle_types[key]
                atom_a, atom_b, atom_c = self.graph.node[a], \
                                         self.graph.node[b], \
                                         self.graph.node[c] 
    
                if angle['potential'] is None:
                    no_angle.append("%5i : %s %s %s"%(key, 
                                          atom_a['force_field_type'], 
                                          atom_b['force_field_type'], 
                                          atom_c['force_field_type']))
                else:
                    if (angle['potential'].name == "class2"):
                        class2angle = True
    
                    string += "%5i %s "%(key, angle['potential'])
                    string += "# %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'])
    
        if(class2angle):
            string += "\nBondBond Coeffs\n\n"
            for key in sorted(self.unique_angle_types.keys()):
                a, b, c, angle = self.unique_angle_types[key]
                atom_a, atom_b, atom_c = self.graph.node[a], \
                                         self.graph.node[b], \
                                         self.graph.node[c]
                if (angle['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, angle['potential'].bb)
                        string += "# %s %s %s\n"%(atom_a['force_field_type'], 
                                                  atom_b['force_field_type'], 
                                                  atom_c['force_field_type'])
                    except AttributeError:
                        pass
        
            string += "\nBondAngle Coeffs\n\n"
            for key in sorted(self.unique_angle_types.keys()):
                a, b, c, angle = self.unique_angle_types[key]
                atom_a, atom_b, atom_c = self.graph.node[a],\
                                         self.graph.node[b],\
                                         self.graph.node[c]
                if (angle['potential'].name!="class2"):
                    string += "%5i skip  "%(key)
                    string += "# %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, angle['potential'].ba)
                        string += "# %s %s %s\n"%(atom_a['force_field_type'], 
                                                  atom_b['force_field_type'], 
                                                  atom_c['force_field_type'])
                    except AttributeError:
                        pass   
    
        class2dihed = False
        if(len(self.unique_dihedral_types.keys()) > 0):
            string +=  "\nDihedral Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if dihedral['potential'] is None:
                    no_dihedral.append("%5i : %s %s %s %s"%(key, 
                                       atom_a['force_field_type'], 
                                       atom_b['force_field_type'], 
                                       atom_c['force_field_type'], 
                                       atom_d['force_field_type']))
                else:
                    if(dihedral['potential'].name == "class2"):
                        class2dihed = True
                    string += "%5i %s "%(key, dihedral['potential'])
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                 atom_b['force_field_type'], 
                                                 atom_c['force_field_type'], 
                                                 atom_d['force_field_type'])
    
        if (class2dihed):
            string += "\nMiddleBondTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]

                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].mbt) 
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                  atom_b['force_field_type'], 
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nEndBondTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].ebt) 
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                  atom_b['force_field_type'], 
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nAngleTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].at) 
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                  atom_b['force_field_type'], 
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nAngleAngleTorsion Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].aat) 
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                  atom_b['force_field_type'], 
                                                  atom_c['force_field_type'],
                                                  atom_d['force_field_type'])
                    except AttributeError:
                        pass
            string += "\nBondBond13 Coeffs\n\n"
            for key in sorted(self.unique_dihedral_types.keys()):
                a, b, c, d, dihedral = self.unique_dihedral_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (dihedral['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                              atom_b['force_field_type'], 
                                              atom_c['force_field_type'],
                                              atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, dihedral['potential'].bb13) 
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                     atom_b['force_field_type'], 
                                                     atom_c['force_field_type'],
                                                     atom_d['force_field_type'])
                    except AttributeError:
                        pass
        
        
        class2improper = False 
        if (len(self.unique_improper_types.keys()) > 0):
            string += "\nImproper Coeffs\n\n"
            for key in sorted(self.unique_improper_types.keys()):
                a, b, c, d, improper = self.unique_improper_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]

                if improper['potential'] is None:
                    no_improper.append("%5i : %s %s %s %s"%(key, 
                        atom_a['force_field_type'], 
                        atom_b['force_field_type'], 
                        atom_c['force_field_type'], 
                        atom_d['force_field_type']))
                else:
                    if(improper['potential'].name == "class2"):
                        class2improper = True
                    string += "%5i %s "%(key, improper['potential'])
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                 atom_b['force_field_type'], 
                                                 atom_c['force_field_type'], 
                                                 atom_d['force_field_type'])
        if (class2improper):
            string += "\nAngleAngle Coeffs\n\n"
            for key in sorted(self.unique_improper_types.keys()):
                a, b, c, d, improper = self.unique_improper_types[key]
                atom_a, atom_b, atom_c, atom_d = self.graph.node[a], \
                                                 self.graph.node[b], \
                                                 self.graph.node[c], \
                                                 self.graph.node[d]
                if (improper['potential'].name!="class2"):
                    string += "%5i skip "%(key)
                    string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                 atom_b['force_field_type'], 
                                                 atom_c['force_field_type'], 
                                                 atom_d['force_field_type'])
                else:
                    try:
                        string += "%5i %s "%(key, improper['potential'].aa)
                        string += "# %s %s %s %s\n"%(atom_a['force_field_type'], 
                                                     atom_b['force_field_type'], 
                                                     atom_c['force_field_type'], 
                                                     atom_d['force_field_type'])
                    except AttributeError:
                        pass
    
        if((len(self.unique_pair_types.keys()) > 0) and (self.pair_in_data)):
            string += "\nPair Coeffs\n\n"
            for key, (n,pair) in sorted(self.unique_atom_types.items()):
                #pair = self.graph.node[n]
                string += "%5i %s "%(key, pair['pair_potential'])
                string += "# %s %s\n"%(pair['force_field_type'], 
                                       pair['force_field_type'])
       
 
        # Nest this in an if statement
        if any([no_bond, no_angle, no_dihedral, no_improper]):
        # WARNING MESSAGE for potentials we think are unique but have not been calculated
            print("WARNING: The following unique bonds/angles/dihedrals/impropers" +
                    " were detected in your crystal")
            print("But they have not been assigned a potential from user_input.txt"+
                    " or from an internal FF assignment routine!")
            print("Bonds")
            for elem in no_bond:
                print(elem)
            print("Angles")
            for elem in no_angle:
                print(elem)
            print("Dihedrals")
            for elem in no_dihedral:
                print(elem)
            print("Impropers")
            for elem in no_improper:
                print(elem)
            print("If you think you specified one of these in your user_input.txt " +
                  "and this is an error, please contact developers\n")
            print("CONTINUING...")
    
    
        #************[atoms]************
    	# Added 1 to all atom, bond, angle, dihedral, improper indices (LAMMPS does not accept atom of index 0)
        sorted_nodes = sorted(self.graph.nodes())
        if(len(self.unique_atom_types.keys()) > 0):
            string += "\nAtoms\n\n"
            for node in sorted_nodes:
                atom = self.graph.node[node]
                string += "%8i %8i %8i %11.5f %10.5f %10.5f %10.5f\n"%(node, 
                                                                       atom['molid'], 
                                                                       atom['ff_type_index'],
                                                                       atom['charge'],
                                                                       atom['cartesian_coordinates'][0], 
                                                                       atom['cartesian_coordinates'][1], 
                                                                       atom['cartesian_coordinates'][2])
    
        #************[bonds]************
        if(len(self.unique_bond_types.keys()) > 0):
            string += "\nBonds\n\n"
            idx = 0
            for n1, n2, bond in sorted(list(self.graph.edges_iter2(data=True))):
                idx += 1
                string += "%8i %8i %8i %8i\n"%(idx,
                                               bond['ff_type_index'], 
                                               n1, 
                                               n2)
    
        #************[angles]***********
        if(len(self.unique_angle_types.keys()) > 0):
            string += "\nAngles\n\n"
            idx = 0
            for node in sorted_nodes:
                atom = self.graph.node[node]
                try:
                    for (a, c), angle in list(atom['angles'].items()):
                        idx += 1
                        string += "%8i %8i %8i %8i %8i\n"%(idx,
                                                           angle['ff_type_index'], 
                                                           a, 
                                                           node,
                                                           c)
                except KeyError:
                    pass

        #************[dihedrals]********
        if(len(self.unique_dihedral_types.keys()) > 0):
            string += "\nDihedrals\n\n"
            idx = 0
            for n1, n2, data in sorted(list(self.graph.edges_iter2(data=True))):
                try:
                    for (a, d), dihedral in list(data['dihedrals'].items()):
                        idx+=1     
                        string += "%8i %8i %8i %8i %8i %8i\n"%(idx, 
                                                              dihedral['ff_type_index'], 
                                                              a, 
                                                              n1,
                                                              n2, 
                                                              d)
                except KeyError:
                    pass
        #************[impropers]********
        if(len(self.unique_improper_types.keys()) > 0):
            string += "\nImpropers\n\n"
            idx = 0
            for node in sorted_nodes:
                atom = self.graph.node[node]
                try:
                    for (a, c, d), improper in list(atom['impropers'].items()):
                        idx += 1
                        string += "%8i %8i %8i %8i %8i %8i\n"%(idx,
                                                               improper['ff_type_index'],
                                                               a, 
                                                               node,
                                                               c,
                                                               d)
                except KeyError:
                    pass
    
        return string
    def fixcount(self, count=[]):
        count.append(1)
        return (len(count))
    
    def computecount(self, count=[]):
        count.append(1)
        return (len(count))

    def construct_input_file(self):
        """Input file construction based on user-defined inputs.
        
        NB: This function is getting huge. We should probably break it 
        up into logical sub-sections.
        
        """
        # sanity check - right now if gcmc is true, insert_molecule must be
        # true. Currently inserts molecule templates only
        if self.options.gcmc:
            if not self.options.insert_molecule:
                print("ERROR: Cannot perform GCMC without a molecule template.")
                sys.exit(1)

        inp_str = ""
        
        inp_str += "%-15s %s\n"%("log","log.%s append"%(self.name))
        inp_str += "%-15s %s\n"%("units","real")
        inp_str += "%-15s %s\n"%("atom_style","full")
        inp_str += "%-15s %s\n"%("boundary","p p p")
        inp_str += "\n"
        if(len(self.unique_pair_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("pair_style", self.pair_style)
        if(len(self.unique_bond_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("bond_style", self.bond_style)
        if(len(self.unique_angle_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("angle_style", self.angle_style)
        if(len(self.unique_dihedral_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("dihedral_style", self.dihedral_style)
        if(len(self.unique_improper_types.keys()) > 0):
            inp_str += "%-15s %s\n"%("improper_style", self.improper_style)
        if(self.kspace_style): 
            inp_str += "%-15s %s\n"%("kspace_style", self.kspace_style) 
        inp_str += "\n"
    
        # general catch-all for extra force field commands needed.
        inp_str += "\n".join(list(set(self.special_commands)))
        inp_str += "\n"
        inp_str += "%-15s %s\n"%("box tilt","large")
        inp_str += "%-15s %s\n"%("read_data","data.%s"%(self.name))
   
        "compute chunk/atom molecule"

        if(not self.pair_in_data):
            inp_str += "#### Pair Coefficients ####\n"
            for pair,data in sorted(self.unique_pair_types.items()):
                n1, n2 = self.unique_atom_types[pair[0]][0], self.unique_atom_types[pair[1]][0]

                try:
                    if pair[2] == 'hb':
                        inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff", 
                            pair[0], pair[1], data['h_bond_potential'],
                            self.graph.node[n1]['force_field_type'],
                            self.graph.node[n2]['force_field_type'])
                    elif pair[2] == 'table':
                        inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff",
                            pair[0], pair[1], data['table_potential'],
                            self.graph.node[n1]['force_field_type'],
                            self.graph.node[n2]['force_field_type'])
                    else:
                        inp_str += "%-15s %-4i %-4i %s # %s %s\n"%("pair_coeff", 
                            pair[0], pair[1], data['pair_potential'],
                            self.graph.node[n1]['force_field_type'],
                            self.graph.node[n2]['force_field_type'])
                except IndexError:
                    pass
            inp_str += "#### END Pair Coefficients ####\n\n"
        
        inp_str += "\n#### Atom Groupings ####\n"
        # Define a group for the template molecules, if they exist.
        # It is conceptually hard to rationalize why this has to be
        # a separate command and not combined with the 'molecule' command
        if self.options.insert_molecule:
            moltypes = []
            for mnode, mdata in self.template_molecule.nodes_iter(data=True):
                moltypes.append(mdata['ff_type_index'])
            
            inp_str += "%-15s %s type   "%("group", self.options.insert_molecule)
            for x in self.groups(list(set(moltypes))):
                x = list(x)
                if (len(x) > 1):
                    inp_str += " %i:%i"%(x[0], x[-1])
                else:
                    inp_str += " %i"%(x[0])
            inp_str += "\n"
        
        # cell move is to flag how the box is able to move during the simulation
        cell_move="tri"
        if self.options.pxrd:
            cell_move="aniso"

        framework_atoms = self.graph.nodes()
        if(self.molecules)and(len(self.molecule_types.keys()) < 32):
            # lammps cannot handle more than 32 groups including 'all' 
            total_count = 0 
            for k,v in self.molecule_types.items():
                total_count += len(v)
            list_individual_molecules = True 
            if total_count > 31:
                list_individual_molecules = False

            idx = 1
            for mtype in list(self.molecule_types.keys()): 
                
                inp_str += "%-15s %-8s %s  "%("group", "%i"%(mtype), "id")
                all_atoms = []
                for j in self.molecule_types[mtype]:
                    all_atoms += self.subgraphs[j].nodes()
                for x in self.groups(all_atoms):
                    x = list(x)
                    if(len(x)>1):
                        inp_str += " %i:%i"%(x[0], x[-1])
                    else:
                        inp_str += " %i"%(x[0])
                inp_str += "\n"
                for atom in reversed(sorted(all_atoms)):
                    del framework_atoms[framework_atoms.index(atom)]
                mcount = 0
                if list_individual_molecules:
                    for j in self.molecule_types[mtype]:
                        if (self.subgraphs[j].molecule_images):
                            for molecule in self.subgraphs[j].molecule_images:
                                mcount += 1
                                inp_str += "%-15s %-8s %s  "%("group", "%i-%i"%(mtype, mcount), "id")
                                for x in self.groups(molecule):
                                    x = list(x)
                                    if(len(x)>1):
                                        inp_str += " %i:%i"%(x[0], x[-1])
                                    else:
                                        inp_str += " %i"%(x[0])
                                inp_str += "\n"
                        elif len(self.molecule_types[mtype]) > 1:
                            mcount += 1
                            inp_str += "%-15s %-8s %s  "%("group", "%i-%i"%(mtype, mcount), "id")
                            molecule = self.subgraphs[j].nodes()
                            for x in self.groups(molecule):
                                x = list(x)
                                if(len(x)>1):
                                    inp_str += " %i:%i"%(x[0], x[-1])
                                else:
                                    inp_str += " %i"%(x[0])
                            inp_str += "\n"

            if(not framework_atoms):
                self.framework = False
        if(self.framework):
            inp_str += "%-15s %-8s %s  "%("group", "fram", "id")
            for x in self.groups(framework_atoms):
                x = list(x)
                if(len(x)>1):
                    inp_str += " %i:%i"%(x[0], x[-1])
                else:
                    inp_str += " %i"%(x[0])
            inp_str += "\n"
        inp_str += "#### END Atom Groupings ####\n\n"
    
        if self.options.dump_dcd:
            inp_str += "%-15s %s\n"%("dump","%s_dcdmov all dcd %i %s_mov.dcd"%
                            (self.name, self.options.dump_dcd, self.name))
        elif self.options.dump_xyz:
            inp_str += "%-15s %s\n"%("dump","%s_xyzmov all xyz %i %s_mov.xyz"%
                                (self.name, self.options.dump_xyz, self.name))
            inp_str += "%-15s %s\n"%("dump_modify", "%s_xyzmov element %s"%(
                                     self.name, 
                                     " ".join([self.unique_atom_types[key][1]['element'] 
                                                for key in sorted(self.unique_atom_types.keys())])))
        elif self.options.dump_lammpstrj:
            inp_str += "%-15s %s\n"%("dump","%s_lammpstrj all atom %i %s_mov.lammpstrj"%
                                (self.name, self.options.dump_lammpstrj, self.name))

            # in the meantime we need to map atom id to element that will allow us to 
            # post-process the lammpstrj file and create a cif out of each 
            # snapshot stored in the trajectory
            f = open("lammpstrj_to_element.txt", "w")
            for key in sorted(self.unique_atom_types.keys()):
                f.write("%s\n"%(self.unique_atom_types[key][1]['element']))
            f.close()
            
        if (self.options.minimize):
            box_min = cell_move 
            min_style = "sd"
            #min_eval = 1e-6   # HKUST-1 will not minimize past 1e-11
            min_eval = 1e-3 
            max_iterations = 100000 # if the minimizer can't reach a minimum in this many steps,
                                    # change the min_eval to something higher.
            inp_str += "%-15s %s\n"%("min_style", min_style)
            inp_str += "%-15s %s\n"%("print", "\"MinStep,CellMinStep,AtomMinStep,FinalStep,Energy,EDiff\"" + 
                                              " file %s.min.csv screen no"%(self.name))
            inp_str += "%-15s %-10s %s\n"%("variable", "min_eval", "equal %.2e"%(min_eval))
            inp_str += "%-15s %-10s %s\n"%("variable", "prev_E", "equal %.2f"%(50000.)) # set unreasonably high for first loop
            inp_str += "%-15s %-10s %s\n"%("variable", "iter", "loop %i"%(max_iterations))
            inp_str += "%-15s %s\n"%("label", "loop_min")
            
            fix = self.fixcount() 
            inp_str += "%-15s %s\n"%("min_style", min_style)
            inp_str += "%-15s %s\n"%("fix","%i all box/relax %s 0.0 vmax 0.01"%(fix, box_min))
            inp_str += "%-15s %s\n"%("minimize","%.2e %.2e 10000 100000"%(min_eval**2, min_eval**2))
            inp_str += "%-15s %s\n"%("unfix", "%i"%fix)
            inp_str += "%-15s %s\n"%("min_style", "cg")
            inp_str += "%-15s %-10s %s\n"%("variable", "tempstp", "equal $(step)")
            inp_str += "%-15s %-10s %s\n"%("variable", "CellMinStep", "equal ${tempstp}")
            inp_str += "%-15s %s\n"%("minimize","%.2e %.2e 10000 100000"%(min_eval**2, min_eval**2))
            inp_str += "%-15s %-10s %s\n"%("variable", "AtomMinStep", "equal ${tempstp}")
            inp_str += "%-15s %-10s %s\n"%("variable", "temppe", "equal $(pe)")
            inp_str += "%-15s %-10s %s\n"%("variable", "min_E", "equal abs(${prev_E}-${temppe})")
            inp_str += "%-15s %s\n"%("print", "\"${iter},${CellMinStep},${AtomMinStep},${AtomMinStep}," + 
                                              "$(pe),${min_E}\"" +
                                              " append %s.min.csv screen no"%(self.name))

            inp_str += "%-15s %s\n"%("if","\"${min_E} < ${min_eval}\" then \"jump SELF break_min\"")
            inp_str += "%-15s %-10s %s\n"%("variable", "prev_E", "equal ${temppe}")
            inp_str += "%-15s %s\n"%("next", "iter")
            inp_str += "%-15s %s\n"%("jump", "SELF loop_min")
            inp_str += "%-15s %s\n"%("label", "break_min")

           # inp_str += "%-15s %s\n"%("unfix", "output")
        # this probably won't work if molecules are being inserted...
        if self.options.pxrd:
            # currently a copper source, but this can be changed later.
            pxrd_cid = self.computecount()
            atomic_str = " ".join([self.unique_atom_types[key][1]['element'] 
                                for key in sorted(self.unique_atom_types.keys())])
            # probably some more replacements here, but will change when they come up.
            # full list of valid atoms found here: http://lammps.sandia.gov/doc/compute_xrd.html
            atomic_str = atomic_str.replace("Zn", "Zn2+")
            atomic_str = atomic_str.replace("Al", "Al3+")
            inp_str += "%-15s %s\n"%("compute","%i all xrd 1.541838 %s 2Theta 2 40 c 1 1 1 LP 1"%(pxrd_cid, atomic_str))
            pxrd_id = self.fixcount()
            neqstps = self.options.neqstp
            if(self.options.npt):
                neqstps*=2
            npdstps = self.options.nprodstp
            low_angle = 2
            high_angle = 40
            xrdstep_size = 0.05
            nxrdsteps = int((high_angle - low_angle) / xrdstep_size)
            inp_str += "%-15s %s\n"%("fix","%i all ave/histo/weight 1 %i %i %i %i %i c_%i[1] c_%i[2] mode vector file %s.xrd"%(
                                            pxrd_id, npdstps, (npdstps + neqstps), low_angle, high_angle, nxrdsteps,  pxrd_cid, pxrd_cid, self.name))

        # delete bond types etc, for molecules that are rigid
        if self.options.insert_molecule:
            inp_str += "%-15s %s %s.molecule\n"%("molecule", self.options.insert_molecule, self.options.insert_molecule)
        
        for mol in sorted(self.molecule_types.keys()):
            rep = self.subgraphs[self.molecule_types[mol][0]]
            if rep.rigid:
                inp_str += "%-15s %s\n"%("neigh_modify", "exclude molecule %i"%(mol))
                # find and delete all bonds, angles, dihedrals, and impropers associated
                # with this molecule, as they will consume unnecessary amounts of CPU time
                inp_str += "%-15s %i %s\n"%("delete_bonds", mol, "multi remove")

        if (self.fix_shake):
            shake_tol = 0.0001
            iterations = 20
            print_every = 0  # maybe set to non-zero, but output files could become huge.
            shk_fix = self.fixcount()
            shake_str = "b "+" ".join(["%i"%i for i in self.fix_shake['bonds']]) + \
                        " a " + " ".join(["%i"%i for i in self.fix_shake['angles']])
                       # fix  id group tolerance iterations print_every [bonds + angles]
            inp_str += "%-15s %i %s %s %f %i %i %s\n"%('fix', shk_fix, 'all', 'shake', shake_tol, iterations, print_every, shake_str)

        if (self.options.random_vel):
            inp_str += "%-15s %s\n"%("velocity", "all create %.2f %i"%(self.options.temp, np.random.randint(1,3000000)))
        
        if (self.options.nvt) or (self.options.npt):
            if(self.options.nvt):
                siml='nvt'
            elif(self.options.npt):
                siml='npt'
            inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
            inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")
            molecule_fixes = []
            mollist = sorted(list(self.molecule_types.keys()))

            if (self.options.npt):
                inp_str += "%-15s %-10s %s\n"%("variable", "pdamp", "equal 1000*${dt}")
            # always start with nvt langevin
            if self.options.insert_molecule:
                id = self.fixcount()
                molecule_fixes.append(id)
                if(self.template_molecule.rigid):
                    if self.template_molecule.rigid_fix < 0:
                        id=self.fixcount()
                        molecule_fixes.append(id)
                        self.template_molecule.rigid_fix = id
                    inp_str += "%-15s %s\n"%("fix", "%i %s rigid/small molecule langevin %.2f %.2f ${tdamp} %i mol %s"%(id, 
                                                                                            self.options.insert_molecule,
                                                                                            self.options.temp, 
                                                                                            self.options.temp,
                                                                                            np.random.randint(1,3000000),
                                                                                            self.options.insert_molecule
                                                                                            ))
                else:
                    # no idea if this will work..
                    inp_str += "%-15s %s\n"%("fix", "%i %s langevin %.2f %.2f ${tdamp} %i"%(id, 
                                                                                        self.options.insert_molecule,
                                                                                        self.options.temp, 
                                                                                        self.options.temp,
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                    id = self.fixcount()
                    molecule_fixes.append(id)
                    inp_str += "%-15s %s\n"%("fix", "%i %i nve"%(id,molid))


            for molid in mollist: 
                id = self.fixcount()
                molecule_fixes.append(id)
                rep = self.subgraphs[self.molecule_types[molid][0]]
                if(rep.rigid):
                    inp_str += "%-15s %s\n"%("fix", "%i %s rigid/small molecule langevin %.2f %.2f ${tdamp} %i"%(id, 
                                                                                            str(molid), 
                                                                                            self.options.temp, 
                                                                                            self.options.temp,
                                                                                            np.random.randint(1,3000000)
                                                                                            ))
                else:
                    inp_str += "%-15s %s\n"%("fix", "%i %s langevin %.2f %.2f ${tdamp} %i"%(id, 
                                                                                        str(molid), 
                                                                                        self.options.temp, 
                                                                                        self.options.temp,
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                    id = self.fixcount()
                    molecule_fixes.append(id)
                    inp_str += "%-15s %s\n"%("fix", "%i %i nve"%(id,molid))

            gcmc_str = ""
            if (self.options.gcmc):
                gcmc_fix = self.fixcount()
                molecule_fixes.append(gcmc_fix)
                # chem. pot. is ignored when the 'pressure' keyword is set. We will default to this.
                mu = 0.5
                # N - invoke fix every N (MD) steps
                N = self.options.gcmc_every
                # X - average number of GCMC exchanges to attempt every N steps.
                X = self.options.gcmc_exch
                # M - average number of MC moves to attempt every N steps.
                M = self.options.gcmc_mc

                gcmc_str += "%-15s %i %s %s %i %i %i %i %i %.2f %.2f %.2f "%('fix', gcmc_fix, self.options.insert_molecule, 
                                         'gcmc', N, X, M, 0, np.random.randint(1,3000000), 
                                        self.options.temp, mu, self.options.gcmc_disp)
                gcmc_str += "%s %s "%('mol', self.options.insert_molecule)
                # splitting this string up, it's long
                gcmc_str += "%s %.2f "%('pressure',self.options.pressure)
                # fugacity coeff defaults to 1. but I'm putting it verbosely here
                # so that people will know it is a keyword.
                # The day I put an EOS in this code to compute this coefficient will
                # be the day I make long rueful regrets bout my life choices.
                gcmc_str += "%s %.2f "%('fugacity_coeff',1.0)
                gcmc_str += "%s %s "%('group', self.options.insert_molecule)

                if(self.template_molecule.rigid):
                    if self.template_molecule.rigid_fix < 0:
                        id=self.fixcount()
                        molecule_fixes.append(id)
                        self.template_molecule.rigid_fix = id
                    gcmc_str += "%s %i "%('rigid', self.template_molecule.rigid_fix)
               
                # currently shake is not associated with a molecule but with the 
                # potentials in the simulation. This should be modified..
                # but here I'm assuming the shake will be matched with the 
                # desired insertion molecule.
                if(self.fix_shake):
                    gcmc_str += "%s %i "%('shake', shk_fix)

                # its dissapointing that LAMMPS will compute the full energy each time a molecule
                # is inserted/deleted/moved in this fix (because of charge interactions)
                # ewald summations can be modified to permit partial energy calculations too...
                gcmc_str += "\n"
                inp_str += gcmc_str
                # needs fix_modify dynamic/dof yes for rigid/small/nvt or rigid/small/npt fix
            if self.framework:
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i %s langevin %.2f %.2f ${tdamp} %i"%(id, 
                                                                                        "fram", 
                                                                                        self.options.temp, 
                                                                                        self.options.temp,
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i fram nve"%id)

            # add a shift of the cell as the deposit of molecules tends to shift things.
            id = self.fixcount()
            inp_str += "%-15s %i all momentum 1 linear 1 1 1 angular\n"%("fix", id)
            # deposit within nvt equilibrium phase.  TODO(pboyd): This entire input file formation Needs to be re-thought.
            if self.options.deposit:
                deposit = self.options.deposit * np.prod(np.array(self.supercell)) 
                
                id = self.fixcount() 
                # define a region the size of the unit cell.
                every = self.options.neqstp/2/deposit
                if every <= 100:
                    print("WARNING: you have set %i equilibrium steps, which may not be enough to "%(self.options.neqstp) + 
                            "deposit %i %s molecules. "%(deposit, self.options.insert_molecule) +
                            "The metric used to create this warning is NEQSTP/2/DEPOSIT. So adjust accordingly.")
                inp_str += "%-15s %-8s %-8s %i %s %i %s %i %s %s\n"%("region", "cell", "block", 0, "EDGE", 
                                                                     0, "EDGE", 0, "EDGE", "units lattice")
                inp_str += "%-15s %i %s %s %i %i %i %i %s %s %s %.2f %s %s"%("fix", id, self.options.insert_molecule, 
                                                                             "deposit", deposit, 0, every, 
                                                                             np.random.randint(1, 3000000), "region", 
                                                                             "cell", "near", 2.0, "mol", 
                                                                             self.options.insert_molecule)
                molecule_fixes.append(id)
                # need rigid fixid
                if self.template_molecule.rigid:
                    inp_str += " rigid %i\n"%(self.template_molecule.rigid_fix)
                else:
                    inp_str += "\n"

            inp_str += "%-15s %i\n"%("thermo", 0)
            inp_str += "%-15s %i\n"%("run", self.options.neqstp)
            while(molecule_fixes):
                fid = molecule_fixes.pop(0)
                inp_str += "%-15s %i\n"%("unfix", fid)
            
            if self.options.insert_molecule:
                if self.template_molecule.rigid:
                    if self.template_molecule.rigid_fix < 0:
                        id=self.fixcount()
                        molecule_fixes.append(id)
                        self.template_molecule.rigid_fix = id
                    inp_str += "%-15s %s"%("fix", "%i %s rigid/%s/small molecule temp %.2f %.2f ${tdamp}"%(
                                                                                            self.template_molecule.rigid_fix, 
                                                                                            self.options.insert_molecule,
                                                                                            siml,
                                                                                            self.options.temp, 
                                                                                            self.options.temp))
                    if self.options.npt:
                        inp_str += " %s"%("%s %.2f %.2f ${pdamp}"%(cell_move, self.options.pressure, self.options.pressure))
                    inp_str += " mol %s\n"%(self.options.insert_molecule)
                else:
                    # no idea if this will work..
                    mol_nvt=self.fixcount()
                    inp_str += "%-15s %s"%("fix", "%i %s %s %.2f %.2f ${tdamp}"%(mol_nvt, 
                                                                                 self.options.insert_molecule,
                                                                                 siml,
                                                                                 self.options.temp, 
                                                                                 self.options.temp
                                                                                 ))
                    if self.options.npt:
                        inp_str += " %s"%("%s %.2f %.2f ${pdamp}"%(cell_move, self.options.pressure, self.options.pressure))
                    inp_str += "\n"
                if self.options.gcmc:
                    # recall the last gcmc_str
                    inp_str += gcmc_str
                    molecule_fixes.append(gcmc_fix)
                    if self.template_molecule.rigid:
                        inp_str += "%-15s %i %s %s\n"%('fix_modify', self.template_molecule.rigid_fix, 'dynamic/dof', 'yes')
                    else:
                        mdtemp_cid = self.computecount()
                        inp_str += "%-15s %i %s %s\n"%('compute', mdtemp_cid, 'all', 'temp')
                        inp_str += "%-15s %i %s %s\n"%('compute_modify', mdtemp_cid, 'dynamic/dof', 'yes')
                        inp_str += "%-15s %i %s %i\n"%('fix_modify', mol_nvt, 'temp', mdtemp_cid)

            for molid in mollist:
                id = self.fixcount()
                molecule_fixes.append(id)
                rep = self.subgraphs[self.molecule_types[molid][0]]
                if(rep.rigid):
                    inp_str += "%-15s %s"%("fix", "%i %s rigid/%s/small molecule temp %.2f %.2f ${tdamp}"%(id, 
                                                                                            str(molid),
                                                                                            siml,
                                                                                            self.options.temp, 
                                                                                            self.options.temp
                                                                                            ))
                    if self.options.npt:
                        inp_str += " %s"%("%s %.2f %.2f ${pdamp}"%(cell_move, self.options.pressure, self.options.pressure))
                    inp_str += "\n"
                else:
                    inp_str += "%-15s %s"%("fix", "%i %s %s temp %.2f %.2f ${tdamp}"%(id, 
                                                                                   str(molid),
                                                                                   siml,
                                                                                   self.options.temp, 
                                                                                   self.options.temp
                                                                                   ))
                    if self.options.npt:
                        inp_str += " %s"%("%s %.2f %.2f ${pdamp}"%(cell_move, self.options.pressure, self.options.pressure))
                    inp_str += "\n"

            if self.framework:
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s"%("fix", "%i %s %s temp %.2f %.2f ${tdamp}"%(id, 
                                                                                   "fram",
                                                                                   siml,
                                                                                   self.options.temp, 
                                                                                   self.options.temp
                                                                                   ))
                if self.options.npt:
                    inp_str += " %s"%("%s %.2f %.2f ${pdamp}"%(cell_move, self.options.pressure, self.options.pressure))
                inp_str += "\n"
            
            # do another round of equilibration if npt is requested.
            if (self.options.npt):
                inp_str += "%-15s %i\n"%("run", self.options.neqstp)

            inp_str += "%-15s %i\n"%("thermo", 1)
            inp_str += "%-15s %i\n"%("run", self.options.nprodstp)
            
            while(molecule_fixes):
                fid = molecule_fixes.pop(0)
                inp_str += "%-15s %i\n"%("unfix", fid)


        if(self.options.bulk_moduli):
            min_style=True
            thermo_style=False

            inp_str += "\n%-15s %s\n"%("dump", "str all atom 1 initial_structure.dump")
            inp_str += "%-15s\n"%("run 0")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "equal step")
            inp_str += "%-15s %-10s %s\n"%("variable", "readstep", "equal ${rs}")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "delete")
            inp_str += "%-15s %s\n"%("undump", "str")
            
            if thermo_style:
                inp_str += "\n%-15s %-10s %s\n"%("variable", "simTemp", "equal %.4f"%(self.options.temp))
                inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
                inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")
            elif min_style:
                inp_str += "%-15s %s\n"%("min_style","fire")
            inp_str += "%-15s %-10s %s\n"%("variable", "at", "equal cella")
            inp_str += "%-15s %-10s %s\n"%("variable", "bt", "equal cellb")
            inp_str += "%-15s %-10s %s\n"%("variable", "ct", "equal cellc")
            inp_str += "%-15s %-10s %s\n"%("variable", "a", "equal ${at}")
            inp_str += "%-15s %-10s %s\n"%("variable", "b", "equal ${bt}")
            inp_str += "%-15s %-10s %s\n"%("variable", "c", "equal ${ct}")
            inp_str += "%-15s %-10s %s\n"%("variable", "at", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "bt", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "ct", "delete")
            
            inp_str += "%-15s %-10s %s\n"%("variable", "N", "equal %i"%self.options.iter_count)
            inp_str += "%-15s %-10s %s\n"%("variable", "totDev", "equal %.5f"%self.options.max_dev)
            inp_str += "%-15s %-10s %s\n"%("variable", "sf", "equal ${totDev}/${N}*2")
            inp_str += "%-15s %s\n"%("print", "\"Loop,CellScale,Vol,Pressure,E_total,E_pot,E_kin" + 
                                              ",E_bond,E_angle,E_torsion,E_imp,E_vdw,E_coul\""+
                                              " file %s.output.csv screen no"%(self.name))
            inp_str += "%-15s %-10s %s\n"%("variable", "do", "loop ${N}")
            inp_str += "%-15s %s\n"%("label", "loop_bulk")
            inp_str += "%-15s %s\n"%("read_dump", "initial_structure.dump ${readstep} x y z box yes format native")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleVar", "equal 1.00-${totDev}+${do}*${sf}")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleA", "equal ${scaleVar}*${a}")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleB", "equal ${scaleVar}*${b}")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleC", "equal ${scaleVar}*${c}")
            inp_str += "%-15s %s\n"%("change_box", "all x final 0.0 ${scaleA} y final 0.0 ${scaleB} z final 0.0 ${scaleC} remap")
            if (min_style):
                inp_str += "%-15s %s\n"%("minimize", "1.0e-15 1.0e-15 10000 100000")
                inp_str += "%-15s %s\n"%("print", "\"${do},${scaleVar},$(vol),$(press),$(etotal),$(pe),$(ke)"+
                                              ",$(ebond),$(eangle),$(edihed),$(eimp),$(evdwl),$(ecoul)\""+
                                              " append %s.output.csv screen no"%(self.name))
            elif (thermo_style):
                inp_str += "%-15s %s\n"%("velocity", "all create ${simTemp} %i"%(np.random.randint(1,3000000)))
                inp_str += "%-15s %s %s %s \n"%("fix", "bm", "all nvt", "temp ${simTemp} ${simTemp} ${tdamp} tchain 5")
                inp_str += "%-15s %i\n"%("run", self.options.neqstp)
                #inp_str += "%-15s %s\n"%("print", "\"STEP ${do} ${scaleVar} $(vol) $(press) $(etotal)\"")
                inp_str += "%-15s %s %s\n"%("fix", "output all print 10", "\"${do},${scaleVar},$(vol),$(press),$(etotal),$(pe),$(ke)" +
                                            ",$(ebond),$(eangle),$(edihed),$(eimp),$(evdwl),$(ecoul)\""+
                                            " append %s.output.csv screen no"%(self.name))
                inp_str += "%-15s %i\n"%("run", self.options.nprodstp)
                inp_str += "%-15s %s\n"%("unfix", "output")
                inp_str += "%-15s %s\n"%("unfix", "bm")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleVar", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleA", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleB", "delete")
            inp_str += "%-15s %-10s %s\n"%("variable", "scaleC", "delete")
            inp_str += "%-15s %s\n"%("next", "do")
            inp_str += "%-15s %s\n"%("jump", "SELF loop_bulk")
            inp_str += "%-15s %-10s %s\n"%("variable", "do", "delete")

        if (self.options.thermal_scaling):
            temperature = self.options.temp # kelvin
            equil_steps = self.options.neqstp 
            prod_steps = self.options.nprodstp 
            temprange = np.linspace(temperature, self.options.max_dev, self.options.iter_count).tolist()
            temprange.append(298.0)
            temprange.insert(0,1.0) # add 1 and 298 K simulations.
            
            inp_str += "\n%-15s %s\n"%("dump", "str all atom 1 initial_structure.dump")
            inp_str += "%-15s\n"%("run 0")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "equal step")
            inp_str += "%-15s %-10s %s\n"%("variable", "readstep", "equal ${rs}")
            inp_str += "%-15s %-10s %s\n"%("variable", "rs", "delete")
            inp_str += "%-15s %s\n"%("undump", "str")

            inp_str += "%-15s %-10s %s\n"%("variable", "sim_temp", "index %s"%(" ".join(["%.2f"%i for i in sorted(temprange)])))
            inp_str += "%-15s %-10s %s\n"%("variable", "sim_press", "equal %.3f"%self.options.pressure) # atmospheres.
            #inp_str += "%-15s %-10s %s\n"%("variable", "a", "equal cella")
            #inp_str += "%-15s %-10s %s\n"%("variable", "myVol", "equal vol")
            #inp_str += "%-15s %-10s %s\n"%("variable", "t", "equal temp")
            # timestep in femtoseconds
            inp_str += "%-15s %-10s %s\n"%("variable", "dt", "equal %.2f"%(1.0))
            inp_str += "%-15s %-10s %s\n"%("variable", "pdamp", "equal 1000*${dt}")
            inp_str += "%-15s %-10s %s\n"%("variable", "tdamp", "equal 100*${dt}")
            inp_str += "%-15s %s\n"%("print", "\"Step,Temp,CellA,Vol\" file %s.output.csv screen no"%(self.name))
            inp_str += "%-15s %s\n"%("label", "loop_thermal")
            #fix1 = self.fixcount()

            inp_str += "%-15s %s\n"%("read_dump", "initial_structure.dump ${readstep} x y z box yes format native")
            inp_str += "%-15s %s\n"%("thermo_style", "custom step temp cella cellb cellc vol etotal")
            
            # the ave/time fix must be after read_dump, or the averages are reported as '0'
            #inp_str += "%-15s %s\n"%("fix", "%i all ave/time 1 %i %i v_t v_a v_myVol ave one"%(fix1, prod_steps,
            #                                                                                   prod_steps + equil_steps))
            molecule_fixes = []
            mollist = sorted(list(self.molecule_types.keys()))
            for molid in mollist: 
                id = self.fixcount()
                molecule_fixes.append(id)
                rep = self.subgraphs[self.molecule_types[molid][0]]
                if(rep.rigid):
                    inp_str += "%-15s %s\n"%("fix", "%i %s rigid/small molecule langevin ${sim_temp} ${sim_temp} ${tdamp} %i"%(id, 
                                                                                            str(molid), 
                                                                                            np.random.randint(1,3000000)
                                                                                            ))
                else:
                    inp_str += "%-15s %s\n"%("fix", "%i %s langevin ${sim_temp} ${sim_temp} ${tdamp} %i"%(id, 
                                                                                        str(molid), 
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                    id = self.fixcount()
                    molecule_fixes.append(id)
                    inp_str += "%-15s %s\n"%("fix", "%i %i nve"%(id,molid))
            if self.framework:
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i %s langevin ${sim_temp} ${sim_temp} ${tdamp} %i"%(id, 
                                                                                        "fram", 
                                                                                        np.random.randint(1,3000000)
                                                                                        ))
                id = self.fixcount()
                molecule_fixes.append(id)
                inp_str += "%-15s %s\n"%("fix", "%i fram nve"%id)
            inp_str += "%-15s %i\n"%("thermo", 0)
            inp_str += "%-15s %i\n"%("run", equil_steps)
            while(molecule_fixes):
                fid = molecule_fixes.pop(0)
                inp_str += "%-15s %i\n"%("unfix", fid)
            id = self.fixcount() 
            # creating velocity may cause instability at high temperatures.
            #inp_str += "%-15s %s\n"%("velocity", "all create 50 %i"%(np.random.randint(1,3000000)))
            inp_str += "%-15s %i %s %s %s %s\n"%("fix", id,
                                        "all npt",
                                        "temp ${sim_temp} ${sim_temp} ${tdamp}",
                                        "tri ${sim_press} ${sim_press} ${pdamp}",
                                        "tchain 5 pchain 5")
            inp_str += "%-15s %i\n"%("thermo", 0)
            inp_str += "%-15s %i\n"%("run", equil_steps)
            inp_str += "%-15s %s %s\n"%("fix", "output all print 10", "\"${sim_temp},$(temp),$(cella),$(vol)\"" +
                                        " append %s.output.csv screen no"%(self.name))
            #inp_str += "%-15s %i\n"%("thermo", 10)
            inp_str += "%-15s %i\n"%("run", prod_steps)
            inp_str += "%-15s %s\n"%("unfix", "output")
            #inp_str += "\n%-15s %-10s %s\n"%("variable", "inst_t", "equal f_%i[1]"%(fix1))
            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_a", "equal f_%i[2]"%(fix1))
            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_v", "equal f_%i[3]"%(fix1))

            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_t", "delete")
            #inp_str += "%-15s %-10s %s\n"%("variable", "inst_a", "delete")
            #inp_str += "%-15s %-10s %s\n\n"%("variable", "inst_v", "delete")
            inp_str += "%-15s %i\n"%("unfix", id) 
            #inp_str += "%-15s %i\n"%("unfix", fix1)
            inp_str += "\n%-15s %s\n"%("next", "sim_temp")
            inp_str += "%-15s %s\n"%("jump", "SELF loop_thermal")
            inp_str += "%-15s %-10s %s\n"%("variable", "sim_temp", "delete")
        
        if self.options.pxrd:
            inp_str += "%-15s %i\n"%("unfix", pxrd_id) 
        if self.options.dump_dcd: 
            inp_str += "%-15s %s\n"%("undump", "%s_dcdmov"%(self.name))
        elif self.options.dump_xyz:
            inp_str += "%-15s %s\n"%("undump", "%s_xyzmov"%(self.name))
        elif self.options.dump_lammpstrj:
            inp_str += "%-15s %s\n"%("undump", "%s_lammpstrj"%(self.name))

        if self.options.restart:
            # for restart files we move xlo, ylo, zlo back to 0 so to have same origin as a cif file
            # also we modify to have unscaled coords so we can directly compute scaled coordinates WITH CIF BASIS
            inp_str += "\n# Dump last snapshot for restart\n"

            inp_str += "variable curr_lx equal lx\n"
            inp_str += "variable curr_ly equal ly\n"
            inp_str += "variable curr_lz equal lz\n"
            inp_str += "change_box all x final 0 ${curr_lx} y final 0 ${curr_ly} z final 0 ${curr_lz}\n\n"
            inp_str += "reset_timestep 0\n"
            inp_str += "%-15s %s\n"%("dump","%s_restart all atom 1 %s_restart.lammpstrj"%
                            (self.name, self.name))
            inp_str += "%-15s %s_restart scale no sort id\n"%("dump_modify",self.name)
            inp_str += "run 0\n"
            inp_str += "%-15s %s\n"%("undump", "%s_restart"%(self.name))

            # write a string that tells you how to read the dump file for this structure
            f=open("dump_restart_string.txt","w")
            f.write("read_dump %s_restart.lammpstrj %d x y z box yes"%(self.name, 
                                                                       0))
            f.close()
        
        try:
            inp_str += "%-15s %i\n"%("unfix", shk_fix)
        except NameError:
            # no shake fix id in this input file.
            pass
        return inp_str
    
    def groups(self, ints):
        ints = sorted(ints)
        for k, g in itertools.groupby(enumerate(ints), lambda ix : ix[0]-ix[1]):
            yield list(map(operator.itemgetter(1), g))

    # this needs to be somewhere else.
    def compute_molecules(self, size_cutoff=0.5):
        """Ascertain if there are molecules within the porous structure"""
        for j in nx.connected_components(self.graph):
            # return a list of nodes of connected graphs (decisions to isolate them will come later)
            # Upper limit on molecule size is 100 atoms.
            if((len(j) <= self.graph.original_size*size_cutoff) or (len(j) < 25)) and (not len(j) > 100) :
                self.molecules.append(j)
    
    def cut_molecule(self, nodes):
        mgraph = self.graph.subgraph(nodes).copy()
        self.graph.remove_nodes_from(nodes)
        indices = np.array(list(nodes)) 
        indices -= 1
        mgraph.coordinates = self.graph.coordinates[indices,:].copy()
        mgraph.sorted_edge_dict = self.graph.sorted_edge_dict.copy()
        mgraph.distance_matrix = self.graph.distance_matrix.copy()
        mgraph.original_size = self.graph.original_size
        for n1, n2 in mgraph.edges_iter():
            try:
                val = self.graph.sorted_edge_dict.pop((n1, n2))
                mgraph.sorted_edge_dict.update({(n1, n2):val})
            except KeyError:
                print("something went wrong")
            try:
                val = self.graph.sorted_edge_dict.pop((n2, n1))
                mgraph.sorted_edge_dict.update({(n2,n1):val})
            except KeyError:
                print("something went wrong")
        return mgraph

###############################################################################
# START Pymatgen custom functions that are necessary to implement our algorithm
# (usually just a lot of work to make sure Pymatgen preserves the site ordering)
###############################################################################

def custom_pymatgen_get_primitive_structure(structure, tolerance=0.25):
    """
    This finds a smaller unit cell than the input. Sometimes it doesn"t
    find the smallest possible one, so this method is recursively called
    until it is unable to find a smaller cell.

    NOTE: if the tolerance is greater than 1/2 the minimum inter-site
    distance in the primitive cell, the algorithm will reject this lattice.

    Args:
        tolerance (float), Angstroms: Tolerance for each coordinate of a
            particular site. For example, [0.1, 0, 0.1] in cartesian
            coordinates will be considered to be on the same coordinates
            as [0, 0, 0] for a tolerance of 0.25. Defaults to 0.25.

    Returns:
        The most primitive structure found.
    """

    print("\nPerforming custom primitve structure search w/o scrambling atom order")

    # group sites by species string
    sites = sorted(structure._sites, key=lambda s: s.species_string)
    #sites = structure._sites
    grouped_sites = [
        list(a[1])
        for a in itertools.groupby(sites, key=lambda s: s.species_string)]
    grouped_fcoords = [np.array([s.frac_coords for s in g])
                       for g in grouped_sites]

    # min_vecs are approximate periodicities of the cell. The exact
    # periodicities from the supercell matrices are checked against these
    # first
    min_fcoords = min(grouped_fcoords, key=lambda x: len(x))
    min_vecs = min_fcoords - min_fcoords[0]

    # fractional tolerance in the supercell
    super_ftol = np.divide(tolerance, structure.lattice.abc)
    super_ftol_2 = super_ftol * 2

    def pbc_coord_intersection(fc1, fc2, tol):
        """
        Returns the fractional coords in fc1 that have coordinates
        within tolerance to some coordinate in fc2
        """
        d = fc1[:, None, :] - fc2[None, :, :]
        d -= np.round(d)
        np.abs(d, d)
        return fc1[np.any(np.all(d < tol, axis=-1), axis=-1)]

    # here we reduce the number of min_vecs by enforcing that every
    # vector in min_vecs approximately maps each site onto a similar site.
    # The subsequent processing is O(fu^3 * min_vecs) = O(n^4) if we do no
    # reduction.
    # This reduction is O(n^3) so usually is an improvement. Using double
    # the tolerance because both vectors are approximate
    for g in sorted(grouped_fcoords, key=lambda x: len(x)):
        for f in g:
            min_vecs = pbc_coord_intersection(min_vecs, g - f, super_ftol_2)

    def get_hnf(fu):
        """
        Returns all possible distinct supercell matrices given a
        number of formula units in the supercell. Batches the matrices
        by the values in the diagonal (for less numpy overhead).
        Computational complexity is O(n^3), and difficult to improve.
        Might be able to do something smart with checking combinations of a
        and b first, though unlikely to reduce to O(n^2).
        """

        def factors(n):
            for i in range(1, n + 1):
                if n % i == 0:
                        yield i

            for det in factors(fu):
                if det == 1:
                    continue
                for a in factors(det):
                    for e in factors(det // a):
                        g = det // a // e
                        yield det, np.array(
                            [[[a, b, c], [0, e, f], [0, 0, g]]
                             for b, c, f in
                             itertools.product(range(a), range(a),
                                               range(e))])

        # we cant let sites match to their neighbors in the supercell
        grouped_non_nbrs = []
        for gfcoords in grouped_fcoords:
            fdist = gfcoords[None, :, :] - gfcoords[:, None, :]
            fdist -= np.round(fdist)
            np.abs(fdist, fdist)
            non_nbrs = np.any(fdist > 2 * super_ftol[None, None, :], axis=-1)
            # since we want sites to match to themselves
            np.fill_diagonal(non_nbrs, True)
            grouped_non_nbrs.append(non_nbrs)

        num_fu = six.moves.reduce(gcd, map(len, grouped_sites))
        for size, ms in get_hnf(num_fu):
            inv_ms = np.linalg.inv(ms)

            # find sets of lattice vectors that are are present in min_vecs
            dist = inv_ms[:, :, None, :] - min_vecs[None, None, :, :]
            dist -= np.round(dist)
            np.abs(dist, dist)
            is_close = np.all(dist < super_ftol, axis=-1)
            any_close = np.any(is_close, axis=-1)
            inds = np.all(any_close, axis=-1)

            for inv_m, m in zip(inv_ms[inds], ms[inds]):
                new_m = np.dot(inv_m, structure.lattice.matrix)
                ftol = np.divide(tolerance, np.sqrt(np.sum(new_m ** 2, axis=1)))

                valid = True
                new_coords = []
                new_sp = []
                new_props = collections.defaultdict(list)
                for gsites, gfcoords, non_nbrs in zip(grouped_sites,
                                                      grouped_fcoords,
                                                      grouped_non_nbrs):
                    all_frac = np.dot(gfcoords, m)

                    # calculate grouping of equivalent sites, represented by
                    # adjacency matrix
                    fdist = all_frac[None, :, :] - all_frac[:, None, :]
                    fdist = np.abs(fdist - np.round(fdist))
                    close_in_prim = np.all(fdist < ftol[None, None, :], axis=-1)
                    groups = np.logical_and(close_in_prim, non_nbrs)

                    # check that groups are correct
                    if not np.all(np.sum(groups, axis=0) == size):
                        valid = False
                        break

                    # check that groups are all cliques
                    for g in groups:
                        if not np.all(groups[g][:, g]):
                            valid = False
                            break
                    if not valid:
                        break

                    # add the new sites, averaging positions
                    added = np.zeros(len(gsites))
                    new_fcoords = all_frac % 1
                    for i, group in enumerate(groups):
                        if not added[i]:
                            added[group] = True
                            inds = np.where(group)[0]
                            coords = new_fcoords[inds[0]]
                            for n, j in enumerate(inds[1:]):
                                offset = new_fcoords[j] - coords
                                coords += (offset - np.round(offset)) / (n + 2)
                            new_sp.append(gsites[inds[0]].species_and_occu)
                            for k in gsites[inds[0]].properties:
                                new_props[k].append(gsites[inds[0]].properties[k])
                            new_coords.append(coords)

                if valid:
                    inv_m = np.linalg.inv(m)
                    new_l = Lattice(np.dot(inv_m, structure.lattice.matrix))
                    s = Structure(new_l, new_sp, new_coords,
                                  site_properties=new_props,
                                  coords_are_cartesian=False)
                    # NOTE was this:
                    return s.get_primitive_structure(
                        tolerance).get_reduced_structure()
                    #return custom_pymatgen_get_primitive_structure(s, 
                    #    tolerance).get_reduced_structure()

    return structure.copy()




def custom_pymatgen_slab_copy(slab,site_properties=None,sanitize=False):
    """
    Convenience method to get a copy of the structure, with options to add
    site properties.

    Args:
        site_properties (dict): Properties to add or override. The
            properties are specified in the same way as the constructor,
            i.e., as a dict of the form {property: [values]}. The
            properties should be in the order of the *original* structure
            if you are performing sanitization.
        sanitize (bool): If True, this method will return a sanitized
            structure. Sanitization performs a few things: (i) The sites are
            sorted by electronegativity, (ii) a LLL lattice reduction is
            carried out to obtain a relatively orthogonalized cell,
            (iii) all fractional coords for sites are mapped into the
            unit cell.

    Returns:
        A copy of the Structure, with optionally new site_properties and
        optionally sanitized.
    """
    if (not site_properties) and (not sanitize):
        # This is not really a good solution, but if we are not changing
        # the site_properties or sanitizing, initializing an empty
        # structure and setting _sites to be sites is much faster (~100x)
        # than doing the full initialization.
        s_copy = slab.__class__(lattice=self._lattice, species=[],
                                coords=[])
        s_copy._sites = list(slab._sites)
        return s_copy
    props = slab.site_properties
    if site_properties:
        props.update(site_properties)
    if not sanitize:
        return slab.__class__(slab._lattice,
                              slab.species_and_occu,
                              slab.frac_coords,
                              site_properties=props)
    else:
        reduced_latt = slab._lattice.get_lll_reduced_lattice()
        new_sites = []
        for i, site in enumerate(slab):
            frac_coords = reduced_latt.get_fractional_coords(site.coords)
            site_props = {}
            for p in props:
                site_props[p] = props[p][i]
            new_sites.append(PeriodicSite(site.species_and_occu,
                                          frac_coords, reduced_latt,
                                          to_unit_cell=True,
                                          properties=site_props))
        #new_sites = sorted(new_sites)
        return slab.__class__.from_sites(new_sites)


def custom_pymatgen_get_slab(this_slabgen, shift=0, tol=0.1, energy=None):
    """
    This method takes in shift value for the c lattice direction and
    generates a slab based on the given shift. You should rarely use this
    method. Instead, it is used by other generation algorithms to obtain
    all slabs.

    Arg:
        shift (float): A shift value in Angstrom that determines how much a
            slab should be shifted.
        tol (float): Tolerance to determine primitive cell.
        energy (float): An energy to assign to the slab.

    Returns:
        (Slab) A Slab object with a particular shifted oriented unit cell.
    """

    h = this_slabgen._proj_height
    nlayers_slab = int(math.ceil(this_slabgen.min_slab_size / h))
    nlayers_vac = int(math.ceil(this_slabgen.min_vac_size / h))
    nlayers = nlayers_slab + nlayers_vac

    species = this_slabgen.oriented_unit_cell.species_and_occu
    props = this_slabgen.oriented_unit_cell.site_properties
    props = {k: v * nlayers_slab for k, v in props.items()}
    frac_coords = this_slabgen.oriented_unit_cell.frac_coords
    frac_coords = np.array(frac_coords) +\
                  np.array([0, 0, -shift])[None, :]
    frac_coords -= np.floor(frac_coords)
    a, b, c = this_slabgen.oriented_unit_cell.lattice.matrix
    new_lattice = [a, b, nlayers * c]
    frac_coords[:, 2] = frac_coords[:, 2] / nlayers
    all_coords = []
    for i in range(nlayers_slab):
        fcoords = frac_coords.copy()
        fcoords[:, 2] += i / nlayers
        all_coords.extend(fcoords)

    slab = Structure(new_lattice, species * nlayers_slab, all_coords,
                     site_properties=props)

    scale_factor = this_slabgen.slab_scale_factor
    # Whether or not to orthogonalize the structure
    if this_slabgen.lll_reduce:
        # NOTE old command: lll_slab = slab.copy(sanitize=True)
        # NOTE modified command to prevent resorting
        lll_slab = custom_pymatgen_slab_copy(slab, sanitize=True)
        mapping = lll_slab.lattice.find_mapping(slab.lattice)
        scale_factor = np.dot(mapping[2], scale_factor)
        slab = lll_slab

    # Whether or not to center the slab layer around the vacuum
    if this_slabgen.center_slab:
        avg_c = np.average([c[2] for c in slab.frac_coords])
        slab.translate_sites(list(range(len(slab))), [0, 0, 0.5 - avg_c])

    # NOTE when we ask for a primitive cell Pymatgen is going to reorder the atoms
    if this_slabgen.primitive:
        # NOTE old command
        #prim = slab.get_primitive_structure(tolerance=tol)
        prim = custom_pymatgen_get_primitive_structure(slab, tolerance=tol)
        #print(prim)
        if energy is not None:
            energy = prim.volume / slab.volume * energy
        slab = prim

    return Slab(slab.lattice, slab.species_and_occu,
                slab.frac_coords, this_slabgen.miller_index,
                this_slabgen.oriented_unit_cell, shift,
                    scale_factor, site_properties=slab.site_properties,
                    energy=energy)

def custom_pymatgen_get_symmetrically_distinct_miller_indices(structure, max_index):
    """
    Returns all symmetrically distinct indices below a certain max-index for
    a given structure. Analysis is based on the symmetry of the reciprocal
    lattice of the structure.
    Args:
        structure (Structure): input structure.
        max_index (int): The maximum index. For example, a max_index of 1
            means that (100), (110), and (111) are returned for the cubic
            structure. All other indices are equivalent to one of these.
    """

    print("\nGrouping symmetrically identical Miller faces")

    symm_ops = pymatgen.core.surface.get_recp_symmetry_operation(structure)
    unique_millers = []
    equiv_millers = []
    equiv_millers_dict = {}

    def is_already_analyzed(miller_index):
        for op in symm_ops:
            #if in_coord_list(unique_millers, op.operate(miller_index)):
            #    return True

            # Pymatgen's find_in_coord_list
            if len(unique_millers) == 0:
                return -1
            this_op=op.operate(miller_index)
            diff = np.array(unique_millers) - np.array(this_op)[None, :]
            return_val = np.where(np.all(np.abs(diff) < 1e-8, axis=1))[0]

            if(len(return_val) > 0):
                #print("Miller index: %s"%str(miller_index))
                #print("Symm Miller index: %s"%str(this_op))
                #print("Equivalent index: %s"%str(return_val))
                return return_val[0]
        return -1

    r = list(range(-max_index, max_index + 1))
    r.reverse()
    for miller in itertools.product(r, r, r):
        if any([i != 0 for i in miller]):
            d = abs(reduce(gcd, miller))
            miller = tuple([int(i / d) for i in miller])
            equiv = is_already_analyzed(miller)
            if(equiv==-1):
                unique_millers.append(miller)
                equiv_millers.append([miller,[]])
            else:
                if(miller not in equiv_millers[equiv][1] and miller != equiv_millers[equiv][0]):
                    equiv_millers[equiv][1].append(miller)

    for elem in equiv_millers:
        for equiv in elem[1]:
            equiv_millers_dict[equiv]=elem[0]

    return unique_millers, equiv_millers, equiv_millers_dict

def get_nlayers(this_slabgen):
    """
    Helper function for pymatgen slab generation

    this_slabgen: SlabGenerator object
    """

    h=this_slabgen._proj_height                                               
    nlayers_slab = int(math.ceil(this_slabgen.min_slab_size / h))               
    nlayers_vac  = int(math.ceil(this_slabgen.min_vac_size / h))                
    nlayers = nlayers_slab + nlayers_vac                                        
                                                                                
    return [nlayers_slab, nlayers_vac, nlayers] 

def num2alpha(n):                                                               
    """
    Helper function for pymatgen slab generation

    turns integer into excel column
    """
                                                                                
    div=n                                                                       
    string=""                                                                   
    temp=0                                                                      
    while div>0:                                                                
        module=(div-1)%26                                                       
        string=chr(97+module)+string                                            
        div=int((div-module)/26)                                                
                                                                                
    return string 

def unique_typing(structure,dataset,debug=False):                               
    """
    Helper function for pymatgen slab generation

    structure: any pymatgen Structure or subclass
    """
                                                                                
    print(dataset.keys())                                                       
    print(structure.sites[0].species_string)                                    
                                                                                
                                                                                
    # map from unique type to the int in dataset['equivalent_atoms']            
    unique_descr_to_int = {}                                                    
                                                                                
    # all site numbers for a given unique descr                                 
    unique_descr_to_sites={}                                                    
                                                                                
    # how many sites exist of the unique type in key                            
    unique_descr_count = {}                                                     
                                                                                
    # how many uniques have been discovered per element, { Si: 2, O: 1, Al: 1}  
    unique_element_counter={}                                                   
                                                                                
    descr=[]                                                                    
                                                                                
    for i in range(len(dataset['equivalent_atoms'])):                           
        # already discovered                                                    
        if dataset['equivalent_atoms'][i] in unique_descr_to_int.keys():        
            descr.append(unique_descr_to_int[dataset['equivalent_atoms'][i]])   
            unique_descr_count[unique_descr_to_int[dataset['equivalent_atoms'][i]]] += 1
            unique_descr_to_sites[unique_descr_to_int[dataset['equivalent_atoms'][i]]].append(i)
        # not discovered                                                        
        else:                                                                   
            element=structure.sites[i].species_string                           
                                                                                
            if(element in unique_element_counter.keys()):                       
                                                                                
                # we now have yet another unique type for this element          
                unique_element_counter[element] += 1                            
                alpha_str=num2alpha(unique_element_counter[element])            
                this_descr=element+'_'+alpha_str                                
                                                                                
                                                                                
            else:                                                               
                # first unique atom type for this element                       
                unique_element_counter[element] = 1                             
                alpha_str=num2alpha(unique_element_counter[element])            
                this_descr=element+'_'+alpha_str                                
                                                                                
                                                                                
            unique_descr_to_int[dataset['equivalent_atoms'][i]]=this_descr      
            unique_descr_count[unique_descr_to_int[dataset['equivalent_atoms'][i]]] = 1
            descr.append(unique_descr_to_int[dataset['equivalent_atoms'][i]])   
            unique_descr_to_sites[unique_descr_to_int[dataset['equivalent_atoms'][i]]]=[i]
                                                                                
    #print("All descriptions:")                                                 
    #print(descr)                                                               
    debug_key=descr[0]                                                          
    print("Sample unique type key")                                             
    print(debug_key)                                                            
    print("All sites for this key")                                             
    print(unique_descr_to_sites[debug_key])                   

    return unique_descr_to_sites


def unique_faces_pym(ifname, verbose=True):

    # initial pymatgen structure object 
    #parser=pic.CifParser(ifname)
    #structure=parser.get_structures()[0]                      
    structure=Structure.from_file(ifname,primitive=False)
    #print(structure.lattice)

    # First check the symmetrically distinct Miller indices
    unique_hkl=pymatgen.core.surface.get_symmetrically_distinct_miller_indices(structure,2)
    custom_unique_hkl, custom_equiv_hkl, custom_equiv_hkl_dict=\
        custom_pymatgen_get_symmetrically_distinct_miller_indices(structure,2)

    if(verbose):
        print("Symmetrically distinct hkl Miller planes are (default):")
        print(unique_hkl)
        print("Symmetrically distinct hkl Miller planes are (custom):")
        print(custom_unique_hkl)
        print("Symmetrically distinct hkl Miller planes and their equivalencies are:")
        for key in custom_equiv_hkl:
            print(key[0], key[1]) 
        print("Equivalent hkl Miller planes mapped to the original:")
        print(custom_equiv_hkl_dict)

    # store the unique Miller faces to make the Wulff shape faster later on
    f=open(ifname[:-4]+".unique_millers","w")
    for key in custom_equiv_hkl:
        unique="%sx%sx%s "%(key[0][0],key[0][1],key[0][2])
        duplicates=""
        for elem in key[1]:
            duplicates+="%sx%sx%s "%(elem[0],elem[1],elem[2])
        duplicates+="\n"
        f.write(unique+duplicates)
    f.close()
    

    return custom_unique_hkl, custom_equiv_hkl, custom_equiv_hkl_dict

def create_slab_pym(ifname,slab_face,slab_thickness,slab_vacuum,verbose=True):
  
    print("\nCreating Pymatgen initial slab from initial file (%s):"%ifname) 

    # initial pymatgen structure object 
    #parser=pic.CifParser(ifname)
    #structure=parser.get_structures()[0]                      
    structure=Structure.from_file(ifname,primitive=False)
    #print(structure.lattice)

    ## First check the symmetrically distinct Miller indices
    #symm_distinct_hkl=pymatgen.core.surface.get_symmetrically_distinct_miller_indices(structure,2)
    #custom_symm_distinct_hkl, custom_symm_distinct_hkl_dict=\
    #    custom_pymatgen_get_symmetrically_distinct_miller_indices(structure,2)

    #if(verbose):
    #    print("Symmetrically distinct hkl Miller planes are:")
    #    print(custom_symm_distinct_hkl)
    #    print("Symmetrically distinct hkl Miller planes and their equivalencies are:")
    #    for key in custom_symm_distinct_hkl_dict.keys():
    #        print(key, custom_symm_distinct_hkl_dict[key])
    #    #print(custom_symm_distinct_hkl_dict)

    # initial slabgen object
    miller_list = tuple([int(elem) for elem in slab_face])
    slabgen=SlabGenerator(structure, 
                 miller_list,
                 min_slab_size=slab_thickness, 
                 min_vacuum_size=slab_vacuum,
                 primitive=False,
                 lll_reduce=True, center_slab=True
                         )


    # recalculate the new min slab thickness to have two additional layers
    layers_list=get_nlayers(slabgen)
    new_min_slab_size=(layers_list[0]+4)*slabgen._proj_height-0.1

    if(verbose):
        print("Projection height of slab for this face: %.3f"%(slabgen._proj_height))
        print("%d slab layers for MIN thickness of %.1f"%(layers_list[0],slab_thickness))

    ## create modified slabgen object
    #slabgen_no_lll=SlabGenerator(structure, 
    #             (int(slab_face[0]),int(slab_face[1]),int(slab_face[2])),
    #             min_slab_size=new_min_slab_size, 
    #             min_vacuum_size=slab_vacuum,
    #             primitive=True,
    #             lll_reduce=False, center_slab=True
    #                     )

    # new SlabGenerator object with a min slab thickness that gives the min+4 # of layers
    slabgen=SlabGenerator(structure, 
                 miller_list,
                 min_slab_size=new_min_slab_size, 
                 min_vacuum_size=slab_vacuum,
                 primitive=False,
                 lll_reduce=True, center_slab=True
                         )
    layers_list=get_nlayers(slabgen)
    print("%d slab layers for thickness of %.1f"%(layers_list[0],new_min_slab_size))

    #slabs=slabgen.get_slabs()
    #for i in range(len(slabs)):
    #    print("\nSlab: %d"%i)                                                  
    #    print(slabs[i].lattice)                                                
    #    print("Num sites: %d"%len(slabs[i].sites))                             
    #                                                                           
    #    # symmetric slab                                                       
    #    outputter=pic.CifWriter(slabs[i])                                      
    #    outputter.write_file("slab_id%05d.cif"%i) 

    frac_coords=slabgen.oriented_unit_cell.frac_coords
    #print("frac_coords in oriented unit cell")
    #print(frac_coords)


    shift=0
    frac_coords = np.array(frac_coords) +\
                  np.array([0, 0, -shift])[None, :]
    #print("frac_coords plus some shift")
    #print(frac_coords)

    frac_coords[:, 2] = frac_coords[:, 2] / layers_list[2]
    #print("frac_coords modded by n layers (%d)"%layers_list[2])
    #print(frac_coords)

    oriented_lattice = slabgen.oriented_unit_cell.lattice
    #print("Oriented lattice vectors")
    #print(oriented_lattice)


    # create slab
    #slab = slabgen.get_slab()
    slab = custom_pymatgen_get_slab(slabgen)

    #print("Orig lab lattice")
    #print
    scale_factor= slabgen.slab_scale_factor
    lll_slab = slab.copy(sanitize=True)
    mapping = lll_slab.lattice.find_mapping(slab.lattice)
    scale_factor = np.dot(mapping[2], scale_factor)
    #print("Mapping, scale factor")
    #print(mapping)
    #print(scale_factor)


    #print("reg lattice")
    #print(structure._lattice)
    reduced_latt =structure._lattice.get_lll_reduced_lattice()
    #print("reduced_latt")
    #print(reduced_latt)

    # get a b shift
    #print("frac coords")
    #print(slab.sites[0]._fcoords)
    #print(slab.sites[a_per_l]._fcoords)
    
    # compute # of atoms per slab layer
    slab_L=layers_list[0]
    a_per_l = int(len(slab.sites)/slab_L)
    # number of Si per layer
    Si_per_l = a_per_l/3
    


    # make sure has inversion symmetry, otherwise for now error
    # when we find a min cut, we know that the other is just the -1 symm group
    laue = ["-1", "2/m", "mmm", "4/m", "4/mmm",                                 
            "-3", "-3m", "6/m", "6/mmm", "m-3", "m-3m"] 

    tol=0.5
    sg = pymatgen.symmetry.analyzer.SpacegroupAnalyzer(slab, symprec=tol)       
    pg = sg.get_point_group_symbol()                                            
    if str(pg) in laue:                                                         
        #return slab                                                            
        print("Point group is %s"%pg)
        print("Laue symmetry found!")                                    
    else:                                                                       
        print("Point group is %s"%pg)
        print("No Laue symmetry found...")   
        print("Result will give cleavage energy (see Pymatgen paper) rather than the rigorous surface energy of this truncation")
        #sys.exit()
   
    # get equivalent atom types by symmetry:
    symm_struct  = sg.get_symmetrized_structure()
    symm_dataset = sg.get_symmetry_dataset()
    unique_type_dataset = unique_typing(symm_struct,symm_dataset)
    inversion_dataset   = inversion_mapping(symm_dataset, unique_type_dataset,slab)
    # get translationally equivalent atom types between slab layers
    translation_dataset = translation_mapping(slab_L,a_per_l,slab)

    # re-write new slab to CIF so LAMMPS interface can re-interpret this new slab
    miller_string=""
    for i in range(len(miller_list)-1):
        miller_string+=str(miller_list[i])+'x'
    miller_string+=str(miller_list[-1])
    ofname = str(ifname[:-4])+"_"+\
             miller_string+"_"+\
             str(slab_L)+"_slab_pym"
    outputter = pic.CifWriter(slab)
    outputter.write_file(ofname+".cif")
  

    layer_props = {'slab_L':slab_L, 'a_per_l':a_per_l, 'proj_height':slabgen._proj_height,
                   'ofname':ofname, 'inversion_mapping':inversion_dataset, 
                   'translation_mapping':translation_dataset}

    print("Layer properties:") 
    print(layer_props)

    return layer_props 


def translation_mapping(slab_L, a_per_l, slab):
    """
    slab_L = number of layers (starting at index 0)
    a_per_L = number of atoms in a layer

    Map a site in layer i (starts at index 0) to a site in layer (slab_L-1-i)
    """
    # NOTE important this is not rigorous since we are sorting possible identical c -values
    # need to get the layer data directly from pymatgen
    
    print("\nCreating translation dictionary:")
    translation_dataset = {}

    # the annoying thing is that Pymatgen may have re-ordered the list of sites
    # to group similar atom types together (but preserving the c-order within that type group)
    # therefore get the order back for all sites 

    # to get around this we need to make sure that the list of sites has 
    # not been altered by Pymatgen sort, hence the custom functions above

    for i in range(len(slab.sites)):
        #print(slab.sites[i])

        # layer of site i is (starting at index 0:
        this_layer=int(np.floor(i/a_per_l))

        opp_layer=(slab_L-1)-this_layer

        delta_layer=opp_layer-this_layer

        # "opposite" site
        opp_of_i = int(delta_layer*a_per_l+i)

        # NOTE nodes start at index 1
        translation_dataset[i+1]        = opp_of_i+1
        
    return translation_dataset
    # NOTE for now the translational mapping of a cut depends on the layers
    # attributes of ALL atoms in the cut
    # but doing this means we could still find the translational atom equivalane in the opposite layer

def inversion_mapping(symm_dataset,unique_type_dataset,slab):
    """
    Get the mapping between any atom and its -1*I+T equivalent 
    aka inversion symmetry across the slab
    """
    
    print("\nCreating inversion dictionary:")
    inversion_dictionary = {}

    # inversion rotation matrix
    inv_rot   = np.array([[-1,0,0],[0,-1,0],[0,0,-1]]) 
    # index in Pymatgen rotations key
    inv_ind   = -1
   
    # find the corresponding translation for this rotation 
    for i in range(len(symm_dataset['rotations'])):
        if(np.array_equal(symm_dataset['rotations'][i],inv_rot)):
            inv_ind = i
    if(inv_ind==-1):
        print("No inversion matrix found -1*I (-1 diagonal matrix)")
        #sys.exit()
        return None


    trans     = symm_dataset['translations'][inv_ind]

    print("All rotations:")
    print(symm_dataset['rotations']) 
    print("Corresponding translations:")
    print(symm_dataset['translations']) 
    print("Rot and trans corresponding to the inversion:")
    print(inv_rot, trans)

    # This gets messy... there must be a way to do this in Pymatgen ...
    for key in unique_type_dataset.keys():
        all_equiv_sites=unique_type_dataset[key]
        all_equiv_coords=[slab.sites[site]._fcoords for site in all_equiv_sites]
        print(key, all_equiv_sites)
        print(key, all_equiv_coords)
        match_ind=-1
        for site in all_equiv_sites:
            print("Site %d:"%site, slab.sites[site]._fcoords)
            rot=np.dot(inv_rot,slab.sites[site]._fcoords)
            rotptrans=rot+trans
            rotptrans = wrap_between_0_and_1(rotptrans)
            print("Mirror %d:"%site, rotptrans)

            found_mirror=False
            for j in range(len(all_equiv_coords)):
                if(np.allclose(rotptrans, all_equiv_coords[j],atol=1e-4)):
                    found_mirror=True
                    match_ind = j 
                    break

            # This means we messed up because Pymatgen already said we have inversion symm
            if(not found_mirror):
                print("Oops, inversion symmetry supposed to exist but symmetric site not found, check source code...")
                print("This is a serious bug that needs to be fixed")
                return None
            
            site_match_ind=all_equiv_sites[match_ind]
            #print("Matches site %d"%site_match_ind)
        
            # add 1 to both indices since equivalent sites are starting index 0
            # but labels and nodes in lammps interface start at 1
            inversion_dictionary[site+1]=site_match_ind+1

    return inversion_dictionary



def wrap_between_0_and_1(it):
    """
    wrap  a 1-D iterable of floats to between 0 and 1
    """
    for i in range(len(it)):
        while(it[i]>1.0):
            it[i]-=1
        while(it[i]<0.0):
            it[i]+=1
        
        # by default if coord is == 1.0 w/in a tol  and not caught by > 1.0 
        # make it == 0
        if(it[i]<1.0+1e-8 and it[i]>1.0-1e-8):
            it[i]=0.0

    return it
###############################################################################
# END Pymatgen slab helper functions
###############################################################################

###############################################################################
# START ASE slab helper functions
###############################################################################

def get_initial_slab_L_ase(ifname,slab_face,slab_L,slab_vacuum):
    """
    Helper function for ASE slab generation

    Estimate the # of layers needed to make a slab of a specified thickness
    """
    cell = read(ifname)
    slab_tmp=surface(cell,
                     (int(slab_face[0]),int(slab_face[1]),int(slab_face[2])),
                     1,
                     vacuum=12.5)   
    num_layers=int(round(30.0/(slab_tmp.get_cell_lengths_and_angles()[2]-25)))+1
    return num_layers

def create_slab_ase(ifname,slab_face,slab_L,slab_vacuum):
    """
    Helper function for ASE slab generation

    Write the ASE slab 
    """
    cell = read(ifname)
    slab=surface(cell,
                 (int(slab_face[0]),int(slab_face[1]),int(slab_face[2])),
                 slab_L,
                 slab_vacuum)
    slab.center()                                                                   
    ofname = str(ifname[:-4])+"_"+\
             str(slab_face[0])+str(slab_face[1])+str(slab_face[2])+"_"+\
             str(slab_L)+"_slab_ase.cif"

    print("Writing slab to %s"% ofname)                                             
    write(ofname,slab)
    return ofname

###############################################################################
# END ASE slab helper functions
###############################################################################

def produce_optimal_surface_slab(options, sim, face):
    """
    Call all necessary functions to create optimal surface slab for the specified
    parameters
    """

    sim = LammpsSimulation(options)

    sim.slab_face         = face
    sim.slab_L            = options.slab_L # no matter what will be overwritten in pymatgen
    sim.slab_vacuum       = options.slab_vacuum
    sim.slab_target_thick = options.slab_target_thick
    sim.slab_verbose      = options.slab_verbose

    # min cut parameters
    sim.mincut_eps = options.mincut_eps
    sim.mincut_k   = options.mincut_k
    sim.maxnumcuts    = options.maxnumcuts 

    print("Input parameters:")
    print("Miller face: " + str(sim.slab_face         ))
    print("Slab vacuum: " + str(sim.slab_vacuum       ))
    print("Slab thick: " + str(sim.slab_target_thick  ))
    print("Mincut eps: " + str(sim.mincut_eps         ))
    print("Mincut k:   " + str(sim.mincut_k           ))
    print("Max num cuts to enumerate: " + str(sim.maxnumcuts))


    # create pymatgen structure and get layer properties
    layer_props=create_slab_pym(options.cif_file,sim.slab_face,sim.slab_target_thick,sim.slab_vacuum)

    # reset the structure properties to reflect the newest Pymatgen slab
    curr_slab_cif_name=str(layer_props['ofname'])+".cif"
    cell, graph = from_CIF(curr_slab_cif_name)                                    
    sim.set_cell(cell)                                                          
    sim.set_graph(graph)                                                        
    #sim.split_graph()                                                           
    sim.assign_force_fields()                                                   
                                                                            
    # Beginning of routines for surface generation
    sim.slabgraph=SlabGraph(sim.graph,cell) 
    sim.slabgraph.check_if_zeolite()                                            
                                                                                
    sim.merge_graphs()                                                          
    # DEBUG file writing: reprint the slab graph since Pymatgen cif can't be read by mercury
    sim.slabgraph.write_slabgraph_cif(sim.slabgraph.slabgraph,cell,bond_block=False,descriptor=None,relabel=False) 
                                                                                
    # assign the slab layers to the initial pymatgen slab
    sim.slabgraph.assign_slab_layers(layer_props)

    # creating slabgraph and condensing it to Si only graph 
    sim.slabgraph.remove_erroneous_disconnected_comps()                         
    sim.slabgraph.condense_graph()                                              

    # DEBUG file writing: output cif with false elements corresponding to slablayer for visualization
    sim.slabgraph.debug_slab_layers()

    # identify the undercoordinated surface nodes
    sim.slabgraph.identify_undercoordinated_surface_nodes()                     
    
    # Here we need to manipulate the edge properties of the graph
    sim.slabgraph.normalize_bulk_edge_weights()                                 
    sim.slabgraph.connect_super_surface_nodes_v2() 

    # Create a directed copy for max-flow/min-cut st problem
    sim.slabgraph.convert_to_digraph()                                            

    # DEBUG file writing before graph cutting 
    if(sim.slab_verbose):
        sim.slabgraph.write_slabgraph_cif(sim.slabgraph.slabgraph,cell,bond_block=False,descriptor="debug",relabel=False) 

    # Balcioglu and Wood 2003
    # add arg max_num_cuts=1 for regular s-t min cut from 
    sim.slabgraph.nx_near_min_cut_digraph_custom(sim.mincut_eps, sim.mincut_k, \
        weight_barrier=True, layer_props=layer_props, max_num_cuts=sim.maxnumcuts)
    # generate all slabs from all cutsets
    sim.slabgraph.generate_all_slabs_from_cutsets()

def main():                                                                     
                                                                                
    # command line parsing                                                      
    options = Options()                                                         
    sim = LammpsSimulation(options)                                             

    cell, graph = from_CIF(options.cif_file)                                    
    sim.set_cell(cell)                                                          
    sim.set_graph(graph)                                                        
    sim.split_graph()                                                           
    sim.assign_force_fields()                                                   
    sim.merge_graphs()

    write_CIF(graph,cell,bond_block=False,descriptor="original",relabel=True)

    # TODO May need a big preparation step here that each Si is more than 2 eges away
    # from its periodic image
    # For now, naiive way is to say that if a lattice dimension is < ~6 Angstrom, 
    # we need to duplicate once in that direction
    ############################
    # START MINIMUM SUPERCELL GENERATION
    ############################
    # minimum supercell has to be 6 A in each orthogonal direction
    # corresponding to 3 A cutoff
    sim.options.cutoff=3.5
    sim.compute_simulation_size() 
    sim.merge_graphs()

    # Overwrite old cif with new minimum supercell
    write_CIF(graph,cell,bond_block=False,descriptor=None)

    # Reload the new CIF
    options = Options()                                                         
    sim = LammpsSimulation(options)                                             

    cell, graph = from_CIF(options.cif_file)                                    
    sim.set_cell(cell)                                                          
    sim.set_graph(graph)                                                        
    sim.split_graph()                                                           
    sim.assign_force_fields()                                                   
    ############################
    # FINISHED MINIMUM SUPERCELL GENERATION
    ############################
   
    sim.slab_pym=options.slab_pym 
    unique_hkl_list, equiv_hkl_list, equiv_hkl_dict\
         = unique_faces_pym(options.cif_file)

    if(not sim.slab_pym):
        print("For now we can only support slab generation with Pymatgen. Use CL arg --slab-pymatgen 1") 
        sys.exit()
        # NOTE for now removing support for any slab except one generated by Pymatgen
        #############################
        ## START AUTOMATED ASE SLAB GENERATION CRITERIA 
        #############################
        ## This is IMPORTANT 
        ## If the aspect ratio of the ASE slab is too small:
        ##    -the Stoer Wagner algorithm will fail and find a min cut partition that is not 2D periodic
        ##    -it basically tunnels between the source and sink nodes because creating a slab 
        ##     would involve cutting too many bonds due to the larger surface area w.r.t the slab thickness
        ##    -it simply means we haven't built a THICK enough slab to get the correct solution
        ##    -so we retry with an initial larger slab

        ## get slab option
        #sim.slab_face         = options.slab_face.split('x')
        #sim.slab_L            = options.slab_L
        #sim.slab_vacuum       = options.slab_vacuum
        #sim.slab_target_thick = options.slab_target_thick
        #sim.slab_verbose      = options.slab_verbose

        #if(sim.slab_L>0 and sim.slab_target_thick > 0.001):
        #    # Note that by default sim.slab_L=0 and sim.slab_target_thick = 30
        #    print("Notice! slab-L was specified!)")
        #    print("Specifying slab-L will create the smallest 2D periodic, min cut slab starting from an ASE slab with thickness L")
        #    print("Not pecifying slab-L will create the smallest 2D periodic, min cut slab with approximate thickness greater than target-thickness (for which 30 Angstrom is the default)")
        #    #sys.exit()

        ## so that thickness of ASE initial slab can be iteratively updated
        #curr_slab_L=int(sim.slab_L)


        ## If sim.slab_L is specified, start with this thickness
        #if(sim.slab_L != 0):
        #    curr_slab_cif_name = create_slab_ase(options.cif_file, sim.slab_face, 
        #                                                     curr_slab_L,
        #                                                     sim.slab_vacuum)
        #    max_slab_L=2*curr_slab_L
        ## otherwise estimate slab_L for a given target_thick
        #else:
        #    initial_slab_L=get_initial_slab_L_ase(options.cif_file,
        #                                          sim.slab_face,
        #                                          1,
        #                                          sim.slab_vacuum)
        #    curr_slab_cif_name=create_slab_ase(options.cif_file, sim.slab_face, 
        #                                                    initial_slab_L,
        #                                                    sim.slab_vacuum)
        #    curr_slab_L = int(initial_slab_L)
        #    max_slab_L  = 2*initial_slab_L
   
 
        ## Booleans to determine if we are finished with this surface    
        #next_iter=True
        #slab_is_2D_periodic=False
        #slab_meets_thickness_criteria=False
        #############################
        ## END AUTOMATED ASE SLAB GENERATION CRITERIA 
        #############################



        #############################
        ## START AUTOMATED ASE SLAB GENERATION
        #############################
        ## try to make surfaces until we hit stopping criteria
        #while(next_iter==True):
        #    # reset everything to beginning
        #    sim = LammpsSimulation(options)                                             
        #    sim.slab_face         = options.slab_face.split('x')
        #    sim.slab_L            = options.slab_L
        #    sim.slab_vacuum       = options.slab_vacuum
        #    sim.slab_target_thick = options.slab_target_thick
        #    sim.slab_verbose      = options.slab_verbose

        #    # reset the structure properties to reflect the newest ASE slab
        #    cell, graph = from_CIF(curr_slab_cif_name)                                    
        #    sim.set_cell(cell)                                                          
        #    sim.set_graph(graph)                                                        
        #    sim.split_graph()                                                           
        #    sim.assign_force_fields()                                                   


        #    # Beginning of routines for surface generation
        #    sim.slabgraph=SlabGraph(sim.graph,cell) 
        #    sim.slabgraph.check_if_zeolite()                                            
        #                                                                                
        #    sim.merge_graphs()                                                          

        #    # DEBUG file writing: reprint the slab graph since ASE cif can't be read by mercury
        #    sim.slabgraph.write_slabgraph_cif(cell,bond_block=False,descriptor=None,relabel=False) 
        #                                                                                
        #    # creating slab graph                                                       
        #    sim.slabgraph.remove_erroneous_disconnected_comps()                         
        #    sim.slabgraph.condense_graph()                                              
        #    sim.slabgraph.identify_undercoordinated_surface_nodes()                     
        #    
        #    # DEBUG file writing                                                                    
        #    if(sim.slab_verbose):
        #        sim.slabgraph.write_slabgraph_cif(cell,bond_block=False,descriptor="debug",relabel=False) 

        #    # Here we need to manipulate the edge properties of the graph
        #    sim.slabgraph.normalize_bulk_edge_weights()                                 
        #    sim.slabgraph.connect_super_surface_nodes()                                 

        #    # Create a directed copy for max-flow/min-cut st problem
        #    sim.slabgraph.convert_to_digraph()                                            

        #    #sim.slabgraph.nx_near_min_cut_digraph_custom(weight_barrier=True)               
        #    # Execute max-flow/min-cut calculation
        #    sim.slabgraph.nx_min_cut_digraph_custom(weight_barrier=True)               

        #    # exclude graph partitions on the "outside" of the min cut
        #    sim.slabgraph.remove_surface_partitions()                                   
        #                                                                                
        #    # Add back in all missing oxygens                                           
        #    sim.slabgraph.add_all_connecting_nodes()                                    

        #    # DEBUG file writing
        #    if(sim.slab_verbose):
        #        sim.slabgraph.write_slabgraph_cif(cell,bond_block=False,descriptor="deH",relabel=False)   
        #                                                                                
        #    # add missing hydrogen caps and validate structural properties 
        #    sim.slabgraph.add_missing_hydrogens()                                       
        #    # check approximate slab thickness
        #    min_thickness = sim.slabgraph.check_approximate_slab_thickness()
        #    max_thickness = sim.slabgraph.check_approximate_slab_thickness_v2()
        #    approximate_thickness = (max_thickness+min_thickness)/2
        #    # check if generated slab is 2D periodic
        #    slab_is_2D_periodic   = sim.slabgraph.check_slab_is_2D_periodic()
        #   
 
        #    # check if we need to do another iteration with an L+=1 ASE slab
        #    if(slab_is_2D_periodic):
        #        print("Identified slab is periodic")
        #        if(sim.slab_L>0):
        #            next_iter=False
        #            print("Current slab_L %d >= specified_slab_L: %d"%(curr_slab_L, sim.slab_L))
        #            print("Stopping slab genertion")
        #        elif(sim.slab_L==0):
        #            if(approximate_thickness>sim.slab_target_thick):
        #                next_iter=False
        #                print("Current slab thickness %.4f > target thickness %.4f"%
        #                      (approximate_thickness, sim.slab_target_thick))
        #                print("Stopping slab genertion")

        #            else:
        #                next_iter=True
        #                curr_slab_L+=1
        #                print("Current slab thickness %.4f < target thickness %.4f"%
        #                      (approximate_thickness, sim.slab_target_thick))
        #                print("Moving on to ASE slab genertion with L = %d"%curr_slab_L)
        #    else:
        #        next_iter=True
        #        curr_slab_L+=1
        #        print("Identified slab is NOT periodic")
        #        print("Moving on to ASE slab genertion with L = %d"%curr_slab_L)
        #    
        #    # finally, we need some override stopping criteria in case something weird going on
        #    if(curr_slab_L>max_slab_L):
        #        print("Error! Maximum slab L exceeded without finding periodic solution")
        #        next_iter=False

        #    # if we are going to do another iteration, need to create the next iteration of ASE slab
        #    if(next_iter==True):
        #        print("Removing ASE slab: %s"%curr_slab_cif_name)
        #        print("\n\n\n")
        #        if(not sim.slab_verbose):
        #            os.remove(curr_slab_cif_name)
        #        curr_slab_cif_name=create_slab_ase(options.cif_file, sim.slab_face, 
        #                                                        curr_slab_L,
        #                                                        sim.slab_vacuum)
        #    else:
        #        # if we have succeeded can write the final files here if verbose
        #        if(sim.slab_verbose):
        #            sim.slabgraph.write_slabgraph_cif(cell,bond_block=False,descriptor="addH",relabel=False)   
        #            sim.slabgraph.write_average_silanol_density(curr_slab_cif_name[:-4]+".addH.dat")
        #        # and if not verbose only write if we suceeded to minimize # of files written
        #        else:
        #            if(slab_is_2D_periodic):
        #                sim.slabgraph.write_slabgraph_cif(cell,bond_block=False,descriptor="addH",relabel=False)   
        #                sim.slabgraph.write_average_silanol_density(curr_slab_cif_name[:-4]+".addH.dat")
        #        pass
        #############################
        ## END AUTOMATED ASE SLAB GENERATION
        #############################

    else:
        ############################
        # START AUTOMATED PYMATGEN SLAB GENERATION
        ############################

        # if the slab face all option is used, we create a slab surface for each
        # symmetrically unique surface for this zeolite
        

        # slab parameters
        sim.slab_face         = options.slab_face.strip().split('x')
        sim.slab_L            = options.slab_L # no matter what will be overwritten in pymatgen
        sim.slab_vacuum       = options.slab_vacuum
        sim.slab_target_thick = options.slab_target_thick
        sim.slab_verbose      = options.slab_verbose

        # min cut parameters
        sim.mincut_eps = options.mincut_eps
        sim.mincut_k   = options.mincut_k
        sim.maxnumcuts = options.maxnumcuts

        print("Input parameters:")
        print("Miller face: " + str(sim.slab_face         ))
        print("Slab vacuum: " + str(sim.slab_vacuum       ))
        print("Slab thick: " + str(sim.slab_target_thick  ))
        print("Mincut eps: " + str(sim.mincut_eps         ))
        print("Mincut k:   " + str(sim.mincut_k           ))
        print("Max num cuts to enumerate: " + str(sim.maxnumcuts))


        if(sim.slab_face[0]=="all"):
            print("Generating all unique miller slab faces: %s"%str(unique_hkl_list))
            for face in unique_hkl_list:
                produce_optimal_surface_slab(options, sim, face)

        else:
            sim.slab_face = tuple([int(elem) for elem in sim.slab_face])
            print("Generating miller slab face: %s"%str(sim.slab_face))

            if(sim.slab_face in equiv_hkl_dict.keys()):
                new_face=deepcopy(equiv_hkl_dict[sim.slab_face])
                print("Warning, hkl of %s is equivalent to hkl of %s"%\
                    (sim.slab_face, new_face))
                print("Creating surface slab of hkl: " + str(new_face))
                produce_optimal_surface_slab(options, sim, new_face)
            else:
                print("Miller slab hkl = %s is symmetrically unique, continuing..."%\
                    str(sim.slab_face))
                produce_optimal_surface_slab(options, sim, sim.slab_face)


        ## create pymatgen structure and get layer properties
        #layer_props=create_slab_pym(options.cif_file,sim.slab_face,sim.slab_target_thick,sim.slab_vacuum)

        ## reset the structure properties to reflect the newest Pymatgen slab
        #curr_slab_cif_name=str(layer_props['ofname'])+".cif"
        #cell, graph = from_CIF(curr_slab_cif_name)                                    
        #sim.set_cell(cell)                                                          
        #sim.set_graph(graph)                                                        
        ##sim.split_graph()                                                           
        #sim.assign_force_fields()                                                   
        #                                                                        
        ## Beginning of routines for surface generation
        #sim.slabgraph=SlabGraph(sim.graph,cell) 
        #sim.slabgraph.check_if_zeolite()                                            
        #                                                                            
        #sim.merge_graphs()                                                          



        ## DEBUG file writing: reprint the slab graph since Pymatgen cif can't be read by mercury
        #sim.slabgraph.write_slabgraph_cif(sim.slabgraph.slabgraph,cell,bond_block=False,descriptor=None,relabel=False) 
        #                                                                            
        ## assign the slab layers to the initial pymatgen slab
        #sim.slabgraph.assign_slab_layers(layer_props)

        ## creating slabgraph and condensing it to Si only graph 
        #sim.slabgraph.remove_erroneous_disconnected_comps()                         
        #sim.slabgraph.condense_graph()                                              

        ## DEBUG file writing: output cif with false elements corresponding to slablayer for visualization
        #sim.slabgraph.debug_slab_layers()

        ## identify the undercoordinated surface nodes
        #sim.slabgraph.identify_undercoordinated_surface_nodes()                     
        #
        ## Here we need to manipulate the edge properties of the graph
        #sim.slabgraph.normalize_bulk_edge_weights()                                 
        #sim.slabgraph.connect_super_surface_nodes_v2() 

        ## Create a directed copy for max-flow/min-cut st problem
        #sim.slabgraph.convert_to_digraph()                                            

        ## DEBUG file writing before graph cutting 
        #if(sim.slab_verbose):
        #    sim.slabgraph.write_slabgraph_cif(sim.slabgraph.slabgraph,cell,bond_block=False,descriptor="debug",relabel=False) 

        ## Balcioglu and Wood 2003
        ## add arg max_num_cuts=1 for regular s-t min cut from 
        #sim.slabgraph.nx_near_min_cut_digraph_custom(sim.mincut_eps, sim.mincut_k, weight_barrier=True, layer_props=layer_props)               
        ## generate all slabs from all cutsets
        #sim.slabgraph.generate_all_slabs_from_cutsets()

        ############################
        # END AUTOMATED PYMATGEN SLAB GENERATION
        ############################
                                                                                
                                                                                
    # Additional capability to write RASPA files if requested                   
    if options.output_raspa:                                                    
        classifier=1                                                            
        print("Writing RASPA files to current WD")                              
        write_RASPA_CIF(graph, cell, classifier)                                
        write_RASPA_sim_files(sim, classifier)                                  
        this_config = MDMC_config(sim)                                          
        sim.set_MDMC_config(this_config)                                        
                                                                                
if __name__ == "__main__":                                                      
    main()                                                                      
                                                                                

