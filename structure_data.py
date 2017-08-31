#/isr/bin/env python
from datetime import date
import numpy as np
from scipy.spatial import distance
import math
import shlex
import re
from CIFIO import CIF
from atomic import METALS, MASS, COVALENT_RADII
from copy import copy
from mof_sbus import InorganicCluster, OrganicCluster
from copy import deepcopy
import itertools
import os, sys
from generic_raspa import GENERIC_PSEUDO_ATOMS_HEADER, GENERIC_PSEUDO_ATOMS, \
                          GENERIC_FF_MIXING_HEADER, GENERIC_FF_MIXING,\
                          GENERIC_FF_MIXING_FOOTER
from uff import UFF_DATA
import networkx as nx
import operator

from collections import OrderedDict
from collections import Counter
from atomic import MASS, ATOMIC_NUMBER, COVALENT_RADII
from atomic import organic, non_metals, noble_gases, metalloids, lanthanides, actinides, transition_metals
from atomic import alkali, alkaline_earth, main_group, metals
from ccdc import CCDC_BOND_ORDERS
DEG2RAD=np.pi/180.

#try:
#    from writeNodesEdges import writeObjects
#except:
#    print("Warning! vtk for python necessary for graph debugging necessary. Code will ImportError if you try to use the functionality to draw a SlabGraph to a VTK style visualization file")

try:
    import networkx as nx
    from networkx.algorithms import approximation
except ImportError:
    print("WARNING: could not load networkx module, this is needed to produce the lammps data file.")
    sys.exit()

import nxstoerwagnercustom as nxswc
try:
    import transforamtions as trans
    import nxmaxflowcustom as nxmfc
except:
    print("WARNING: Not able to load custom nx algorithms (make sure at most recent git commit)")


class MolecularGraph(nx.Graph):
    """Class to contain all information relating a structure file
    to a fully described classical system.
    Important specific arguments for atomic nodes:
    - mass
    - force_field_type
    - charge
    - cartesian_coordinates
    - description {contains all information about electronic environment
                   to make a decision on the final force_field_type}
        -hybridization [sp3, sp2, sp, aromatic]
    
    Important arguments for bond edges:
    - weight = 1
    - length
    - image_flag
    - force_field_type
    """
    node_dict_factory = OrderedDict
    def __init__(self, **kwargs):
        nx.Graph.__init__(self, **kwargs)
        # coordinates and distances will be kept in a matrix because 
        # networkx edge and node lookup is slow.
        try:
            self.name = kwargs['name']
        except KeyError:
            self.name = 'default'
        self.coordinates = None
        self.distance_matrix = None
        self.original_size = 0
        self.molecule_id = 444
        self.inorganic_sbus = {}
        self.find_metal_sbus = False
        self.organic_sbus = {}
        self.find_organic_sbus = False
        self.cell = None
        self.rigid = False
        #TODO(pboyd): networkx edges do not store the nodes in order!
        # Have to keep a dictionary lookup to make sure the nodes 
        # are referenced properly (particularly across periodic images)
        self.sorted_edge_dict = {}
        self.molecule_images = []
        #FIXME(pboyd): latest version of NetworkX has removed nodes_iter...

    def edges_iter2(self, **kwargs):
        for n1, n2, d in self.edges_iter(**kwargs):
            yield (self.sorted_edge_dict[(n1, n2)][0], self.sorted_edge_dict[(n1,n2)][1], d)
    
    def count_dihedrals(self):
        count = 0
        for n1, n2, data in self.edges_iter(data=True):
            try:
                for dihed in data['dihedrals'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_angles(self):
        count = 0
        for node, data in self.nodes_iter(data=True):
            try:
                for angle in data['angles'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def count_impropers(self):
        count = 0
        for node, data in self.nodes_iter(data=True):
            try:
                for angle in data['impropers'].keys():
                    count += 1
            except KeyError:
                pass
        return count

    def reorder_labels(self, reorder_dic):
        """Re-order the labels of the nodes so that LAMMPS doesn't complain.
        This issue only arises when a supercell is built, but isolated molecules
        are not replicated in the supercell (by user request).
        This creates a discontinuity in the indices of the atoms, which breaks
        some features in LAMMPS.

        """

        old_nodes = sorted([(i,self.node[i]) for i in self.nodes()])
        #old_nodes = list(self.nodes_iter(data=True))
        old_edges = list(self.edges_iter2(data=True))
        for node, data in old_nodes:
            
            if 'angles' in data:
                ang_data = list(data['angles'].items())
                for (a,c), val in ang_data:
                    data['angles'].pop((a,c))
                    data['angles'][(reorder_dic[a], reorder_dic[c])] = val
            if 'impropers' in data:
                imp_data = list(data['impropers'].items())
                for (a, c, d), val in imp_data:
                    data['impropers'].pop((a,c,d))
                    data['impropers'][(reorder_dic[a], reorder_dic[c], reorder_dic[d])] = val

            self.remove_node(node)
            data['index'] = reorder_dic[node]
            self.add_node(reorder_dic[node], **data)

        for b, c, data in old_edges:
            if 'dihedrals' in data:
                dihed_data = list(data['dihedrals'].items())
                for (a, d), val in dihed_data:
                    data['dihedrals'].pop((a,d))
                    data['dihedrals'][(reorder_dic[a], reorder_dic[d])] = val
            try:
                self.remove_edge(b,c)
            except nx.exception.NetworkXError:
                # edge already removed from 'remove_node' above
                pass
            self.add_edge(reorder_dic[b], reorder_dic[c], **data)
        
        old_edge_dict = self.sorted_edge_dict.items()
        self.sorted_edge_dict = {}
        for (a,b), val in old_edge_dict:
            self.sorted_edge_dict[(reorder_dic[a], reorder_dic[b])] = (reorder_dic[val[0]], reorder_dic[val[1]])

        old_images = self.molecule_images[:]
        self.molecule_images = []
        for m in old_images:
            newm = [reorder_dic[i] for i in m]
            self.molecule_images.append(newm)

    def add_atomic_node(self, **kwargs):
        """Insert nodes into the graph from the cif file"""
        #update keywords with more atom info
        # rename this to something more intuitive
        label="_atom_site_type_symbol"
        orig_keys = list(kwargs.keys())
        if(label not in kwargs):
            label = "_atom_site_label"
            if (label not in kwargs):
                print("ERROR: could not find the keyword for the element types in the cif file!"+
                        " Please use '_atom_site_type_symbol' or '_atom_site_label' for the element"+
                        " column.")
                sys.exit()

        charge_keywords = ["_atom_type_partial_charge", 
                           "_atom_type_parital_charge", 
                           "_atom_type_charge", 
                           "_atom_site_charge" # RASPA cif file
                           ]
        element = kwargs.pop(label)

        # replacing Atom.__init__
        kwargs.update({'mass':MASS[element]})
        kwargs.update({'molid':self.molecule_id})
        kwargs.update({'element':element})
        kwargs.update({'cycle':False})
        kwargs.update({'rings':[]})
        kwargs.update({'atomic_number':ATOMIC_NUMBER.index(element)})
        kwargs.update({'pair_potential':None})
        kwargs.update({'h_bond_donor':False})
        kwargs.update({'h_bond_potential':None})
        kwargs.update({'tabulated_potential':False})
        kwargs.update({'table_potential':None})
        if set(orig_keys) & set(charge_keywords):
            key = list(set(orig_keys)&set(charge_keywords))[0]
            try:
                kwargs['charge'] = float(kwargs[key])
            except ValueError:
                print("Warning %s could not be converted "%(kwargs[key]) + 
                      "to a charge value for atom %s"%(element) + 
                      ", setting charge as 0.0 for this atom")
                kwargs['charge'] = 0.0
        else:
            kwargs['charge'] = 0.0
        try:
            fftype = kwargs.pop('_atom_site_description')
        except KeyError:
            fftype = None

        kwargs.update({'force_field_type':fftype})
        idx = self.number_of_nodes() + 1
        kwargs.update({'index':idx})
        #TODO(pboyd) should have some error checking here..
        try:
            n = kwargs.pop('_atom_site_label')
        except KeyError:
            n = label
        kwargs.update({'ciflabel':n})
        # to identify Cu paddlewheels, etc.
        #kwargs.update({'special_flag':None})
        self.add_node(idx, **kwargs)
   
    def compute_bonding(self, cell, scale_factor = 0.9):
        """Computes bonds between atoms based on covalent radii."""
        # here assume bonds exist, populate data with lengths and 
        # symflags if needed.
        if (self.number_of_edges() > 0):
            # bonding found in cif file
            sf = []
            for n1, n2, data in self.edges_iter2(data=True):
                # get data['ciflabel'] for self.node[n1] and self.node[n2]
                # update the sorted_edge_dict with the indices, not the 
                # cif labels
                n1data = self.node[n1]
                n2data = self.node[n2]
                n1label = n1data['ciflabel']
                n2label = n2data['ciflabel']
                try:
                    nn1, nn2 = self.sorted_edge_dict.pop((n1label, n2label))
                    if nn2 == n1label:
                        nn1 = n2
                        nn2 = n1
                    self.sorted_edge_dict.update({(n1, n2):(nn1, nn2)})
                    self.sorted_edge_dict.update({(n2, n1):(nn1, nn2)})
                except KeyError:
                    pass
                try:
                    nn1, nn2 = self.sorted_edge_dict.pop((n2label, n1label))
                    if nn2 == n1label:
                        nn1 = n2
                        nn2 = n1
                    self.sorted_edge_dict.update({(n2, n1):(nn1, nn2)})
                    self.sorted_edge_dict.update({(n1, n2):(nn1, nn2)})
                except KeyError:
                    pass

                sf.append(data['symflag'])
                bl = data['length']
                if bl <= 0.01:
                    id1, id2 = self.node[n1]['index']-1, self.node[n2]['index']-1
                    dist = self.distance_matrix[id1,id2]
                    data['length'] = dist

            if (set(sf) == set(['.'])):
                # compute sym flags
                for n1, n2, data in self.edges_iter2(data=True):
                    flag = self.compute_bond_image_flag(n1, n2, cell)
                    data['symflag'] = flag
            return

        # Here we will determine bonding from all atom pairs using 
        # covalent radii.
        for n1, n2 in itertools.combinations(self.nodes(), 2):
            node1, node2 = self.node[n1], self.node[n2]
            e1, e2 = node1['element'],\
                    node2['element']
            elements = set([e1, e2])
            i1,i2 = node1['index']-1, node2['index']-1
            rad = (COVALENT_RADII[e1] + COVALENT_RADII[e2])
            dist = self.distance_matrix[i1,i2]
            tempsf = scale_factor
            # probably a better way to fix these kinds of issues..
            if (set("F") < elements) and  (elements & metals): 
                tempsf = 0.8

            if (set("O") < elements) and (elements & metals):
                tempsf = 0.85
            # fix for water particle recognition.
            if(set(["O", "H"]) <= elements):
                tempsf = 0.8
            # very specific fix for Michelle's amine appended MOF
            if(set(["N","H"]) <= elements):
                tempsf = 0.67
            if(set(["Mg","N"]) <= elements):
                tempsf = 0.80
            if(set(["C","H"]) <= elements):
                tempsf = 0.80
            if dist*tempsf < rad and not (alkali & elements):

                flag = self.compute_bond_image_flag(n1, n2, cell)
                self.sorted_edge_dict.update({(n1,n2): (n1, n2), (n2, n1):(n1, n2)})
                self.add_edge(n1, n2, key=self.number_of_edges() + 1, 
                              order=1.0, 
                              weight=1,
                              length=dist,
                              symflag = flag,
                              potential = None
                              )
    #TODO(pboyd) update this
    def compute_bond_image_flag(self, n1, n2, cell):
        """Update bonds to contain bond type, distances, and min img
        shift."""
        supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        unit_repr = np.array([5,5,5], dtype=int)
        atom1 = self.node[n1]
        atom2 = self.node[n2]
        #coord1 = self.coordinates[atom1['index']-1]
        #coord2 = self.coordinates[atom2['index']-1]
        coord1 = self.node[n1]['cartesian_coordinates']
        coord2 = self.node[n2]['cartesian_coordinates']
        fcoords = np.dot(cell._inverse, coord2) + supercells
        
        coords = np.array([np.dot(j, cell.cell) for j in fcoords])
        
        dists = distance.cdist([coord1], coords)
        dists = dists[0].tolist()
        image = dists.index(min(dists))
        dist = min(dists)
        sym = '.' if all([i==0 for i in supercells[image]]) else \
                "1_%i%i%i"%(tuple(np.array(supercells[image],dtype=int) +
                                  unit_repr))
        if(dist > 7):
            #print("BEFORE: ", self[n1][n2]['symflag'])
            print("WARNING: bonded atoms %i and %i are %.3f Angstroms apart."%(n1,n2,dist) + 
                    " This probably has something to do with the redefinition of the unitcell "+
                    "to a supercell. Please contact the developers!")
        return sym
    
    def compute_angle_between(self, l, m, r):
        coordl = self.node[l]['cartesian_coordinates']
        coordm = self.node[m]['cartesian_coordinates']
        coordr = self.node[r]['cartesian_coordinates']
       
        try:
            v1 = self.min_img(coordl - coordm)
            v2 = self.min_img(coordr - coordm)
        except AttributeError:
            v1 = coordl - coordm
            v2 = coordr - coordm

        v1 /= np.linalg.norm(v1)
        v2 /= np.linalg.norm(v2)

        a = np.arccos(np.dot(v1, v2))
        if np.isnan(a):
            if np.allclose((v1 + v2),np.zeros(3)):
                a = 180
            else:
                a = 0

        angle = a / DEG2RAD
        return angle
   
    def coplanar(self, node):
        """ Determine if this node, and it's neighbors are
        all co-planar.

        """
        coord = self.node[node]['cartesian_coordinates']
        vects = []
        for j in self.neighbors(node):
            vects.append(self.node[j]['cartesian_coordinates'] - coord)

        # use the first two vectors to define a plane
        v1 = vects[0]
        v2 = vects[1]
        n = np.cross(v1, v2)
        n /= np.linalg.norm(n)
        for v in vects[2:]:
            v /= np.linalg.norm(v)
            # what is a good tolerance for co-planarity in MOFs?
            # this is used solely to determine if a 4-coordinated metal atom
            # is square planar or tetrahedral..
            if not np.allclose(np.dot(v,n), 0., atol=0.02):
                return False
        return True

    def compute_dihedral_between(self, a, b, c, d):
        coorda = self.node[a]['cartesian_coordinates']
        coordb = self.node[b]['cartesian_coordinates']
        coordc = self.node[c]['cartesian_coordinates']
        coordd = self.node[d]['cartesian_coordinates']
        
        v1 = self.min_img(coorda - coordb)
        v2 = self.min_img(coordc - coordb)
        v3 = self.min_img(coordb - coordc) 
        v4 = self.min_img(coordd - coordc)

        n1 = np.cross(v1, v2)
        n2 = np.cross(v3, v4)

        n1 /= np.linalg.norm(n1)
        n2 /= np.linalg.norm(n2)

        a = np.arccos(np.dot(n1, n2))
        if np.isnan(a):
            a = 0
        angle = a / DEG2RAD
        return angle

    def add_bond_edge(self, **kwargs):
        """Add bond edges (weight factor = 1)"""
        #TODO(pboyd) should figure out if there are other cif keywords to identify
        # atom types
        #TODO(pboyd) this is .cif specific and should be contained within the cif 
        # file reading portion of the code. This is so that other file formats
        # can eventually be adopted if need be.

        n1 = kwargs.pop('_geom_bond_atom_site_label_1')
        n2 = kwargs.pop('_geom_bond_atom_site_label_2')
        try:
            length = float(del_parenth(kwargs.pop('_geom_bond_distance')))
        except KeyError:
            length = 0.0

        try:
            order = CCDC_BOND_ORDERS[kwargs['_ccdc_geom_bond_type']]
        except KeyError:
            order = 1.0

        try:
            flag = kwargs.pop('_geom_bond_site_symmetry_2')
        except KeyError:
            # assume bond does not straddle a periodic boundary
            flag = '.'
        kwargs.update({'length':length})
        kwargs.update({'weight': 1})
        kwargs.update({'order': order})
        kwargs.update({'symflag': flag})
        kwargs.update({'potential': None})
        # get the node index to avoid headaches
        for k,data in self.nodes_iter(data=True):
            if data['ciflabel'] == n1:
                n1 = k
            elif data['ciflabel'] == n2:
                n2 =k

        self.sorted_edge_dict.update({(n1,n2): (n1, n2), (n2, n1):(n1, n2)})
        self.add_edge(n1, n2, key=self.number_of_edges()+1, **kwargs)

    def compute_cartesian_coordinates(self, cell):
        """Compute the cartesian coordinates for each atom node"""
        coord_keys = ['_atom_site_x', '_atom_site_y', '_atom_site_z']
        fcoord_keys = ['_atom_site_fract_x', '_atom_site_fract_y', '_atom_site_fract_z']
        self.coordinates = np.empty((self.number_of_nodes(), 3))
        for node, data in self.nodes_iter(data=True):
            #TODO(pboyd) probably need more error checking..
            try:
                coordinates = np.array([float(del_parenth(data[i])) for i in coord_keys])
            except KeyError:
                coordinates = np.array([float(del_parenth(data[i])) for i in fcoord_keys])
                coordinates = np.dot(coordinates, cell.cell)
            data.update({'cartesian_coordinates':coordinates})

            self.coordinates[data['index']-1] = coordinates

    def compute_min_img_distances(self, cell):
        self.distance_matrix = np.empty((self.number_of_nodes(), self.number_of_nodes()))
        for n1, n2 in itertools.combinations(self.nodes(), 2):
            id1, id2 = self.node[n1]['index']-1,\
                                self.node[n2]['index']-1
            #coords1, coords2 = self.coordinates[id1], self.coordinates[id2]
            coords1, coords2 = self.node[n1]['cartesian_coordinates'], self.node[n2]['cartesian_coordinates']
            try:
                dist = self.min_img_distance(coords1, coords2, cell)
            except TypeError:
                sys.exit()
            self.distance_matrix[id1][id2] = dist
            self.distance_matrix[id2][id1] = dist
    
    def min_img(self, coord):
        f = np.dot(self.cell.inverse, coord)
        f -= np.around(f)
        return np.dot(f, self.cell.cell)
    
    def in_cell(self, coord):
        f = np.dot(self.cell.inverse, coord) % 1
        return np.dot(f, self.cell.cell)
    
    def fractional(self, coord):
        f = np.dot(self.cell.inverse, coord) 
        return f 

    def min_img_distance(self, coords1, coords2, cell):
        one = np.dot(cell.inverse, coords1) % 1
        two = np.dot(cell.inverse, coords2) % 1
        three = np.around(one - two)
        four = np.dot(one - two - three, cell.cell)
        return np.linalg.norm(four)

    def compute_init_typing(self):
        """Find possible rings in the structure and 
        initialize the hybridization for each atom.
        More refined determinations of atom and bond types
        is computed below in compute_bond_typing

        """
        #TODO(pboyd) return if atoms already 'typed' in the .cif file
        # compute and store cycles
        cycles = []
        for node, data in self.nodes_iter(data=True):
            for n in self.neighbors(node):
                # fastest way I could think of..
                edge = self[node][n].copy()
                self.remove_edge(node, n)
                cycle = []
                try:
                    cycle = list(nx.all_shortest_paths(self, node, n))
                except nx.exception.NetworkXNoPath:
                    pass
                self.add_edge(node, n, **edge)
                #FIXME MW edit to only store cycles < len(20)
                # should be a harmless edit but maybe need to test
                if(len(cycle) <= 10):
                    cycles += cycle

        for label, data in self.nodes_iter(data=True):
            # N O C S
            neighbours = self.neighbors(label)
            element = data['element']
            if element == "C":
                if len(neighbours) >= 4:
                    self.node[label].update({'hybridization':'sp3'})
                elif len(neighbours) == 3:
                    self.node[label].update({'hybridization':'sp2'})
                elif len(neighbours) <= 2:
                    self.node[label].update({'hybridization':'sp'})
            elif element == "N":
                if len(neighbours) >= 3:
                    self.node[label].update({'hybridization':'sp3'})
                elif len(neighbours) == 2:
                    self.node[label].update({'hybridization':'sp2'})
                elif len(neighbours) == 1:
                    self.node[label].update({'hybridization':'sp'})
                else:
                    self.node[label].update({'hybridization':'sp3'})
            elif element == "O":
                if len(neighbours) >= 2:
                    n_elems = set([self.node[k]['element'] for k in neighbours])
                    # if O is bonded to a metal, assume sp2 - like ... 
                    # there's probably many cases where this fails,
                    # but carboxylate groups, bridging hydroxy groups
                    # make this true.
                    if (n_elems <= metals):
                        self.node[label].update({'hybridization':'sp2'})
                    else:
                        self.node[label].update({'hybridization':'sp3'})
                elif len(neighbours) == 1:
                    self.node[label].update({'hybridization':'sp2'})
                else:
                    # If it has no neighbours, just give it SP3
                    self.node[label].update({'hybridization':'sp3'})
            elif element == "S":
                if len(neighbours) >= 2:
                    self.node[label].update({'hybridization':'sp3'})
                elif len(neighbours) == 1:
                    self.node[label].update({'hybridization':'sp2'})
                else:
                    self.node[label].update({'hybridization':'sp3'})

            else:
                #default sp3
                self.node[label].update({'hybridization':'sp3'})
        # convert to aromatic
        # probably not a good test for aromaticity..
        arom = set(["C", "N", "O", "S"])
        # NOTE MW adding Si so that cycles are identified in zeolites
        arom_zeo = set(["O", "S", "Si"])
        for cycle in cycles:
            elements = [self.node[k]['element'] for k in cycle]
            neigh = [self.degree(k) for k in cycle]
            if np.all(np.array(neigh) <= 3) and set(elements) <= arom:
                for a in cycle:
                    self.node[a]['hybridization'] = 'aromatic'
                    self.node[a]['cycle'] = True
                    self.node[a]['rings'].append(cycle)
            # also need to check about zeolite rings
            elif np.all(np.array(neigh) <= 4) and set(elements) <= arom_zeo:
                for a in cycle:
                    self.node[a]['hybridization'] = 'aromatic'
                    self.node[a]['cycle'] = True
                    self.node[a]['rings'].append(cycle)

    def compute_bond_typing(self):
        """ Compute bond types and atom types based on the local edge
        environment.
        Messy, loads of 'ifs'
        is there a better way to catch chemical features?
        """ 
        #TODO(pboyd) return if bonds already 'typed' in the .cif file
        double_check = [] 
        for n1, n2, data in self.edges_iter2(data=True):
            elements = [self.node[a]['element'] for a in (n1,n2)]
            hybridization = [self.node[a]['hybridization'] for a in (n1, n2)]
            rings = [self.node[a]['rings'] for a in (n1, n2)]
            samering = False

            if set(hybridization) == set(['aromatic']):
                for r in rings[0]:
                    if n2 in r:
                        samering = True
                if(samering):
                    data.update({"order" : 1.5})

            if set(elements) == set(["C", "O"]):
                car = n1 if self.node[n1]['element'] == "C" else n2
                car_data = self.node[car]
                oxy = n2 if self.node[n2]['element'] == "O" else n1
                oxy_data = self.node[oxy]

                carnn = [i for i in self.neighbors(car) if i != oxy]
                try:
                    carnelem = [self.node[j]['element'] for j in carnn]
                except:
                    carnelem = []

                oxynn = [i for i in self.neighbors(oxy) if i != car]
                try:
                    oxynelem = [self.node[j]['element'] for j in oxynn]
                except:
                    oxynelem = []
                if "O" in carnelem:
                    at = carnn[carnelem.index("O")]
                    at_data = self.node[at]
                    if self.degree(at) == 1:
                        if self.degree(oxy) == 1:
                            #CO2
                            car_data['hybridization'] = 'sp'
                            oxy_data['hybridization'] = 'sp2'
                            data['order'] = 2.
                        else:
                            # ester
                            if set(oxynelem) <= organic:
                                car_data['hybridization'] = 'sp2'
                                oxy_data['hybridization'] = 'sp2'
                                data['order'] = 1 # this is the ether part of an ester... 
                            #carboxylate?
                            else:
                                car_data['hybridization'] = 'aromatic'
                                oxy_data['hybridization']= 'aromatic'
                                data['order'] = 1.5

                    else:
                        atnelem = [self.node[k]['element'] for k in self.neighbors(at)]
                        if (set(atnelem) <= organic):
                            # ester
                            if len(oxynn) == 0:
                                car_data['hybridization'] = 'sp2'
                                oxy_data['hybridization'] = 'sp2'
                                data['order'] = 2. # carbonyl part of ester
                            # some kind of resonance structure?
                            else:
                                car_data['hybridization'] = 'aromatic'
                                oxy_data['hybridization'] = 'aromatic'
                                data['order'] = 1.5
                        else:
                            car_data['hybridization'] = 'aromatic'
                            oxy_data['hybridization'] = 'aromatic'
                            data['order'] = 1.5

                if "N" in carnelem:
                    at = carnn[carnelem.index("N")]
                    # C=O of amide group
                    if self.degree(oxy) == 1:
                        data['order'] = 1.5
                        car_data['hybridization'] = 'aromatic'
                        oxy_data['hybridization'] = 'aromatic'
                # only one carbon oxygen connection.. could be C=O, R-C-O-R, R-C=O-R
                if (not "O" in carnelem) and (not "N" in carnelem):
                    if len(oxynn) > 0:
                        # ether
                        oxy_data['hybridization'] = 'sp3'
                        data['order'] = 1.0
                    else:
                        if car_data['cycle'] and car_data['hybridization'] == 'aromatic':
                            oxy_data['hybridization'] = 'aromatic'
                            data['order'] = 1.5
                        # carbonyl
                        else:
                            oxy_data['hybridization'] = 'sp2'
                            data['order'] = 2.0
            elif set(elements) == set(["C", "N"]) and not samering:
                car = n1 if self.node[n1]['element'] == "C" else n2
                car_data = self.node[car]
                nit = n2 if self.node[n2]['element'] == "N" else n1
                nit_data = self.node[nit]
                carnn = [j for j in self.neighbors(car) if j != nit]
                carnelem = [self.node[k]['element'] for k in carnn]
                nitnn = [j for j in self.neighbors(nit) if j != car]
                nitnelem = [self.node[k]['element'] for k in nitnn]
                # aromatic amine connected -- assume part of delocalized system
                if car_data['hybridization'] == 'aromatic' and set(['H']) == set(nitnelem):
                    data['order'] = 1.5
                    nit_data['hybridization'] = 'aromatic'
                # amide?
                elif len(self.neighbors(car)) == 3 and len(nitnn) >=2:
                    if "O" in carnelem:
                        data['order'] = 1.5 # (amide)
                        nit_data['hybridization'] = 'aromatic'
                    # nitro
                    if set(nitnelem) == set(["O"]):
                        data['order'] = 1. 
                        nit_data['hybridization'] = 'aromatic'
                        for oatom in nitnn:
                            nobond = self[nit][oatom]['order'] = 1.5
                            self.node[oatom]['hybridization'] = 'aromatic'

            elif (not self.node[n1]['cycle']) and (not self.node[n2]['cycle']) and (set(elements) <= organic):
                if set(hybridization) == set(['sp2']):
                    try:
                        cr1 = COVALENT_RADII['%s_2'%elements[0]]
                    except KeyError:
                        cr1 = COVALENT_RADII[elements[0]]
                    try:
                        cr2 = COVALENT_RADII['%s_2'%(elements[1])]
                    except KeyError:
                        cr2 = COVALENT_RADII[elements[1]]
                    covrad = cr1 + cr2
                    # first pass: assign all to 2.0 bond order
                    data['order'] = 2.0
                    double_check += [n1, n2]
                    #if (data['length'] <= covrad*.95):
                    #    data['order'] = 2.0
                elif set(hybridization) == set(['sp']):
                    try:
                        cr1 = COVALENT_RADII['%s_1'%elements[0]]
                    except KeyError:
                        cr1 = COVALENT_RADII[elements[0]]
                    try:
                        cr2 = COVALENT_RADII['%s_1'%elements[1]]
                    except KeyError:
                        cr2 = COVALENT_RADII[elements[1]]
                    # first pass: assign all to 3.0 bond order
                    double_check += [n1, n2]
                    data['order'] = 3.0
                    #covrad = cr1 + cr2 
                    #if (data['length'] <= covrad*.95):
                    #    data['order'] = 3.0
        # second pass, check organic unsaturated bonds to make
        # sure alkyl chains are alternating etc.
        while double_check:
            n = double_check.pop()
            # rewind this atom to a 'terminal' connected atom
            for i in self.recurse_bonds_to_end(n, pool=[], visited=[]):
                start = i
                try:
                    idn = double_check.index(i)
                    del double_check[idn]
                except ValueError:
                    pass
            # iterate over all linear chains
            # BE CAREFUL about overwriting bond orders here, the recursion
            # can have duplicate bonds for each 'k' iteration since it iterates over
            # all possible linear chains. So if the molecule is branched, there
            # will be multiple recursions over the same set of bonds.
            for k in self.recurse_linear_chains(start, visited=[], excluded=[]):
                bond_orders = []
                # first pass, store all the bond orders
                for idx in range(len(k)-1):
                    n1 = k[idx]
                    n2 = k[idx+1]
                    bond = self[n1][n2]
                    bond_orders.append(bond['order'])
                # second pass, check continuity
                for idx in range(len(k)-1):
                    n1 = k[idx]
                    n2 = k[idx+1]
                    data1 = self.node[n1]
                    data2 = self.node[n2]
                    
                    hyb1 = data1['hybridization']
                    hyb2 = data2['hybridization']

                    elem1 = data1['element']
                    elem2 = data2['element']
                    if(idx == 0):
                        order = bond_orders[idx]
                        next_order = bond_orders[idx+1]
                        if (hyb1 == 'sp2') and (hyb2 == 'sp2'):
                            bond_orders[idx] = 2.
                    elif (idx < len(k)-2):
                        prev_order = bond_orders[idx-1]
                        next_order = bond_orders[idx+1]
                        order = bond_orders[idx]
                        if (hyb1 == 'sp2') and (hyb2 == 'sp2'):
                            if (prev_order == 2.) and (next_order == 1):
                                bond_orders[idx-1] = 1.5
                                bond_orders[idx] = 1.5
                            elif (prev_order == 2.) and (next_order == 2.):
                                bond_orders[idx] = 1.
                        elif (hyb1 == 'sp') and (hyb2 == 'sp'):
                            if (prev_order == 3.) and (next_order == 3.):
                                bond_orders[idx] = 1.
                    else:
                        prev_order = bond_orders[idx-1]
                        order = bond_orders[idx]
                        if (hyb1 == 'sp2') and (hyb2 == 'sp2'):
                            if (prev_order == 2.) and (order == 2):
                                if set([elem1, elem2]) == set(["C", "O"]):
                                    onode = n2 if self.node[n2]['element'] == "O" else n1
                                    # this very specific case is a enol
                                    if self.degree(onode) == 1:
                                        bond_orders[idx] = 1.5
                                        bond_orders[idx-1] = 1.5
                        elif (hyb1 == 'sp') and (hyb2 == 'sp'):
                            if (prev_order == 3.):
                                bond_orders[idx] = 1.

                for idx in range(len(k)-1):
                    n1 = k[idx]
                    n2 = k[idx+1]
                    bond = self[n1][n2]
                    # update bond orders.
                    bond['order'] = bond_orders[idx]

                #print([self.node[r]['element'] for r in k])
    
    def recurse_linear_chains(self, node, visited=[], excluded=[]):
        """Messy recursion function to return all unique chains from a set of atoms between two 
        metals (or terminal atoms in the case of molecules)"""
        if self.node[node]['element'] == 'H':
            yield
        neighbors = [i for i in self.neighbors(node) if i not in excluded and self.node[i]['element'] != "H"]
        if (not neighbors) and (node in excluded) and (not visited):
            return
        elif (not neighbors) and (node in excluded):
            nde = visited.pop()
        elif (not neighbors) and (not (node in excluded)):
            excluded.append(node)
            visited.append(node)
            yield visited
            nde = visited.pop()
        else:
            excluded.append(node)
            visited.append(node)
            nde = neighbors[0]
        for x in self.recurse_linear_chains(nde, visited, excluded):
            yield x

    def recurse_bonds_to_end(self, node, pool=[], visited=[]):
        if self.node[node]['element'] == 'H':
            return
        visited.append(node)
        neighbors = [i for i in self.neighbors(node) if i not in visited and self.node[i]['element'] != "H"]
        pool += neighbors
        yield node
        if (not pool) or (self.node[node]['element'] in list(metals)):
            return
        for x in self.recurse_bonds_to_end(pool[0], pool[1:], visited):
            yield x

    def atomic_node_sanity_check(self):
        """Check for specific keyword/value pairs. Exit if non-existent"""

    def compute_angles(self):
        """angles are attached to specific nodes, this way
        if a node is cut out of a graph, the angle comes with it.

               
               b-----a
              /
             /
            c
               
        Must be updated with different adjacent nodes if a
        supercell is requested, and the angle crosses a 
        periodic image.
        
        """
        for b, data in self.nodes_iter(data=True):
            if self.degree(b) < 2:
                continue
            angles = itertools.combinations(self.neighbors(b), 2)
            for (a, c) in angles:
                data.setdefault('angles', {}).update({(a,c):{'potential':None}})
    
    def compute_dihedrals(self):
        """Dihedrals are attached to specific edges in the graph.
           a
            \ 
             b -- c
                   \ 
                    d

        the edge between b and c will contain all possible dihedral
        angles between the neighbours of b and c (this includes a
        and d and other possible bonded atoms)

        """
        for b, c, data in self.edges_iter2(data=True):
            b_neighbours = [k for k in self.neighbors(b) if k != c]
            c_neighbours = [k for k in self.neighbors(c) if k != b]
            for a in b_neighbours:
                for d in c_neighbours:
                    data.setdefault('dihedrals',{}).update({(a, d):{'potential':None}})
    
    def compute_improper_dihedrals(self):
        """Improper Dihedrals are attached to specific nodes in the graph.
           a
            \ 
             b -- c
             |     
             d    

        the node b will contain all possible improper dihedral 
        angles between the neighbours of b 

        """
        for b, data in self.nodes_iter(data=True):
            if self.degree(b) != 3:
                continue
            # three improper torsion angles about each atom
            local_impropers = list(itertools.permutations(self.neighbors(b)))
            for idx in range(0, 6, 2):
                (a, c, d) = local_impropers[idx]
                data.setdefault('impropers',{}).update({(a,c,d):{'potential':None}})

    def compute_topology_information(self, cell, tol, num_neighbours):
        self.compute_cartesian_coordinates(cell)
        self.compute_min_img_distances(cell)
        self.compute_bonding(cell)
        self.compute_init_typing()
        self.compute_bond_typing()
        if (self.find_metal_sbus):
            self.detect_clusters(num_neighbours, tol) # num neighbors determines how many nodes from the metal element to cut out for comparison 
        if (self.find_organic_sbus):
            self.detect_clusters(num_neighbours, tol,  type="Organic")
        self.compute_angles()
        self.compute_dihedrals()
        self.compute_improper_dihedrals()

    def sorted_node_list(self):
        return [n[1] for n in sorted([(data['index'], node) for node, data in self.nodes_iter(data=True)])]

    def sorted_edge_list(self): 
        return [e[1] for e in sorted([(data['index'], (n1, n2)) for n1, n2, data in self.edges_iter2(data=True)])]

    def show(self):
        nx.draw(self)

    def img_offset(self, cells, cell, maxcell, flag, redefine, n1=0):
        unit_repr = np.array([5, 5, 5], dtype=int)
        if(flag == '.'):
            return cells.index(tuple([tuple([i]) for i in cell]))
        translation = np.array([int(j) for j in flag[2:]]) - unit_repr
        ocell = np.array(cell + translation, dtype=np.float64)

        # have to find the off-diagonal values, and what their
        # multiples are to determine the image cell of this
        # bond.
        imgcell = ocell % maxcell
        if redefine is None:
            return cells.index(tuple([tuple([i]) for i in imgcell]))
        
        olde_imgcell = imgcell
        newcell = cell + np.dot(cell, redefine)%maxcell
        newocell = (newcell + translation) #% maxcell
        rd2 = redefine - np.identity(3)*maxcell
        # check if the newcell translation spans a periodic boundary
        if (np.any(newocell >= maxcell, axis=0)) or (np.any(newocell < 0.)):
            
            # get indices of where the newocell is spanning a pbc
            indexes = np.where(newocell - maxcell >= 0)[0].tolist() + np.where(newocell < 0)[0].tolist()

            # determine if the translation spans a lattice vector which is a linear combination
            # of the other vectors, in which case, make a correction in the image cell
            # this is checked by finding which lattice indices are spanning the cell, then
            # checking if these vectors are a linear combination of the other ones by
            # summing the columns together.
            # this ONLY works if b and c are lin combs of a and a is the unit vector
            newrd = np.sum(rd2[indexes], axis=0)%maxcell
            #newrd = np.dot(newocell, redefine)%maxcell
            if np.any(newrd != 0):
                imgcell = (newocell - np.dot(newocell, redefine)) % maxcell

        return cells.index(tuple([tuple([i]) for i in imgcell]))

    def update_symflag(self, cell, symflag, mincell, maxcell):
        unit_repr = np.array([5, 5, 5], dtype=int)
        ocell = cell + np.array([int(j) for j in symflag[2:]]) - unit_repr
        imgcell = ocell % maxcell
        if any(ocell < mincell) or any(ocell >= maxcell):
            newflaga = np.array([5,5,5])
            newflaga[np.where(ocell >= maxcell)] = 6
            newflaga[np.where(ocell < np.zeros(3))] = 4
            newflag = "1_%i%i%i"%(tuple(newflaga))
          
        else:
            newflag = '.'
        return newflag
    
    def correspondence_graph(self, graph, tol, general_metal=False, node_subset=None):
        """Generate a correspondence graph between the nodes
        and the SBU.
        tolerance is the distance tolerance for the edge generation 
        in the correspondence graph.

        """
        if node_subset is None:
            node_subset = self.nodes()
        graph_nodes = graph.nodes()
        cg = nx.Graph()
        # add nodes to cg 
        for (i, j) in itertools.product(node_subset, graph_nodes):
            # match element-wise
            elementi = self.node[i]['element']
            elementj = graph.node[j]['element']
            match = False
            if general_metal:
                if ATOMIC_NUMBER.index(elementi) in METALS and ATOMIC_NUMBER.index(elementj) in METALS:
                    match = True
                elif elementi == elementj:
                    match = True
            elif elementi == elementj:
                match = True
            if match:
                cg.add_node((i,j))
        # add edges to cg
        for (a1, b1), (a2, b2) in itertools.combinations(cg.nodes(), 2):
            if (a1 != a2) and (b1 != b2):
                da = self.distance_matrix[a1-1, a2-1]
                db = graph.distance_matrix[b1-1, b2-1]
                if np.allclose(da, db, atol=tol):
                    cg.add_edge((a1,b1), (a2,b2))
        return cg

    def detect_clusters(self, num_neighbors, tol, type='Inorganic', general_metal=False):
        """Detect clusters such as the copper paddlewheel using
        maximum clique detection. This will assign specific atoms
        with a special flag for use when building their force field.

        setting general_metal to True will allow for cluster recognition of 
        inorganic SBUs while ignoring the specific element type of the metal,
        so long as it is a metal, it will be paired with other metals.
        This may increase the time for SBU recognition.

        """
        print("Detecting %s clusters"%type)

        reference_nodes = []
        
        if type=="Inorganic":
            types = InorganicCluster.keys()
            ref_sbus = InorganicCluster
            store_sbus = self.inorganic_sbus
        elif type == "Organic":
            types = OrganicCluster.keys()
            ref_sbus = OrganicCluster
            store_sbus = self.organic_sbus

        for node, data in self.nodes_iter(data=True):
            if (type=='Inorganic') and (general_metal) and (data['atomic_number'] in METALS)\
                    and ('special' not in data.keys()): # special means that this atom has already been found in a previous clique detection
                reference_nodes.append(node) 

            elif (data['element'] in types) and ('special' not in data.keys()): 
                reference_nodes.append(node)
        
        no_cluster = []
        #FIXME(pboyd): This routine doesn't work for finding 1-D rod SBUs.
        # At each step the atoms found that belong to an SBU are deleted
        # to improve the search efficiency. In 1-D rod SBUs there are 
        # repeating elements in the 'discretized' version used to discover
        # the rod in a MOF, so in some cases only a fragment of the rod
        # is found. This results in the wrong force field types being
        # assigned to these atoms (UFF4MOF).
        #for node in reference_nodes:
        while reference_nodes:
            node = reference_nodes.pop() 
            data = self.node[node]
            possible_clusters = {}
            toln = tol
            if type=="Inorganic" and general_metal:
                for j in ref_sbus.keys():
                    possible_clusters.update(ref_sbus[j])
            else:
                possible_clusters.update(ref_sbus[data['element']])
            try:
                neighbour_nodes = [] 
                instanced_neighbours = self.neighbors(node)
                if (data['element'] == "C"):
                    chk_neighbors = num_neighbors # organic clusters can be much bigger.
                else:
                    chk_neighbors = num_neighbors
                # tree-like spanning of original node
                for j in range(chk_neighbors):
                    temp_neighbours = []
                    for n in instanced_neighbours:
                        if('special' not in self.node[n].keys()):
                            neighbour_nodes.append(n)
                            temp_neighbours += [j for j in self.neighbors(n) if j not in neighbour_nodes]
                    instanced_neighbours = temp_neighbours
                cluster_found = False
                # sort by descending number of nodes, this will ensure the largest SBU will be found
                # instead of a collection of smaller ones (e.g. multiple aromatic rings).
                clusters = [(cluster.number_of_nodes(), name, cluster) for name,cluster in possible_clusters.items()]
                for count, name, cluster in list(reversed(sorted(clusters))):
                    # ignore if it is impossible to find a clique with the current cluster.
                    if (len(neighbour_nodes)+1) < cluster.number_of_nodes():
                        continue
                    cg = self.correspondence_graph(cluster, toln, general_metal=general_metal, node_subset=neighbour_nodes + [node])
                    cliques = nx.find_cliques(cg)
                    #cliques = approximation.clique.max_clique(cg)
                    for clique in cliques:
                        #print(len(clique), cluster.number_of_nodes())
                        if len(clique) == cluster.number_of_nodes():
                            # found cluster
                            # update the 'hybridization' data
                            for i,j in clique:
                                self.node[i]['special_flag'] = cluster.node[j]['special_flag']
                            cluster_found = True
                            print("Found %s"%(name))
                            store_sbus.setdefault(name, []).append([i for (i,j) in clique])
                            break

                    if(cluster_found):
                        for n in neighbour_nodes:
                            try:
                                reference_nodes.pop(reference_nodes.index(n))
                            except:
                                pass
                        break
                    else:
                        # put node back into the pool
                        reference_nodes.append(node)
                if not (cluster_found):
                    no_cluster.append(data['element'])
            except KeyError:
                # no recognizable clusters for element
                no_cluster.append(data['element'])
        for j in set(no_cluster):
            print ("No recognizable %s clusters for %i elements %s"%(type.lower(), no_cluster.count(j),  j))

    def redefine_lattice(self, redefinition, lattice):
        """Redefines the lattice based on the old lattice vectors. This was designed to convert
        non-orthogonal cells to orthogonal boxes, but it could in principle be used to 
        convert any cell to any other cell. (As long as the redefined lattice
        are integer multiples of the old vectors)

        """
        #print(redefinition)
        #redefinition = np.array([[1., 0., 0.], [ -2., 2.,0.], [ -1., 3., 2.]])
        # determine how many replicas of the atoms is necessary to produce the supercell.
        vol_change = np.prod(np.diag(redefinition))
        if vol_change > 20:
            print("ERROR: The volume change is %i times greater than the unit cell. "%(vol_change) +
                    "I cannot process structures of this size! I am making a non-orthogonal simulation.")
            #sys.exit()
            return
        
        print("The redefined cell will be %i times larger than the original."%(int(vol_change)))

        # replicate supercell
        sc = (tuple([int(i) for i in np.diag(redefinition)]))
        self.build_supercell(sc, lattice, redefine=redefinition)
        # re-define the cell
        old_cell = np.multiply(self.cell._cell.T, sc).T
        self.cell.set_cell(np.dot(redefinition, self.cell._cell))
        # the node cartesian_coordinates must be shifted by the periodic boundaries.
        for node, data in self.nodes_iter(data=True):
            coord = data['cartesian_coordinates']
            data['cartesian_coordinates'] = self.in_cell(coord) 

        # the bonds which span a periodic boundary will change
        for n1, n2, data in self.edges_iter2(data=True):
            flag = self.compute_bond_image_flag(n1, n2, self.cell)
            data['symflag'] = flag #'.' 

        # not sure what this may break, but have to assume this new cell is the 'original'
        self.store_original_size()

    def build_supercell(self, sc, lattice, track_molecule=False, molecule_len=0, redefine=None):
        """Construct a graph with nodes supporting the size of the 
        supercell (sc)
        Oh man.. so ugly.        
        NB: this replaces and overwrites the original unit cell data 
            with a supercell. There may be a better way to do this 
            if one needs to keep both the super- and unit cells.
        """
        # preserve indices across molecules.
        unitatomlen = self.original_size
        totatomlen = nx.number_of_nodes(self)
        # keep a numerical index of the nodes.. this is to make sure that the molecules
        # are kept in their positions in the supercell (if replicated)
        unit_node_ids = sorted(self.nodes())
        origincell = np.array([0., 0., 0.])
        cells = list(itertools.product(*[itertools.product(range(j)) for j in sc]))
        maxcell = np.array(sc)
        rem_edges = []
        add_edges = []
        union_graphs = []
        orig_copy = deepcopy(self)
        for count, cell in enumerate(cells):
            newcell = np.array(cell).flatten()
            offset = count * unitatomlen
            mol_offset = count * molecule_len

            cartesian_offset = np.dot(newcell, lattice.cell)
            # Initial setup of new image in the supercell.
            if (count == 0):
                graph_image = self
            else:
                # rename nodes
                graph_image = nx.relabel_nodes(deepcopy(orig_copy), {unit_node_ids[i-1]: offset+unit_node_ids[i-1] for i in range(1, totatomlen+1)})
                graph_image.sorted_edge_dict = self.sorted_edge_dict.copy()
                # rename the edges with the offset associated with this supercell.
                for k,v in list(graph_image.sorted_edge_dict.items()):
                    newkey = (k[0] + offset, k[1] + offset) 
                    newval = (v[0] + offset, v[1] + offset)
                    del graph_image.sorted_edge_dict[k]
                    graph_image.sorted_edge_dict.update({newkey:newval})

            # keep track of original index value from the unit cell.
            for i in range(1, totatomlen+1):
                graph_image.node[unit_node_ids[i-1]+offset]['image'] = unit_node_ids[i-1]
            if track_molecule:
                self.molecule_images.append(graph_image.nodes())
                graph_image.molecule_id = orig_copy.molecule_id + mol_offset
            # update cartesian coordinates for each node in the image
            for node, data in graph_image.nodes_iter(data=True):
                n_orig = data['image']
                if track_molecule:
                    data['molid'] = graph_image.molecule_id
                data['cartesian_coordinates'] = data['cartesian_coordinates'] + cartesian_offset
                # update all angle and improper terms to the curent image indices. Dihedrals will be done in the edge loop
                # angle check
                try:
                    for (a, c), val in list(data['angles'].items()):
                        aid, cid = offset + a, offset + c
                        e_ba = graph_image[node][aid]
                        e_bc = graph_image[node][cid]
                        ba_symflag = e_ba['symflag']
                        order_ba = graph_image.sorted_edge_dict[(aid, node)]
                        if order_ba != (node, aid) and e_ba['symflag'] != '.':
                            ba_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in e_ba['symflag'][2:]])))
                        bc_symflag = e_bc['symflag']
                        order_bc = graph_image.sorted_edge_dict[(node, cid)]
                        if order_bc != (node, cid) and e_bc['symflag'] != '.':
                            bc_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in e_bc['symflag'][2:]]))) 
                        os_a = self.img_offset(cells, newcell, maxcell, ba_symflag, redefine) * unitatomlen
                        os_c = self.img_offset(cells, newcell, maxcell, bc_symflag, redefine) * unitatomlen
                        data['angles'].pop((a,c))
                        data['angles'][(a + os_a, c + os_c)] = val

                except KeyError:
                    # no angles for n1
                    pass
                # improper check
                try:
                    for (a, c, d), val in list(data['impropers'].items()):
                        aid, cid, did = offset + a, offset + c, offset + d
                        e_ba = graph_image[node][aid]
                        order_ba = graph_image.sorted_edge_dict[(node, aid)]
                        ba_symflag = e_ba['symflag']
                        e_bc = graph_image[node][cid]
                        order_bc = graph_image.sorted_edge_dict[(node, cid)]
                        bc_symflag = e_bc['symflag']
                        e_bd = graph_image[node][did]
                        order_bd = graph_image.sorted_edge_dict[(node, did)]
                        bd_symflag = e_bd['symflag']
                        if order_ba != (node, aid) and e_ba['symflag'] != '.':
                            ba_symflag = "1_%i%i%i"%(tuple([10 - int(j) for j in e_ba['symflag'][2:]]))
                        if order_bc != (node, cid) and e_bc['symflag'] != '.':
                            bc_symflag = "1_%i%i%i"%(tuple([10 - int(j) for j in e_bc['symflag'][2:]]))
                        if order_bd != (node, did) and e_bd['symflag'] != '.':
                            bd_symflag = "1_%i%i%i"%(tuple([10 - int(j) for j in e_bd['symflag'][2:]])) 

                        os_a = self.img_offset(cells, newcell, maxcell, ba_symflag, redefine) * unitatomlen
                        os_c = self.img_offset(cells, newcell, maxcell, bc_symflag, redefine) * unitatomlen
                        os_d = self.img_offset(cells, newcell, maxcell, bd_symflag, redefine) * unitatomlen
                        data['impropers'].pop((a,c,d))
                        data['impropers'][(a + os_a, c + os_c, d + os_d)] = val

                except KeyError:
                    # no impropers for n1
                    pass
            
            # update nodes and edges to account for bonding to periodic images.
            #unique_translations = {}
            for n1, n2, data in graph_image.edges_iter2(data=True):
                # flag boundary crossings, and determine updated nodes.
                # check symmetry flags if they need to be updated,
                n1_data = graph_image.node[n1]
                n2_data = graph_image.node[n2]
                try:
                    n1_orig = n1_data['image']
                    n2_orig = n2_data['image']
                except KeyError:
                    n1_orig = n1
                    n2_orig = n2
                # TODO(pboyd) the data of 'rings' for each node is not updated, do so if needed..
                # update angle, dihedral, improper indices.
                # dihedrals are difficult if the edge spans one of the terminal atoms..

                if (data['symflag'] != '.'):
                    # DEBUGGING
                    unit_repr = np.array([5, 5, 5], dtype=int)
                    translation = tuple(np.array([int(j) for j in data['symflag'][2:]]) - unit_repr)
                    #unique_translations.setdefault(translation,0)
                    #unique_translations[translation] += 1
                    # DEBUGGING
                    os_id = self.img_offset(cells, newcell, maxcell, data['symflag'], redefine, n1) 
                    offset_c = os_id * unitatomlen
                    img_n2 = offset_c + n2_orig
                    #if (n1 == 1712):
                    #    print(os_id)
                    #    print(newcell)
                    #    print(redefine)
                    # pain...
                    opposite_flag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in data['symflag'][2:]]))) 
                    rev_n1_img = self.img_offset(cells, newcell, maxcell, opposite_flag, redefine) * unitatomlen + n1_orig 
                    # dihedral check
                    try:
                        for (a, d), val in list(data['dihedrals'].items()):
                            # check to make sure edge between a, n1 is not crossing an image
                            edge_n1_a = orig_copy[n1_orig][a]
                            order_n1_a = graph_image.sorted_edge_dict[(n1, a+offset)]
                            n1a_symflag = edge_n1_a['symflag']

                            edge_n2_d = orig_copy[n2_orig][d]
                            order_n2_d = graph_image.sorted_edge_dict[(n2, d+offset)]
                            n2d_symflag = edge_n2_d['symflag']

                            offset_a = offset
                            if order_n1_a != (n1, a+offset) and edge_n1_a['symflag'] != '.':
                                n1a_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n1_a['symflag'][2:]])))
                            if (edge_n1_a['symflag'] != '.'):
                                offset_a = self.img_offset(cells, newcell, maxcell, n1a_symflag, redefine) * unitatomlen
                            # check to make sure edge between n2, c is not crossing an image
                            offset_d = offset_c

                            if order_n2_d != (n2, d+offset) and edge_n2_d['symflag'] != '.':
                                n2d_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n2_d['symflag'][2:]])))
                            if (edge_n2_d['symflag'] != '.'):
                                offset_d = self.img_offset(cells, np.array(cells[os_id]).flatten(), maxcell, n2d_symflag, redefine) * unitatomlen

                            aid, did = offset_a + a, offset_d + d
                            copyover = data['dihedrals'].pop((a,d))
                            data['dihedrals'][(aid, did)] = copyover
                    except KeyError:
                        # no dihedrals here.
                        pass

                    # Update symmetry flag of bond
                    data['symflag'] = self.update_symflag(newcell, data['symflag'], origincell, maxcell)
                    add_edges += [((n1, img_n2),data)]
                    rem_edges += [(n1, n2)]
                else:
                    # dihedral check
                    try:
                        for (a, d), val in list(data['dihedrals'].items()):
                            # check to make sure edge between a, n1 is not crossing an image
                            edge_n1_a = orig_copy[n1_orig][a] 
                            order_n1_a = graph_image.sorted_edge_dict[(n1, a+offset)]
                            n1a_symflag = edge_n1_a['symflag']

                            edge_n2_d = orig_copy[n2_orig][d]
                            order_n2_d = graph_image.sorted_edge_dict[(n2, d+offset)]
                            n2d_symflag = edge_n2_d['symflag']

                            offset_a = offset
                            if order_n1_a != (n1, a+offset) and edge_n1_a['symflag'] != '.':
                                n1a_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n1_a['symflag'][2:]])))
                            if (edge_n1_a['symflag'] != '.'):
                                offset_a = self.img_offset(cells, newcell, maxcell, n1a_symflag, redefine) * unitatomlen
                            # check to make sure edge between n2, c is not crossing an image
                            offset_d = offset

                            if order_n2_d != (n2, d+offset) and edge_n2_d['symflag'] != '.':
                                n2d_symflag = "1_%i%i%i"%(tuple(np.array([10,10,10]) - np.array([int(j) for j in edge_n2_d['symflag'][2:]])))
                            if (edge_n2_d['symflag'] != '.'):
                                offset_d = self.img_offset(cells, newcell, maxcell, n2d_symflag, redefine) * unitatomlen

                            aid, did = offset_a + a, offset_d + d
                            copyover = data['dihedrals'].pop((a,d))
                            data['dihedrals'][(aid, did)] = copyover
                    except KeyError:
                        # no dihedrals here.
                        pass
            if (count > 0):
                union_graphs.append(graph_image)
        for G in union_graphs:
            for node, data in G.nodes_iter(data=True):
                self.add_node(node, **data)
           #once nodes are added, add edges.
        for G in union_graphs:
            self.sorted_edge_dict.update(G.sorted_edge_dict)
            for (n1, n2, data) in G.edges_iter2(data=True):
                self.add_edge(n1, n2, **data)

        for (n1, n2) in rem_edges:
            self.remove_edge(n1, n2)
            self.sorted_edge_dict.pop((n1,n2))
            self.sorted_edge_dict.pop((n2,n1))
        for (n1, n2), data in add_edges:
            self.add_edge(n1, n2, **data)
            self.sorted_edge_dict.update({(n1,n2):(n1,n2)})
            self.sorted_edge_dict.update({(n2,n1):(n1,n2)})
        #print(list(unique_translations.keys()))
    def unwrap_node_coordinates(self, cell):
        """Must be done before supercell generation.
        This is a recursive method iterating over all edges.
        The design is totally unpythonic and 
        written in about 5 mins.. so be nice (PB)
        
        """
        supercells = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
        # just use the first node as the unwrapping point..
        # probably a better way to do this to keep most atoms in the unit cell,
        # but I don't think it matters too much.
        nodelist = self.nodes()
        n1 = nodelist[0] 
        queue = []
        while (nodelist or queue):
            for n2, data in self[n1].items():
                if n2 not in queue and n2 in nodelist:
                    queue.append(n2)
                    coord1 = self.node[n1]['cartesian_coordinates'] 
                    coord2 = self.node[n2]['cartesian_coordinates']
                    fcoords = np.dot(cell.inverse, coord2) + supercells
                    
                    coords = np.array([np.dot(j, cell.cell) for j in fcoords])
                    
                    dists = distance.cdist([coord1], coords)
                    dists = dists[0].tolist()
                    image = dists.index(min(dists))
                    self.node[n2]['cartesian_coordinates'] += np.dot(supercells[image], cell.cell)
                    data['symflag'] = '.'
            del nodelist[nodelist.index(n1)]
            try:
                n1 = queue[0]
                queue = queue[1:]

            except IndexError:
                pass

    def store_original_size(self):
        self.original_size = self.number_of_nodes()

    def __iadd__(self, newgraph):
        self.sorted_edge_dict.update(newgraph.sorted_edge_dict)
        for n, data in newgraph.nodes_iter(data=True):
            self.add_node(n, **data)
        for n1,n2, data in newgraph.edges_iter2(data=True):
            self.add_edge(n1,n2, **data)
        return self

    def __or__(self, graph):
        if (len(graph.nodes()) == 1) and len(self.nodes()) == 1:
            return list([0]) 
        cg = self.correspondence_graph(graph, 0.4)
        cliques = list(nx.find_cliques(cg))
        cliques.sort(key=len)
        return cliques[-1]

    def to_openbabel_structure(self):
        try:
            obstruct = ob.OBMol()
        except NameError:
            # openbabel could not be imported.
            return
        # add atoms and bonds one-by-one
        for node in self.nodes_iter():
            # shift by periodic boundaries??
            a = obstruct.NewAtom()
            a.SetAtomicNum(self.node[node]['atomic_number'])
            a.SetVector(*self.node[node]['cartesian_coordinates'])
        for (n1, n2, data) in self.edges_iter2(data=True):
            obstruct.AddBond(n1, n2, data['order'])

        return obstruct

# END MolecularGraph class

# New class to extend MolecularGraph and turn it into a slab
class SlabGraph(MolecularGraph):
    def __init__(self,graph,cell):
        self.refgraph=graph.copy()
        self.slabgraph=graph
        self.cell=cell
        #a=0,b=1,c=2
        self.vacuum_direc=2
        # store original number of nodes so we know what index to use when we start adding H's
        self.num_nodes=self.slabgraph.number_of_nodes()
    
    def __str__(self):
        pass

    def check_if_zeolite(self):

        zeo_types = set(["Si","O","Al"])
        for node, data in self.slabgraph.nodes_iter(data=True):
            if(set(data['element'])>zeo_types):
                print("Warning! Structure determined not to be zeolite! Undefined behavior...")

    def reset_node_edge_labelling(self,graph):

        new_graph = graph.copy()
        pass
        

    #def draw_slabgraph(self):

    #    print("\n\nAttempting to output graph as VTK file for visualization debugging")
    #    numberNodes, numberEdges = 100, 500
    #    H = nx.gnm_random_graph(numberNodes,numberEdges)
    #    #print 'nodes:', H.nodes()
    #    #print 'edges:', H.edges()
    #    # return a dictionary of positions keyed by node
    #    pos = nx.random_layout(H,dim=3)
    #    # convert to list of positions (each is a list)
    #    xyz = [list(pos[i]) for i in pos]
    #    print(xyz)
    #    print(H.edges())
    #    print(type(xyz))
    #    print(type(xyz[0]))
    #    print(type(xyz[0][0]))
    #    degree = H.degree().values()
    #    print(type(degree))
    #    print(degree)
    #    #print(type(degree[0]))
    #    writeObjects(xyz, edges=H.edges(), scalar=degree, name='degree', fileout='network')    

    #    degree=[]
    #    xyz=[]
    #    node_convert_dict={}
    #    node_convert_dict_rev={}
    #    iterate=0
    #    for node,data in self.slabgraph.nodes_iter(data=True):
    #        #xyz.append([np.float64(data['cartesian_coordinates'][0]),
    #        #            np.float64(data['cartesian_coordinates'][1]),
    #        #            np.float64(data['cartesian_coordinates'][2])])
    #        xyz.append([np.float64(data['_atom_site_fract_x']),
    #                    np.float64(data['_atom_site_fract_y']),
    #                    np.float64(data['_atom_site_fract_z'])])
    #        if(data['element']=="Si"):
    #            degree.append(5)
    #        elif(data['element']=="X"):
    #            degree.append(20)
    #        else:
    #            degree.append(1)

    #        node_convert_dict[iterate]=node
    #        node_convert_dict_rev[node]=iterate
    #        iterate+=1

    #    duplicate_edges=[]
    #    for edge in self.slabgraph.edges():
    #        n1=edge[0]
    #        n2=edge[1]
    #        duplicate_edges.append((node_convert_dict_rev[n1],
    #                                node_convert_dict_rev[n2]))



    #    #writeObjects(xyz, edges=self.slabgraph.edges(), scalar=degree, name='degree', fileout='network')    
    #    writeObjects(xyz, edges=duplicate_edges, scalar=degree, name='degree', fileout='network')    
    #    print(type(xyz))
    #    print(type(xyz[0]))
    #    print(type(xyz[0][0]))
    #    print(type(degree))
    #    print(degree)
    #    print(xyz)
    #    print(self.slabgraph.edges())
    #    print(duplicate_edges)
    #    #print(type(degree[0]))

    #    print(len(degree))
    #    print(len(xyz))
    #    f=open('test.xyz','w')
    #    f.write("%d\n\n"%(len(xyz)))
    #    for elem in xyz:
    #        f.write("Si %.5f %.5f %.5f\n"%(elem[0],elem[1],elem[2]))
    #    f.close()



    def remove_erroneous_disconnected_comps(self):
        """
        Remove erroneous disconnected components created by ASE
        Doesn't happy except for extremely high miller faces w/ASE but nonetheless
        we should handle it and remove them, and alert the user
        """

        if not nx.is_connected(self.slabgraph):
            # get a list of unconnected networks
            sub_graphs = list(nx.connected_component_subgraphs(self.slabgraph))

            main_graph = sub_graphs[0]

            # find the largest network in that list
            for sg in sub_graphs:
                if len(sg.nodes()) > len(main_graph.nodes()):
                    main_graph = sg

            self.slabgraph = main_graph

            print("WARNING! You passed in a graph with disconnected components...\
                  Assuming the largest component is the slab and continuing...")

            # Important!! The reference graph must now be copied from 
            # our new slabgraph after the disconnected components have been removed
            self.refgraph=self.slabgraph.copy()

        #print("Reference graph:")
        #for node, data in self.refgraph.nodes_iter(data=True):
        #    print(node, data['ciflabel'],data['element'])

    def condense_graph(self):
        """
        If we have a zeolite graph, condense it to a Si only graph
        """

        print("\n\n Consolidating Si-O graph to Si only graph\n")
        # store the node indices of all removed O's
        self.removed_nodes = nx.Graph()
        self.removed_edges = []
        self.removed_edges_data = []
        self.added_edges = []
        self.added_edges_data = []

        for node, data in self.slabgraph.nodes_iter(data=True):
            if(data['element']=="O"): 

                neighbors=self.slabgraph.neighbors(node)


                if(len(neighbors)==2):
                    # normal O coordination environment
                    #print("Node %d is a bulk O node"%node)

                    # identify edges to remove AND the data associated with that node
                    self.removed_edges.append((neighbors[0],node))
                    self.removed_edges_data.append(self.slabgraph.edge[neighbors[0]][node])

                    self.removed_edges.append((neighbors[1],node))
                    self.removed_edges_data.append(self.slabgraph.edge[neighbors[1]][node])
                
                    # create an edge between the adjacent Si
                    self.added_edges.append((neighbors[0],neighbors[1]))

                    # need to keep the periodicity information
                    this_symflag=''
                    if(self.slabgraph.edge[neighbors[0]][node]['symflag']!='.'):
                        this_symflag=self.slabgraph.edge[neighbors[0]][node]['symflag']
                    elif(self.slabgraph.edge[neighbors[1]][node]['symflag']!='.'):
                        this_symflag=self.slabgraph.edge[neighbors[1]][node]['symflag']
                    else:
                        this_symflag=self.slabgraph.edge[neighbors[0]][node]['symflag']
                    self.added_edges_data.append({ 'order':1, 'length': 4.0, 'symflag':this_symflag })
                    

                elif(len(neighbors)==1):
                    #print("Node %d is a surface O node"%node)
                    # the arbitrary initial slab config can have dangling O's
                    # remove both edges to Si
                    self.removed_edges.append((neighbors[0],node))
                    self.removed_edges_data.append(self.slabgraph.edge[neighbors[0]][node])

                    # no edge to add to the all Si graph
                else:
                    # a disconnected O existed in the intial ase truncation
                    # do nothing since we have already removed disconnected
                    # componenets in remove_erroneous_disconnected_comps
                    pass
                
                # remove the O node
                self.removed_nodes.add_node(node,data)
        #    elif(data['element']=='Si'):
        #        print("Si has %d neighbors"%len(self.slabgraph.neighbors(node)))

        #print("All nodes to remove:")
        #print(self.removed_nodes)
        #print("All edges to remove:")
        #print(self.removed_edges)
        #print("All edges to add:")
        #print(self.added_edges)


        # remove Si-O or Al-O edges
        for edge in self.removed_edges:
            self.slabgraph.remove_edge(edge[0],edge[1])            
    
            # the sorted edge dict is used by write_CIF but IS NOT updated when modifying
            # the Nx graph data structure, hence we need to manually add here
            try:
                del self.slabgraph.sorted_edge_dict[(edge[0],edge[1])]
            except:
                pass
            try:
                del self.slabgraph.sorted_edge_dict[(edge[1],edge[0])]
            except:
                pass


        # Add Si-Si/Si-Al/Al-Al,etc edges
        #edge_data={ 'order':1, 'length': 4.0, 'symflag':'--' } 
        #for edge in self.added_edges:
        for edge,edge_data in zip(self.added_edges,self.added_edges_data):
            self.slabgraph.add_edge(*edge,attr_dict=edge_data)

            # the sorted edge dict is used by write_CIF but IS NOT updated when modifying
            # the Nx graph data structure, hence we need to manually add here
            if(edge[0]<edge[1]):
                self.slabgraph.sorted_edge_dict[(edge[0],edge[1])] = edge
            else:
                self.slabgraph.sorted_edge_dict[(edge[1],edge[0])] = edge

        # Remove O nodes
        for node in self.removed_nodes:
            self.slabgraph.remove_node(node)

      
        print(self.slabgraph.name)      

    def normalize_bulk_edge_weights(self):
        """
        Make sure weight for all existing edges in graph is 1
        """

        for n1,n2,data in self.slabgraph.edges_iter(data=True):
            self.slabgraph[n1][n2]['weight']=1


    def identify_undercoordinated_surface_nodes(self):

        self.surface_nodes=[]
        self.surface_nodes_0=[]
        self.surface_nodes_max=[]
        self.bulk_nodes=[]

        # temporary number of edges look up
        # this is necessary because a fully coordinated Si can still have
        # only 3 neighors if it connects another Si by a both bulk edge AND a 
        # periodic edge 
        temp_neigh_list = {}
        for edge in self.added_edges:
            if(edge[0] not in temp_neigh_list.keys()):
                temp_neigh_list[edge[0]]=1
            else:
                temp_neigh_list[edge[0]]+=1
                
            if(edge[1] not in temp_neigh_list.keys()):
                temp_neigh_list[edge[1]]=1
            else:
                temp_neigh_list[edge[1]]+=1


        print("\n\nIdentifying surface/bulk nodes\n")
        for node,data in self.slabgraph.nodes_iter(data=True):
            if(data['element']=="O"):
                print("ERROR: O's have to be removed from graph first")
                sys.exit()

            # Here need to be very careful
            # For small unit cells, one Si can be fully coordinated w/o having
            # 4 neighbors as it can be connected to a different Si AND the
            # periodic image  
            neighbors=self.slabgraph.neighbors(node)
            
            #print("Si has %d neighbors"%len(self.slabgraph.neighbors(node)))
            #print("Si has %d edges"%temp_neigh_list[node])

            #if(len(neighbors)<4):
            if(temp_neigh_list[node]<4):
                # if node is undercoordinated we identify it as a surface node
                self.surface_nodes.append(node)
                data['element']='X'
                # For now rough approximation to distinguish nodes between the 2 surfaces
                # TODO better
                if(self.vacuum_direc==0):
                    if(float(data['_atom_site_fract_x'])<0.5):
                        self.surface_nodes_0.append(node)
                    else:
                        self.surface_nodes_max.append(node)
                elif(self.vacuum_direc==1):
                    if(float(data['_atom_site_fract_y'])<0.5):
                        self.surface_nodes_0.append(node)
                    else:
                        self.surface_nodes_max.append(node)
                elif(self.vacuum_direc==2):
                    if(float(data['_atom_site_fract_z'])<0.5):
                        self.surface_nodes_0.append(node)
                    else:
                        self.surface_nodes_max.append(node)
            else:
                # any fully coordinated node is automatically a bulk node
                self.bulk_nodes.append(node)

        #print("Bulk nodes:")
        #print(self.bulk_nodes)
        print("Surface 0 nodes:")
        print(self.surface_nodes_0)
        print("Surface max nodes:")
        print(self.surface_nodes_max)


    def connect_super_surface_nodes_v2(self):
        """
        Add two new super surface nodes 

        Connect each super surface node to its corresponding surface nodes
        with weight 100000
        """
        
        # new super surface nodes and associated data
        self.super_surface_node_0   = -1
        self.super_surface_node_max = -2
        cartesian_0   = self.to_cartesian([0.5,0.5,0])
        cartesian_max = self.to_cartesian([0.5,0.5,1])
        data_0={'element':'X','ciflabel':'X-1', '_atom_site_fract_x':0.5,
                                                '_atom_site_fract_y':0.5,
                                                '_atom_site_fract_z':0.0,
                                                'cartesian_coordinates':cartesian_0,
                                                'slablayer':0
               }
        data_max={'element':'X','ciflabel':'X-2', '_atom_site_fract_x':0.5,
                                                '_atom_site_fract_y':0.5,
                                                '_atom_site_fract_z':1.0,
                                                'cartesian_coordinates':cartesian_max,
                                                'slablayer':0
               }
    
        # add to our undirected slab graph
        self.slabgraph.add_node(self.super_surface_node_0,data_0)
        self.slabgraph.add_node(self.super_surface_node_max,data_max)

        # for now add the super surface nodes to the refgraph for technicality reasons
        self.refgraph.add_node(self.super_surface_node_0,data_0)
        self.refgraph.add_node(self.super_surface_node_max,data_max)

        # add weights of 100000 from super surface nodes to surface nodes
        for i in range(0, len(self.surface_nodes_0)):
            n2=self.surface_nodes_0[i]
            edge=(self.super_surface_node_0,n2)
            self.slabgraph.add_edge(*edge,weight=100000)

        for i in range(0, len(self.surface_nodes_max)):
            n2=self.surface_nodes_max[i]
            edge=(self.super_surface_node_max,n2)
            self.slabgraph.add_edge(*edge,weight=100000)


            

    def connect_super_surface_nodes(self):
        """
        Choose the first node on surface "0"
        Connect all other surface_0->bulk connections to this first node
        Each added node has weight 1 
        Remove all other surface_0 nodes
        """

        self.super_surface_node_0=self.surface_nodes_0[0]
        print("First node listed on 0 surface = %s%d"%(
            self.slabgraph.node[self.super_surface_node_0]['element'],
            self.super_surface_node_0))
        num_connected_to_0=0
        for i in range(1,len(self.surface_nodes_0)):
            n1=self.surface_nodes_0[i]
            #print("Surface node: %d"%n1)
            for n2 in self.slabgraph.neighbors(n1):
                if(n2 in self.bulk_nodes):
                    #print("Bulk node: %d"%n2)
                    edge=(self.super_surface_node_0,n2)
                    # instead of adding duplicate edges, increase edge weight
                    if edge not in self.slabgraph.edges():
                        self.slabgraph.add_edge(*edge,weight=1)
                    else:
                        self.slabgraph[edge[0]][edge[1]]['weight']+=1
                    num_connected_to_0+=1
    
            self.slabgraph.remove_node(n1)

        # calculate total weight along the super node
        self.super_surface_node_0_weight=0 
        for node in self.slabgraph.neighbors(self.super_surface_node_0):
            self.super_surface_node_0_weight+=\
                self.slabgraph[self.super_surface_node_0][node]['weight']

        print("%d edges added to super_surface_node_0"%(num_connected_to_0))
        print("edges to super_surface_node_max: %s"%\
              str(self.slabgraph.edges(self.super_surface_node_0,data=True)))
        print("%d neighbors of super_surface_node_max"%\
              (len(self.slabgraph.neighbors(self.super_surface_node_0))))
        print("Total weight out: %d"%self.super_surface_node_0_weight)

        
    
        self.super_surface_node_max=self.surface_nodes_max[0]
        print("First node listed on max surface = %s%d"%\
              (self.slabgraph.node[self.super_surface_node_max]['element'],
              self.super_surface_node_max))

        num_connected_to_max=0
        for i in range(1,len(self.surface_nodes_max)):
            n1=self.surface_nodes_max[i]
            for n2 in self.slabgraph.neighbors(n1):
                if(n2 in self.bulk_nodes):
                    edge=(self.super_surface_node_max,n2)
                    # instead of adding duplicate edges, increase edge weight
                    if edge not in self.slabgraph.edges():
                        self.slabgraph.add_edge(*edge,weight=1)
                    else:
                        self.slabgraph[edge[0]][edge[1]]['weight']+=1
                    num_connected_to_max+=1
    
            self.slabgraph.remove_node(n1)

        # calculate total weight along the super node
        self.super_surface_node_max_weight=0 
        for node in self.slabgraph.neighbors(self.super_surface_node_max):
            self.super_surface_node_max_weight+=\
                self.slabgraph[self.super_surface_node_max][node]['weight']

        print("%d edges added to super_surface_node_max"%(num_connected_to_max))
        print("edges to super_surface_node_0: %s"%\
              str(self.slabgraph.edges(self.super_surface_node_max)))
        print("%d neighbors of super_surface_node_max"%\
              (len(self.slabgraph.neighbors(self.super_surface_node_max))))
        print("Total weight out: %d"%self.super_surface_node_max_weight)

    def equiv_symflag(self,string):
        """
        Helper function for comparing symflags under special circumstances
        """
        #return string
        new_string=re.sub('4','6',string)
        return new_string
    
    def check_D4R(self):
        """
        Check for D4R rings (preferential Ge sites) that can react with water
        to dislocate the structure
        """

        print("Checking for D4R rings:")
       
        tmpG=deepcopy(self.slabgraph)
        #simple_cycles=list(nx.simple_cycles(tmpG))
        #print(simple_cycles)

        cycles = []
        # not the most efficient, can def be improved
        for node, data in tmpG.nodes_iter(data=True):
            print("Node %d:"%(node))
            #cycle = list(nx.all_shortest_paths(tmpG, node, node))
            for n in tmpG.neighbors(node):
                # fastest way I could think of..
                edge = tmpG[node][n].copy()
                tmpG.remove_edge(node, n)
                cycle = []
                try:
                    cycle = list(nx.all_shortest_paths(tmpG, node, n))
                except nx.exception.NetworkXNoPath:
                    pass
                tmpG.add_edge(node, n, **edge)
                print(len(cycle))
                print(cycle)
                filter_cycles=[]
                for ind in cycle:
                    # make sure its a 4 ring
                    if(len(ind)==4):
                        # make sure its periodic
                        if(ind[3] in tmpG.neighbors(ind[0])):
                            # make sure it doesn't contain the super surface nodes
                            # because these edges don't have a symflag bc it 
                            # carries no physical meaning for this edge
                            if(self.super_surface_node_max not in ind and \
                               self.super_surface_node_0 not in ind):
                                filter_cycles.append(ind)
                cycles += filter_cycles

        # now we have a list of all 4 cycles in the graph
        #print("Cycles: %s"%(str(cycles)))

        nexttmpG=nx.Graph()
        for four_ring in cycles:
            # get the symflag (aka periodicity) of each bond in the identified 4 cycle
            # -
            # this is important because it's not a 4 ring in the edge case
            # where a periodicity flag occurs exactly once when listing all the edges' periodicities
            # -
            # in such a case, making the supercell larger would find that this
            # motif is NOT a 4R, so we must exclude it

            periodicity=[]
            periodicity2=[]
            periodicity.append(self.equiv_symflag(tmpG[four_ring[0]][four_ring[1]]['symflag']))
            periodicity.append(self.equiv_symflag(tmpG[four_ring[1]][four_ring[2]]['symflag']))
            periodicity.append(self.equiv_symflag(tmpG[four_ring[2]][four_ring[3]]['symflag']))
            periodicity.append(self.equiv_symflag(tmpG[four_ring[3]][four_ring[0]]['symflag']))

            periodicity2.append(self.equiv_symflag(tmpG[four_ring[3]][four_ring[0]]['symflag']))
            periodicity2.append(self.equiv_symflag(tmpG[four_ring[2]][four_ring[3]]['symflag']))
            periodicity2.append(self.equiv_symflag(tmpG[four_ring[1]][four_ring[2]]['symflag']))
            periodicity2.append(self.equiv_symflag(tmpG[four_ring[0]][four_ring[1]]['symflag']))
            print(four_ring)
            print(periodicity)
            print(periodicity2)

            # create a dict that tells how many times each sym flag occurs in periodicity
            counter=Counter(periodicity)
            # now we determine if a ring is actually a cycle or only appears that way bc of 
            # crossing one periodic boundary
            # NOTE this is def not rigorous just a simple quick hack for the case of 4
            # membered rings
            to_add=False
            if(set(counter.keys()).union(set("."))==set(".")):
                # All bonds are non PBC so must be a true cycle
                to_add=True
            else:
                for key in counter:
                    if(key=="."):
                        pass
                    elif(counter[key]%2==0):
                        # found a PBC direction that is crossed by 2 different edges
                        to_add=True
                        break 

            if(to_add==True):
                nexttmpG.add_edge(four_ring[0],four_ring[1])
                nexttmpG.add_edge(four_ring[1],four_ring[2])
                nexttmpG.add_edge(four_ring[2],four_ring[3])
                nexttmpG.add_edge(four_ring[3],four_ring[0])

        connected_comps=sorted(nx.connected_components(nexttmpG), key=len)
        print("All connected comps:")
        print(connected_comps)

        # group all four rings that share any nodes (could also just do in 
        # NetworkX and use connected components) which are D4Rs or a collection
        # of connected D4Rs
        #print("All 4 cycles:")
        #print(cycles)
        #D4R = []
        #l=deepcopy(cycles)
        #while len(l)>0:
        #    first=l[0]
        #    rest=l[1:]
        #    #first, *rest = l
        #    first = set(first)

        #    lf = -1
        #    while len(first)>lf:
        #        lf = len(first)

        #        rest2 = []
        #        for r in rest:
        #            if len(first.intersection(set(r)))>0:
        #                first |= set(r)
        #            else:
        #                rest2.append(r)     
        #        rest = rest2

        #    D4R.append(first)
        #    l = rest
        # filter out anything not containing 8 nodes


        # finally we can check if a connected component is indeed a D4R
        final_D4R=[]
        for struct in connected_comps:
            #if(len(struct)%4==0 and len(struct)/4>=2):
            # each node in the component must have at least two neighbors
            # that are also in the component
            still_a_D4R=True
            for node in struct:
                if(len(set(self.slabgraph.neighbors(node)).intersection(struct))<3):
                    still_a_D4R=False
                    print("Exculding component (not a true D4R): %s"%(str(struct)))
                    break
            if(still_a_D4R):        
                final_D4R.append(struct)

        # final subunits 
        print("All D4Rs:")
        print(str(final_D4R))
        if(len(final_D4R)==0):
            print("Warning, structure passed in with no D4R units, but surface\
                   was requested to be generated w/Ge delamination. Exiting...")
        else:
            f=open('D4R.txt','w')
            for struct in final_D4R:
                f.write("%s\n"%str(struct))
            f.close()
            


        # finally, weight all edges that go from a node in a D4R to a node
        # not in a D4R with a value of 0.0
        self.all_D4R_nodes=set()
        self.all_D4R_nodes=self.all_D4R_nodes.union(*final_D4R)
        print("All nodes in D4Rs:")
        print(self.all_D4R_nodes)
        for struct in final_D4R:
            for n in struct:
                for neigh in self.slabgraph.neighbors(n):
                    # if the edge goes from a node in the D4R to a node outside
                    # the D4R, the weight is 0
                    #if(neigh not in self.all_D4R_nodes):
                    if(neigh not in struct):
                        self.slabgraph.edge[n][neigh]['capacity']=0.0
                        self.slabgraph.edge[n][neigh]['weight']=0.0

        


                    
    def convert_to_digraph(self):
        """
        Turn the slabgraph into a tree
        """
        #self.slabgraphdirec = nx.bfs_tree(self.slabgraph, self.super_surface_node_0)
        #self.iterative_BFS_tree_structure(self.super_surface_node_0)

        # keep copy of the old undirected graph
        self.slabgraphundirected=self.slabgraph.copy()

        # create directed copy of the slabgraph
        self.slabgraphdirec=self.slabgraph.to_directed()

        # add large weight to all edges from the super surface node to the first layer
        # of bulk nodes
        self.create_weighted_barrier_on_super_surface_edges(super_surface_weight='inf')
   


    def iterative_BFS_tree_structure(self, v):                                  
        """                                                                     
        Construct a dict with key that indexes depth of BFS tree,               
        and the value is a set of all nodes at that depth                       
        """                                                  

        if(nx.is_tree(self.slabgraphdirec)):                   
            print("\n\nPRINTING SLAB GRAPH AT EACH LEVEL OF TREE DEPTH")                       
            print("--------------------------------------")                         
                                                                                    
                                                                                    
            # intitialize first level                                               
            stack = set()                                                           
            stack.add(v)                                                            
            curr_depth = 0                                                          
                                                                                    
            self.BFS_tree_dict = {                                                  
                                    curr_depth: set(stack)                          
                                 }                                                  
                                                                                    
            curr_depth += 1                                                         
                                                                                    
                                                                                    
            # Move through every depth level in tree                                
            while(len(stack) != 0):                                                 
                                                                                    
                # iterate over all up_nodes in stack                                
                for up_node in stack.copy():                                        
                                                                                    
                    # get all down nodes from this up_node                          
                    for down_node in self.slabgraphdirec.successors_iter(up_node):            
                        stack.add(down_node)                                        
                                                                                    
                    # after we've gotten all down nodes, remove this up node        
                    stack.remove(up_node)                                           
                                                                                    
                # add this depth and all nodes to the graph                         
                if(len(stack) != 0):                                                
                    self.BFS_tree_dict[curr_depth] = set(stack)                     
                    curr_depth += 1                                                 
                                                                                    
                                                                                    
                                                                                    
            print("Depth of BFS tree: " + str(len(self.BFS_tree_dict.keys())))      
            for i in range(len(self.BFS_tree_dict.keys())):                         
                print("Level " + str(i) + ": " + str(len(self.BFS_tree_dict[i])))   
                print(self.BFS_tree_dict[i])


            # Doesn't seem to be a way to override the 1 parent rule in any default
            # tree generator function in networkx, therefore need to go back in manually
            for i in range(len(self.BFS_tree_dict.keys())-1):
                #print("LEvel %d:"%i)
                for n1 in self.BFS_tree_dict[i]:
                    for n2 in self.BFS_tree_dict[i+1]:
                        directed_edge=(n1, n2)
                        rev_directed_edge=(n2, n1)
                        if(directed_edge not in self.slabgraphdirec.edges()):
                            if(directed_edge in self.slabgraph.edges() or
                               rev_directed_edge in self.slabgraph.edges()):
                                #print("Adding directed edge: %s"%str(directed_edge))
                                self.slabgraphdirec.add_edge(*directed_edge)

        else:
            # we already have made sure one child can have multiple parents
            pass
        
        self.slabgraphdirec=self.slabgraph.to_directed()
        self.create_weighted_barrier_on_super_surface_edges(super_surface_weight='max')

    def redirect_slab_tree_by_coordinate_directionality(self,start="min"):
        """
        Redirect the edges in the directed version of the slab graph
        solely based on the fractional coordinate that represents the
        crystallographic position perpendicular to the surface (parallel to the
        vacuum)

        if start=='min', the parent node must have a vacuum_coord < child node
        if start=='max', the parent node must have a vacuum_coord > child node
        """

        edges_to_reverse=[]
        for edge in self.slabgraphdirec.edges_iter():
            n1=edge[0]
            n2=edge[1]

            to_reverse=False
            
            if(self.vacuum_direc==0):
                if(self.slabgraph.node[n1]['_atom_site_fract_x']<
                   self.slabgraph.node[n2]['_atom_site_fract_x']):
                    if(start=="max"):
                        to_reverse=True
                else:
                    if(start=="min"):
                        to_reverse=True
            elif(self.vacuum_direc==1):
                if(self.slabgraph.node[n1]['_atom_site_fract_y']<
                   self.slabgraph.node[n2]['_atom_site_fract_y']):
                    if(start=="max"):
                        to_reverse=True
                else:
                    if(start=="min"):
                        to_reverse=True
            elif(self.vacuum_direc==2):
                if(self.slabgraph.node[n1]['_atom_site_fract_z']<
                   self.slabgraph.node[n2]['_atom_site_fract_z']):
                    if(start=="max"):
                        to_reverse=True
                else:
                    if(start=="min"):
                        to_reverse=True
                        
            # Take special care to ensure the correct directionality each edge 
            # betweeen the super surface node and the first bulk node
            if(start=="min"):
                if(n2 == self.super_surface_node_0):
                    to_reverse=True
            elif(start=="max"):
                if(n1 == self.super_surface_node_max):
                    to_reverse=True
    
            if(to_reverse):
                data=self.slabgraphdirec[n1][n2].copy()
                edges_to_reverse.append((n1,n2,data))



        for n1,n2,data in edges_to_reverse:
            print("Reversing! ",n1,n2, data)
            self.slabgraphdirec.remove_edge(n1,n2)
            self.slabgraphdirec.add_edge(n2,n1,data)



    def create_weighted_barrier_on_super_surface_edges(self,super_surface_weight='one'):
        """
        Depending on which min cut algo we are using, we may want to reweight
        the value of each edge between the super surface node and each of its bulk
        neighbors

        - 'one' sets the weight of each one of these edges to one
        - 'max' sets the weight of each one of these edges to the sum of the number
            of neighbors of the super surface node
        """


        if(super_surface_weight=='one'):
            surface_0_weight=1
            surface_max_weight=1
        elif(super_surface_weight=='max'):
            surface_0_weight=float(self.super_surface_node_0_weight)
            surface_max_weight=float(self.super_surface_node_max_weight)
        elif(super_surface_weight=='inf'):
            surface_0_weight=100000
            surface_max_weight=100000
        else:
            print("ERROR: weight to super surface node can only be 'one','max', or 'inf'")
            sys.exit()

        # now reweight each edge between the supersurface node and the bulk node neighbors
        for edge in self.slabgraphdirec.edges_iter():
            self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity']=\
                float(self.slabgraph.edge[edge[0]][edge[1]]['weight'])
            self.slabgraphdirec.edge[edge[0]][edge[1]]['weight']=\
                float(self.slabgraph.edge[edge[0]][edge[1]]['weight'])

            # However, if one node in the edge is the super surface node,
            # reset the capacity to the max weight of the supernode
            if(edge[0] == self.super_surface_node_0 or \
               edge[1] == self.super_surface_node_0):
                # adjust directed graph edge properties
                self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity'] = \
                    surface_0_weight 
                self.slabgraphdirec.edge[edge[0]][edge[1]]['weight'] = \
                    surface_0_weight 

                # adjust undirected graph edge properties
                self.slabgraphundirected.edge[edge[0]][edge[1]]['capacity'] = \
                    surface_0_weight 
                self.slabgraphundirected.edge[edge[0]][edge[1]]['weight'] = \
                    surface_0_weight 

            elif(edge[0] == self.super_surface_node_max or \
                 edge[1] == self.super_surface_node_max):
                # adjust directed graph edge properties
                self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity'] = \
                    surface_max_weight
                self.slabgraphdirec.edge[edge[0]][edge[1]]['weight'] = \
                    surface_max_weight

                # adjust undirected graph edge properties
                self.slabgraphundirected.edge[edge[0]][edge[1]]['capacity'] = \
                    surface_max_weight
                self.slabgraphundirected.edge[edge[0]][edge[1]]['weight'] = \
                    surface_max_weight


            #print(edge)
            #print(self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity'])


    def create_weighted_barrier_on_two_outer_surface_layers(self,new_weight='inf'):
        """
        Reweight the edges that have both nodes in the 2-n layers of the slab
        """
        if(new_weight=='one'):
            surface_0_weight=1
            surface_max_weight=1
        elif(new_weight=='max'):
            surface_0_weight=float(self.super_surface_node_0_weight)
            surface_max_weight=float(self.super_surface_node_max_weight)
        elif(new_weight=='inf'):
            surface_0_weight=100000
            surface_max_weight=100000
        else:
            print("ERROR: weight to super surface node can only be 'one','max', or 'inf'")
            sys.exit()

        # already iterates over forward and reverse edges
        for edge in self.slabgraphdirec.edges_iter():

            #print(self.slabgraphdirec.node[edge[0]]['slablayer'])
            #print(self.slabgraphdirec.node[edge[1]]['slablayer'])
            if(self.slabgraphdirec.node[edge[0]]['slablayer']>=2 and
               self.slabgraphdirec.node[edge[1]]['slablayer']>=2):
            
                self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity'] = \
                    surface_max_weight
                self.slabgraphdirec.edge[edge[0]][edge[1]]['weight'] = \
                    surface_max_weight

    # NOTE not really needed at all anymore (I think)                       
    #def create_weighted_barrier_at_slab_center(self, start='weight'):
    #    """
    #    Here just set a high weight for any edge that is bisected by the center plane
    #    of the vacuum_direc coordinate

    #    i.e. node1 has a c-coordinate less than 0.5 and and node2 has a 
    #    c-coordinate greater than 0.5
    #    """

    #    for edge in self.slabgraphdirec.edges_iter():
    #        n1=edge[0]
    #        n2=edge[1]

    #        if(n1 in self.bulk_nodes and n2 in self.bulk_nodes):
    #            if(start=='weight'):
    #                if(self.vacuum_direc==0):
    #                    if(((float(self.slabgraph.node[n1]['_atom_site_fract_x'])-0.5) < 0) !=\
    #                       ((float(self.slabgraph.node[n2]['_atom_site_fract_x'])-0.5) < 0)):
    #                        self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
    #                        self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
    #                elif(self.vacuum_direc==1):
    #                    if(((float(self.slabgraph.node[n1]['_atom_site_fract_y'])-0.5) < 0) !=\
    #                       ((float(self.slabgraph.node[n2]['_atom_site_fract_y'])-0.5) < 0)):
    #                        self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
    #                        self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
    #                elif(self.vacuum_direc==2):
    #                    if(((float(self.slabgraph.node[n1]['_atom_site_fract_z'])-0.5) < 0) !=\
    #                       ((float(self.slabgraph.node[n2]['_atom_site_fract_z'])-0.5) < 0)):
    #                        self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
    #                        self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
    #            elif(start=='unweight'):
    #                self.slabgraphdirec.edge[n1][n2]['capacity'] = 1
    #                self.slabgraphdirec.edge[n1][n2]['weight'] = 1

            
    def create_weighted_barrier_on_opposite_half(self, start='min'):
        """
        If we have a weighted barrier:
        if start == 'min':
             all non-super surface edges with vacuum_coord > 0.5 have weight 1,000,000
        else if start == 'max'
             all non-super surface edges with vacuum_coord < 0.5 have weight 1,000,000
        else if start == 'neutral'
             all non-super surface edges reset to weigth 1
        """
        for edge in self.slabgraphdirec.edges_iter():
            n1=edge[0]
            n2=edge[1]

            if(n1 in self.bulk_nodes and n2 in self.bulk_nodes):
                # if source in the minimum (0) super surface node, make every edge with nodes
                # greater than 0.5 vacuum coord a capacity 100000
                if(start=='min'):
                    if(self.vacuum_direc==0):
                        if(float(self.slabgraph.node[n1]['_atom_site_fract_x']) < 0.5 and 
                           float(self.slabgraph.node[n2]['_atom_site_fract_x']) < 0.5):
                            self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
                            self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
                    elif(self.vacuum_direc==1):
                        if(float(self.slabgraph.node[n1]['_atom_site_fract_y']) < 0.5 and 
                           float(self.slabgraph.node[n2]['_atom_site_fract_y']) < 0.5):
                            self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
                            self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
                    elif(self.vacuum_direc==2):
                        if(float(self.slabgraph.node[n1]['_atom_site_fract_z']) < 0.5 and 
                           float(self.slabgraph.node[n2]['_atom_site_fract_z']) < 0.5):
                            self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
                            self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000

                elif(start=='max'):
                    if(self.vacuum_direc==0):
                        if(float(self.slabgraph.node[n1]['_atom_site_fract_x']) > 0.5 and 
                           float(self.slabgraph.node[n2]['_atom_site_fract_x']) > 0.5):
                            self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
                            self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
                    elif(self.vacuum_direc==1):
                        if(float(self.slabgraph.node[n1]['_atom_site_fract_y']) > 0.5 and 
                           float(self.slabgraph.node[n2]['_atom_site_fract_y']) > 0.5):
                            self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
                            self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
                    elif(self.vacuum_direc==2):
                        if(float(self.slabgraph.node[n1]['_atom_site_fract_z']) > 0.5 and 
                           float(self.slabgraph.node[n2]['_atom_site_fract_z']) > 0.5):
                            self.slabgraphdirec.edge[n1][n2]['capacity'] = 1000000
                            self.slabgraphdirec.edge[n1][n2]['weight'] = 1000000
                
                elif(start=='neutral'):
                    self.slabgraphdirec.edge[n1][n2]['capacity'] = 1
                    self.slabgraphdirec.edge[n1][n2]['weight'] = 1

                else:
                    print("ERROR: only three options for weighted barrier (min, max, neutral)")
                    sys.exit()
                


    def add_surface_edges(self):
        """
        Make sure all existing edges in graph have weight of 1
        Add weights of infinity to all "surface edges in graph"
        """

        self.surface_edges=[]

        edge_data={ 'order':1000000, 'length': 4.0, 'symflag':'--' } 
        for i in range(1,len(self.surface_nodes)):
            edge=(self.surface_nodes[0],self.surface_nodes[i])
            self.surface_edges.append(edge)
            self.slabgraph.add_edge(*edge,weight=1000000,attr_dict=edge_data)



    def nx_stoer_wagner_cut_custom(self, weight_barrier=False):
        print("\n\nNx stoer_wagner function on undirected slab graph...")
        cut_value, partition = nxswc.stoer_wagner_custom(self.slabgraphundirected,
                                                         s=self.super_surface_node_0,
                                                         t=self.super_surface_node_max)
        print(cut_value)        
        print(partition)

        sys.exit()


    def minimum_edge_slab_cut(self):

        edge_cut_set=nx.minimum_edge_cut(self.slabgraph,
                                         s=self.super_surface_node_0,
                                         t=self.super_surface_node_max)

        print(len(edge_cut_set))
        print(str(edge_cut_set))

    def minimum_edge_slab_tree_cut(self):

        edge_cut_set=nx.minimum_edge_cut(self.slabgraphdirec,
                                         s=self.super_surface_node_0,
                                         t=self.super_surface_node_max)

        print(len(edge_cut_set))
        print(str(edge_cut_set))
   
    def union_2_tuple(self, set1, set2):
        
        new_set = deepcopy(set1)
        for elem in set2:
            new_set.add(elem)

        return new_set

    def relative_complement_2_tuple(self,set1, set2):
        new_set=set()
        for elem in set1:
            if(elem not in set2):
                new_set.add(elem)

        return new_set

    def get_edges_between_partitions(self,partition):
        """
        Given two sets of nodes find all edges that connect the two components
        """

        edges_that_were_cut=[]
        for s in partition[0]:
            for t in partition[1]:
                if((s,t) in self.slabgraphdirec.edges()):
                    edges_that_were_cut.append((s,t))

        return edges_that_were_cut

    def keep_N_layers(self, G, max_layer):
        """
        For an input graph G, return a copy that has only nodes w/attribute 
        slablayer in layers
        """

        print("\nReducing graph to only contain nodes up to and including slablayer %d"%max_layer)

        Gp=deepcopy(G)

        # new super surface node
        self.new_super_surface_node_max=-3

        # new list of surface nodes (ones connected to the max layer)
        self.new_surface_nodes_max=[]

        # all nodes to remove
        remove_nodes=[]

        for n1, data in Gp.nodes_iter(data=True):
            #print(data['slablayer'])
            #print(max_layer)
            if(data['slablayer'] == max_layer+1):
                # check if that node is connected to the max layer
                to_remove=True
                for n2 in Gp.neighbors(n1):
                    if(Gp.node[n2]['slablayer'] <= max_layer):
                        to_remove=False
                        self.new_surface_nodes_max.append(n1)
                        break
                
                if(to_remove):
                    remove_nodes.append(n1)

            elif(data['slablayer'] > max_layer+1):
                # remove node 
                remove_nodes.append(n1)

            else:
                # Do nothing
                pass
        
        print("new surface nodes max: ", self.new_surface_nodes_max)

        cartesian_max = self.to_cartesian([0.5,0.5,1])
        new_data_max={'element':'X','ciflabel':'X-3', '_atom_site_fract_x':0.5,
                                                '_atom_site_fract_y':0.5,
                                                '_atom_site_fract_z':1.0,
                                                'cartesian_coordinates':cartesian_max,
                                                'slablayer':0
               }

        # remove nodes outside the max slablayer
        for n in remove_nodes:
            Gp.remove_node(n)
    
        # add to our DIRECTED slab graph
        Gp.add_node(self.new_super_surface_node_max,new_data_max)

        # add forward and reverse edges to new supersurface node
        for i in range(0, len(self.new_surface_nodes_max)):
            n2=self.new_surface_nodes_max[i]
            edge=(self.new_super_surface_node_max,n2)
            Gp.add_edge(*edge,weight=100000)

            edge=(n2,self.new_super_surface_node_max)
            Gp.add_edge(*edge,weight=100000)

        print("Graph reduced from %d to %d nodes"%(len(G.nodes()),len(Gp.nodes())))
        print("Neighbors of super_surface_node_0: ", Gp.neighbors(self.super_surface_node_0))
        print("Neightbors of NEW super_surface_node_max: ", Gp.neighbors(self.new_super_surface_node_max))


        return Gp

    def keep_cut_in_N_layers(self, max_layer, all_cut_C0, all_cut_ids):
        """
        For the slab layer specified by max_layer, we only keep a cutset
        if its first cut contains a node in this layer (the other cut
        is the symmetric equivalent on the opposite side so its redundant info
        for thie purposes of this funciton
        """

        new_list=[]
        new_ids=[]

        for i in range(len(all_cut_C0)):
            cut0=all_cut_C0[i][0]
            
            to_keep=False
            for (u,v) in cut0:
                if(self.slabgraph.node[u]['slablayer'] == max_layer):
                    to_keep=True
                elif(self.slabgraph.node[v]['slablayer'] == max_layer):
                    to_keep=True

            if(to_keep==True):
                new_list.append(all_cut_C0[i])
                new_ids.append(all_cut_ids[i])
    
        print("\nCutsets containing node in layer %d (%d):"%(max_layer,len(new_list)))
        print(new_list)
        print(new_ids)

        return new_list, new_ids


    #def remove_duplicate_cuts_in_c(self, layer_props, all_cut_C0, all_cut_ids,c_min_max,debug=True):
    #    """
    #    Huge pain in the but

    #    Find out if the first cut of cutsets is eqiuvalent to any other first cut
    #    and if so exclude it from the final set of slabs

    #    layer_props - relevant layer info for the slab

    #    all_cut_C0 - all (near) min cuts of the form [(cut

    #    c_min_max - if a duplicate cut is found (same a and b for each cut but 
    #    a constant c shift equal to the width of the layer) keep the cut that
    #    either has the min or the max c-coordinate
    #    """

    #    print("\nRemoving duplicate c cuts")
    #    print(all_cut_C0)

    #    # create copy of data so the refernence object not altered
    #    all_cut_C0=deepcopy(all_cut_C0)        
    #    all_cut_ids=deepcopy(all_cut_ids)        

    #    # list for unique cuts
    #    new_list=[]
    #    new_ids=[]

    #    c_offset = layer_props['proj_height']/self.cell.c
    #    tol = np.float64(1e-4)
    #    print("c layer shift = %.5f"%c_offset)
    #    for i in range(len(all_cut_C0)):
    #        # cut one of ith cutset, edge order determined by min c coordinate
    #        cut= []
    #        # for each edge of the cut (u,v) is listed such that u has the minimal c-coord
    #        for u,v in all_cut_C0[i][0]:
    #            if(self.refgraph.node[u]['_atom_site_fract_z']<
    #               self.refgraph.node[v]['_atom_site_fract_z']):
    #                cut.append((u,v,[float(self.refgraph.node[u]['_atom_site_fract_x']),
    #                                 float(self.refgraph.node[u]['_atom_site_fract_y']),
    #                                 float(self.refgraph.node[u]['_atom_site_fract_z'])],
    #                                [float(self.refgraph.node[v]['_atom_site_fract_x']),
    #                                 float(self.refgraph.node[v]['_atom_site_fract_y']),
    #                                 float(self.refgraph.node[v]['_atom_site_fract_z'])]))
    #            else:
    #                cut.append((v,u,[float(self.refgraph.node[v]['_atom_site_fract_x']),
    #                                 float(self.refgraph.node[v]['_atom_site_fract_y']),
    #                                 float(self.refgraph.node[v]['_atom_site_fract_z'])],
    #                                [float(self.refgraph.node[u]['_atom_site_fract_x']),
    #                                 float(self.refgraph.node[u]['_atom_site_fract_y']),
    #                                 float(self.refgraph.node[u]['_atom_site_fract_z'])]))

    #        # sort each edges position in the cut based on a fract of that edge's first node (u) first, 
    #        # then on b fract of that edge's first node (u)
    #        cut=sorted(cut, key=lambda tup: tup[2][0]) 
    #        cut=sorted(cut, key=lambda tup: tup[2][1]) 
    #        
    #        # find a cutset to compare to
    #        # don't duplicate comparisons by starting at i
    #        to_keep = True # whether or not to keep cut i: (if a duplicate is found in cut j), it j will be included later when it becomes i
    #        for j in range(i+1,len(all_cut_C0)):
    #            # if we find an equivalent cut


    #            # cut one of jth cutset (cutset to compare to)
    #            cut_compare=[]
    #            # for each edge of the cut (u,v) is listed such that u has the minimal c-coord
    #            for u,v in all_cut_C0[j][0]:
    #                if(self.refgraph.node[u]['_atom_site_fract_z']<
    #                   self.refgraph.node[v]['_atom_site_fract_z']):
    #                    cut_compare.append((u,v,[float(self.refgraph.node[u]['_atom_site_fract_x']),
    #                                             float(self.refgraph.node[u]['_atom_site_fract_y']),
    #                                             float(self.refgraph.node[u]['_atom_site_fract_z'])],
    #                                            [float(self.refgraph.node[v]['_atom_site_fract_x']),
    #                                             float(self.refgraph.node[v]['_atom_site_fract_y']),
    #                                             float(self.refgraph.node[v]['_atom_site_fract_z'])]))
    #                else:
    #                    cut_compare.append((v,u,[float(self.refgraph.node[v]['_atom_site_fract_x']),
    #                                             float(self.refgraph.node[v]['_atom_site_fract_y']),
    #                                             float(self.refgraph.node[v]['_atom_site_fract_z'])],
    #                                            [float(self.refgraph.node[u]['_atom_site_fract_x']),
    #                                             float(self.refgraph.node[u]['_atom_site_fract_y']),
    #                                             float(self.refgraph.node[u]['_atom_site_fract_z'])]))

    #            # sort each edges position in the cut based on a fract of that edge's first node (u) first, 
    #            # then on b fract of that edge's first node (u)
    #            cut_compare=sorted(cut_compare, key=lambda tup: tup[2][0]) 
    #            cut_compare=sorted(cut_compare, key=lambda tup: tup[2][1]) 

    #            if(debug):
    #                print("Comparing first cuts of cutset i=%d and j=%d"%(i,j))
    #                print(cut)
    #                print(cut_compare)


    #            if(len(cut)!=len(cut_compare)):
    #                # Definitely can't be equivalent cuts
    #                to_keep=True
    #                if(debug):
    #                    print("Cuts of non-equivalent length")
    #        
    #            else:
    #                # a temporary bool as we make the criteria for a duplicate cut stricter and stricter
    #                to_continue=True

    #                # keep printing for debug
    #                keep_printing=True

    #                # check if (a, b) coordinates are consistent througout both cuts
    #                for k in range(len(cut)):
    #                    # make sure a coordinates of 1st node in both cuts equivalent
    #                    if(cut[k][2][0]>(cut_compare[k][2][0]+tol) or 
    #                       cut[k][2][0]<(cut_compare[k][2][0]-tol)):
    #                        to_continue=False
    #                        break
    #                    # make sure b coordinates of 1st node in both cuts equivalent
    #                    if(cut[k][2][1]>(cut_compare[k][2][1]+tol) or 
    #                       cut[k][2][1]<(cut_compare[k][2][1]-tol)):
    #                        to_continue=False
    #                        break
    #                    # make sure a coordinates of 2nd node in both cuts equivalent
    #                    if(cut[k][3][0]>(cut_compare[k][3][0]+tol) or 
    #                       cut[k][3][0]<(cut_compare[k][3][0]-tol)):
    #                        to_continue=False
    #                        break
    #                    # make sure b coordinates of 2nd node in both cuts equivalent
    #                    if(cut[k][3][1]>(cut_compare[k][3][1]+tol) or 
    #                       cut[k][3][1]<(cut_compare[k][3][1]-tol)):
    #                        to_continue=False
    #                        break

    #                # now check c coordinates 
    #                if(to_continue): 

    #                    # need to check if cuts are equivalent            
    #                    list_of_shifts=[]

    #                    for k in range(len(cut)):
    #                        # c-offset between first nodes of edge tuple
    #                        list_of_shifts.append(cut_compare[k][2][2]-cut[k][2][2])
    #                        # c-offset between second nodes of edge tuple
    #                        list_of_shifts.append(cut_compare[k][3][2]-cut[k][3][2])
    #                
    #                    # first check that cuts are equivalent by a constant c shift    
    #                    avg_shift=np.average(list_of_shifts)
    #                    if(debug):
    #                        print("avg shift: all shifts")
    #                        print(avg_shift,":",list_of_shifts)
    #                    for shift in list_of_shifts:
    #                        if(shift>(avg_shift+tol) or shift<(avg_shift-tol)):
    #                            if(debug):
    #                                print("non-unique (this shift, avg shift + tol, avg_shift -tol")
    #                                print(shift, avg_shift+tol, avg_shift-tol)
    #                            to_continue=False
    #                            break
    #                else:
    #                    if(debug and keep_printing):
    #                        print("Not equivalent a b positions")
    #                    keep_printing=False

    #                # if still equivalent, make sure c shift is equal to the c shift of each layer
    #                if(to_continue):
    #                    # NOTE, if avg_shift is positive, it means cut_compare has greater c coordinates
    #                    # NOTE, if avg_shift is negative, it means cut has greater c coordinates
    #                    if(np.abs(avg_shift)>(c_offset+tol) or np.abs(avg_shift)<(c_offset-tol)):
    #                        to_continue=False
    #                else:
    #                    if(debug and keep_printing):
    #                        print("Not all coords have the same c shift")
    #                    keep_printing=False

    #                # finally, if we have satisfied all checks for equivalency...
    #                # WE CANNOT keep cut i, rather we either pass on adding it to the new list
    #                # or we replace cut j with cut i if cut i is the one we want to keep
    #                if(to_continue):
    #                    if(debug):
    #                        print("Cut %d has duplicate in %d"%(i,j))
    #                    # cut i has a duplicate, therefore can't be kept
    #                    to_keep = False
    #                
    #                    # if we want to keep only the identical cut with maximal c
    #                    if(c_min_max=='max'):
    #                        if(debug):
    #                            print("Want cut with maximum c coords")
    #                        if(avg_shift > 0):
    #                            # cut_compare (index j) has a LARGER c, therefore we DO want to keep index j
    #                            # do nothing
    #                            if(debug):
    #                                print("Cut j=%d has LARGER c coords, do nothing for now"%(j))
    #                            pass
    #                            # this cut j will eventually become cut i,
    #                            #      after which it will be kept if no subsequent equivalencies found
    #                        else:
    #                            # cut_compare (index j) has a SMALLER c, therefore we DO NOT want to keep index j
    #                            # therefore index j should be REPLACED with index i
    #                            if(debug):
    #                                print("Cut j=%d has SMALLER c coords, need to replace with cut i=%d"%(j,i))
    #                            all_cut_C0[j]  = deepcopy(all_cut_C0[i])
    #                            all_cut_ids[j] = deepcopy(all_cut_ids[i])

    #                    # if we want to keep only the identical cut with minimal c
    #                    elif(c_min_max=='min'):
    #                        if(debug):
    #                            print("Want cut with minimum c coords")
    #                        if(avg_shift > 0):
    #                            # cut_compare (index j) has LARGER c, therefore we DO NOT want to keep index j
    #                            # therefore index j should be REPLACED with index i
    #                            if(debug):
    #                                print("Cut j=%d has LARGER c coords, need to replace with cut i=%d"%(j,i))
    #                            all_cut_C0[j]  = deepcopy(all_cut_C0[i])
    #                            all_cut_ids[j] = deepcopy(all_cut_ids[i])
    #                        else:
    #                            # cut_compare (index j) has SMALLER c, therefore we DO want to keep index j
    #                            # do nothing
    #                            if(debug):
    #                                print("Cut j=%d has SMALLER c coords, do nothing for now"%(j))
    #                            pass
    #                            # this cut j will eventually become cut i,
    #                            #      after which it will be kept if no subsequent equivalencies found
    #                    else:
    #                        print("For eliminating duplicate cuts, c_min_max can only be 'min' or 'max'") 
    #                        sys.exit()
    #                else:
    #                    if(debug and keep_printing):
    #                        print("Although all coords have same c shift, it is not equivalent to the c layer shift")
    #                    keep_printing=False
    #                
    #                            
    #                if(to_keep==False):
    #                    break

   
    #        if(to_keep==True):
    #            new_list.append(all_cut_C0[i]) 
    #            new_ids.append(all_cut_ids[i]) 

    #    print("\nCutsets that are identical except for a constant c - shift (%d):"%(len(new_list)))
    #    print(new_list)
    #    print(new_ids)    
    #    

    #    return new_list, new_ids
            

    def remove_duplicate_cuts_in_c_v2(self, layer_props, all_cut_C0, 
                                      all_cut_ids,c_min_max,debug=True):
        """

        c_min_max - if 2 duplicate cuts are found, remove the one with either 
        the min or max coords
    
        """
        new_list=[]
        new_ids=[]

        # iterate through all cuts
        for i in range(len(all_cut_C0)):
            this_i_cut=set()
            for elem in all_cut_C0[i][0]:
                this_i_cut.add(elem[0])
                this_i_cut.add(elem[1])

            to_keep_cut_i=True
            # compare w/all cuts w/index greater than i to avoid double counting
            for j in range(i+1,len(all_cut_C0)):
                this_j_cut=set()
                for elem in all_cut_C0[j][0]:
                    this_j_cut.add(elem[0])
                    this_j_cut.add(elem[1])
                print(this_j_cut)

                # if j is identical to i, ignore it.  j will become i later on for comparison
                if(this_i_cut==this_j_cut):
                    to_keep_cut_i=False
                    break

                # check if j is a larger c duplicate of i
                this_i_cut_plus=set([elem+int(layer_props['a_per_l']/3) for elem in this_i_cut])
                if(this_i_cut_plus == this_j_cut):
                    # found a duplicate
                    if(c_min_max=='max'):
                        # j will become i at a later point and will be kept, so do nothing
                        to_keep_cut_i=False
                        break
                    elif(c_min_max=='min'):
                        # we want to keep i, but we don't know if we will find another
                        # duplicate later on, therefore replace j w/i and continue
                        all_cut_C0[j]  = deepcopy(all_cut_C0[i])
                        all_cut_ids[j] = deepcopy(all_cut_ids[i])

                        to_keep_cut_i=False
                        break
                    else:
                        raise TypeError("c_min_max can only be 'min' or 'max'")


                # check if j is a smaller c duplicate of i
                this_i_cut_minus=set([elem-int(layer_props['a_per_l']/3) for elem in this_i_cut])
                if(this_i_cut_minus == this_j_cut):
                    # found a duplicate
                    if(c_min_max=='max'):
                        # we want to keep i, but we don't know if we will find another
                        # duplicate later on, therefore replace j w/i and continue
                        all_cut_C0[j]  = deepcopy(all_cut_C0[i])
                        all_cut_ids[j] = deepcopy(all_cut_ids[i])

                        to_keep_cut_i=False
                        break
                    elif(c_min_max=='min'):

                        # j will become i at a later point and will be kept, so do nothing
                        to_keep_cut_i=False
                        break

                    else:
                        raise TypeError("c_min_max can only be 'min' or 'max'")


            if(to_keep_cut_i==True):
                new_list.append(all_cut_C0[i]) 
                new_ids.append(all_cut_ids[i]) 
                

        return new_list, new_ids
        

    def enumerate_cuts(self, G, s, t, e_incl, e_excl, max_weight, all_min_cuts,max_num_cuts):
        """
        Recursive algorithm to enumerate cuts constrained to contain various
        edges and exclude others (Balcioglu 2003)
        """
        # return if we reach the max number of cuts
        if(max_num_cuts is not None):
            if(len(all_min_cuts)>=max_num_cuts):
                return

        Gp=deepcopy(G)

        # weight exclusion edges
        for edge in e_excl:
            #print(edge)
            Gp.edge[edge[0]][edge[1]]['capacity']=100000
            Gp.edge[edge[1]][edge[0]]['capacity']=100000
        # add artificial edges to insure inclusion of the inclusion edges
        for (u,v) in e_incl:
            # add artificial edge from s to u
            Gp.add_edge(s, u)
            Gp.add_edge(u, s)
            Gp.edge[s][u]['capacity']=100000
            Gp.edge[u][s]['capacity']=100000

            # add artificial edge from v to t
            Gp.add_edge(v, t)
            Gp.add_edge(t, v)
            Gp.edge[v][t]['capacity']=100000
            Gp.edge[t][v]['capacity']=100000

        # compute new max-flow/min-cut (bc we need to ensure we have actually
        # generated a new min cut after forcing edge inclusion/exclusion
        cut_value, partition = nx.minimum_cut(
            Gp, s, t, flow_func=nx.algorithms.flow.shortest_augmenting_path)
        # get the cutset
        Cp = set(self.get_edges_between_partitions(partition))

        # stopping criteria
        # return if non near minimal cut
        if(cut_value > max_weight):
            return

        # ensure that despite the adding of the artificial edges, the proposed cut
        # is indeed still minimal in the original graph
        tmp_weight=sum([self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity'] for edge in Cp])
        if(tmp_weight<=(max_weight*1.000001)):
            print(tmp_weight, Cp)
            all_min_cuts.append(Cp)
            self.debug_min_cuts(Cp,len(all_min_cuts)-1)
       
        #print("Cp") 
        #print(Cp)
        #print("Cp-e_incl") 
        #print(Cp-e_incl)
        for edge in Cp - e_incl:
            edge=set([(edge[0],edge[1])])

            e_excl = e_excl.union(edge)
            #e_excl = e_excl + edge
            #e_excl = self.union_2_tuple(e_excl, edge)

            self.enumerate_cuts(G, s, t, e_incl, e_excl, max_weight, all_min_cuts,max_num_cuts)

            #print("e_excl")
            #print(e_excl)
            #print("edge")
            #print(edge)
            e_excl = e_excl - edge
            #e_excl = self.relative_complement_2_tuple(e_excl, edge)
            #print("relative complement")
            #print(e_excl)

            e_incl = e_incl.union(edge)
            #e_incl = e_incl + edge
            #e_incl = self.union_2_tuple(e_incl, edge)
        
        return
        

    def nx_near_min_cut_digraph_custom(self, eps=0, k=0, weight_barrier=False,
                                                         layer_props=None,
                                                         max_num_cuts=5):
        """
        Find all near min cuts in a graph

        "nearness" controlled by eps and k

        weight barrier is important for the application 
            -ensures we find a solution independent of the initial surface termination
            -esures the solution maintains 2D periodicity of the slab

        layer_props tells us info about which nodes belong to which layers

        max_num_cuts allows us to return after finding the value of the min cut
            rather than enumerating all solutions
        """ 

        print("\n\nNx near minimum cut function on directed slab graph...")

        self.all_cut_C0  =[]
        self.all_cut_paritions=[]
    
        # First, create a barrier on all super surface edges
        # handles when aspect ratio of the slab is too large and the solution is a tunnel from s to t
        if(weight_barrier):
            self.create_weighted_barrier_on_super_surface_edges(super_surface_weight='inf')

        if(layer_props is not None):
            # create a weighted barrier on all but the outer 2 most surface layers
            #self.create_weighted_barrier_on_two_outer_surface_layers(new_weight='inf')

            # rather than create a "weight" barrier to exclude solutions in every layer, 
            # we can simply make a copy of the  graph where slab layers 1,2 are kept and 3..N are removed
            # this reduced graph doesn't lose any info and will speed things up
            Gred = self.keep_N_layers(self.slabgraphdirec,1)
            s=self.super_surface_node_0
            t=self.new_super_surface_node_max
        else:
            Gred=deepcopy(self.slabgraphdirec)
            s=self.super_surface_node_0
            t=self.super_surface_node_max

        self.cut_value1, self.partition1 = nx.minimum_cut(
                #self.slabgraphdirec,
                Gred,
                #self.super_surface_node_0,
                s,
                #self.super_surface_node_max,
                t,
                flow_func=nx.algorithms.flow.shortest_augmenting_path)

        C = set(self.get_edges_between_partitions(self.partition1))

        print("Initial solution:")
        print(self.cut_value1, C)

    
        # We need all the layer properties to do anything fancy with the cuts
        if(layer_props is not None):

            if(max_num_cuts==1):
                # provide escape option if we only care about the value of the min cut
                C  = list(C)
                self.debug_min_cuts(C,0)

                # if we know the inversion_mapping, find the symmetric cut
                if(layer_props['inversion_mapping'] is not None):
                    Cp = self.id_symmetric_cut(C,layer_props)
                # if we don't have inversion symmetry, do the next best thing
                # find the same cut in the opposite slab layer
                elif(layer_props['translation_mapping'] is not None):
                    Cp = self.id_trans_layer_cut(C,layer_props)
                # if no inversion or translation info between layers, get naiive sol'n   
                else:
                    # reverse s and t nodes and find a min cut on the other side of the slab 
                    self.cut_value2, self.partition2 = nx.minimum_cut(
                            #self.slabgraphdirec,
                            #self.super_surface_node_max,
                            #self.super_surface_node_0,
                            Gred,
                            s,
                            t,
                            flow_func=nx.algorithms.flow.shortest_augmenting_path)

                    Cp = set(self.get_edges_between_partitions(self.partition2))

                # finalize opposite cut as a list
                Cp = list(Cp)

                # create the all cutset list
                self.all_cut_C0.append((C,Cp))
                self.all_cut_ids=["%05d"%0]

            else:
                # All near min cut of Balcioglu                

                # the edge inclusion and exclusion sets
                e_incl = set()
                e_excl = set()

                # For now not interested in finding near minimal, only minimal
                fract_tol=eps
                int_tol=k

                if(int_tol != 0):
                    max_weight=self.cut_value1+int_tol
                else:
                    max_weight=self.cut_value1*(1+fract_tol)

                # to track all min cuts
                all_min_cuts=[]

                # run
                self.enumerate_cuts(
                               #self.slabgraphdirec,
                               Gred,
                               #self.super_surface_node_0, 
                               s,
                               #self.super_surface_node_max,
                               t,
                               e_incl,
                               e_excl,
                               max_weight,
                               all_min_cuts,
                               max_num_cuts)

                if(layer_props['inversion_mapping'] is not None):
                    self.all_cut_C0=[(list(C),self.id_symmetric_cut(list(C),layer_props)) for C in all_min_cuts]
                elif(layer_props['translation_mapping'] is not None):
                    self.all_cut_C0=[(list(C),self.id_trans_layer_cut(list(C),layer_props)) for C in all_min_cuts]
                else:
                    pass
                    # TODO how to handle this case if no tranlation mapping
                    # i.e. slabs not made w/pymatgen
                    #print("For now, can't handle multiple cuts w/o inversion or translation mapping to find the opposite cut")
                    #sys.exit()

                self.all_cut_ids=["%05d"%i for i in range(len(self.all_cut_C0))]
                print("\nAll initial cutsets (%d):"%len(self.all_cut_C0))
                print(self.all_cut_C0)
                print(self.all_cut_ids)


                self.all_cut_C0, self.all_cut_ids = self.keep_cut_in_N_layers(\
                    1, self.all_cut_C0, self.all_cut_ids)


                self.all_cut_C0, self.all_cut_ids = self.remove_duplicate_cuts_in_c_v2(\
                    layer_props, self.all_cut_C0, self.all_cut_ids,c_min_max='max')
        

        # if no layer properties known, there is no point in being careful about the 
        # cuts, just find one, switch s and t, find the other, and finish
        else:
           
            # reverse s and t nodes and find a min cut on the other side of the slab 
            self.cut_value2, self.partition2 = nx.minimum_cut(
                    #self.slabgraphdirec,
                    #self.super_surface_node_max,
                    #self.super_surface_node_0,
                    Gred,
                    s,
                    t,
                    flow_func=nx.algorithms.flow.shortest_augmenting_path)

            Cp = set(self.get_edges_between_partitions(self.partition2))

            C  = list(C)
            Cp = list(Cp)

            self.all_cut_C0.append((C,Cp))
            self.all_cut_ids=["%05d"%0]


        print("\nAll final cutsets (%d):"%len(self.all_cut_C0))
        print(self.all_cut_C0)



    #def nx_min_cut_digraph_custom(self,weight_barrier=False):

    #    print("\n\nNx minimum_cut function on directed slab graph...")
    #    all_min_cutsets=[]
    #    curr_min_cutsets=[]

    #    # Firt create a barrier (aspect ratio of the slab is too large)
    #    if(weight_barrier):
    #        #self.create_weighted_barrier_on_opposite_half(start='min')
    #        self.create_weighted_barrier_on_super_surface_edges(super_surface_weight='inf')
    #    # uses stoer-wagner to do max flow (and indirectly min cut)
    #    # given source and target node
    #    self.cut_value1, self.partition1, self.sat_edges1= nxmfc.minimum_cut(
    #            self.slabgraphdirec,
    #            self.super_surface_node_0,
    #            self.super_surface_node_max,
    #            flow_func=nx.algorithms.flow.shortest_augmenting_path)

    #    print("Num saturated edges: %d (%d total edges)"%(len(self.sat_edges1),self.slabgraph.number_of_edges()))
    #    print("Actual min cut value: %d"%int(self.cut_value1))

    #    this_cutset=self.get_edges_between_partitions(self.partition1)
    #    all_min_cutsets.append(this_cutset.copy())
    #    curr_min_cutsets.append(this_cutset.copy())

    #    #print(curr_min_cutsets)
    #    while(len(curr_min_cutsets)!=0):
    #        print("Iterating cutset:" + str(curr_min_cutsets[0]))

    #        for edge in curr_min_cutsets[0]:
    #            print("Augmenting edge:"+str(edge))

    #            # augment capacity of one edge for the current cutset
    #            # this will destroy this cutset as the minimum cut, so we will
    #            # find another minimum cut with equal max flow if it exists
    #            self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity']+=1
    #            self.slabgraphdirec.edge[edge[1]][edge[0]]['capacity']+=1

    #            this_cut_value1, this_partition1, this_sat_edges= nxmfc.minimum_cut(
    #                    self.slabgraphdirec,
    #                    self.super_surface_node_0,
    #                    self.super_surface_node_max,
    #                    flow_func=nx.algorithms.flow.preflow_push)

    #            this_cutset1=self.get_edges_between_partitions(this_partition1)
    #            # reset the capacity back to the normal graph
    #            #self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity']-=1
    #            #self.slabgraphdirec.edge[edge[1]][edge[0]]['capacity']-=1

    #            # if this new cut has same minimum value AND
    #            # is unique to all previously identified minimum cutsets
    #            # keep it
    #            if(this_cut_value1 == self.cut_value1 and this_cutset1 not in all_min_cutsets):
    #                print("Found new cutset!:")
    #                print(str(this_cutset1))
    #                print(
    #                      [(self.slabgraph.node[n1]['ciflabel'],
    #                        self.slabgraph.node[n2]['ciflabel']) 
    #                       for (n1, n2) in this_cutset1] 
    #                     )
    #                # keep a copy of this new min cutset
    #                all_min_cutsets.append(this_cutset1)
    #                # add this new min cutset to the stack of cuts that need to be
    #                # augmented to find a new one
    #                curr_min_cutsets.append(this_cutset1)

    #                # reset the capacity back to the normal graph
    #                #self.slabgraphdirec.edge[edge[0]][edge[1]]['capacity']-=1
    #                #self.slabgraphdirec.edge[edge[1]][edge[0]]['capacity']-=1

    #                # move on to newest identified min cut
    #                break


    #        curr_min_cutsets.pop(0)

    #    print("Multiple min cutsets identified (%d):"%len(all_min_cutsets))
    #    print(all_min_cutsets)
    #    print([
    #            [(self.slabgraph.node[n1]['ciflabel'],
    #              self.slabgraph.node[n2]['ciflabel']) 
    #            for (n1, n2) in all_min_cutsets[i]] 
    #            for i in range(len(all_min_cutsets))
    #          ])
    #    print("Debugging structures to cif:")
    #    for i in range(len(all_min_cutsets)):
    #        # change element so we can vis it
    #        tmp_nodes=[]
    #        for (u,v) in all_min_cutsets[i]:
    #            # color nodes
    #            self.slabgraph.node[u]['element']='Ge'
    #            self.slabgraph.node[v]['element']='Ge'
    #            
    #            # color edges
    #            this_intersect = set(self.refgraph.neighbors(u)).intersection(self.refgraph.neighbors(v))     
    #            if(len(this_intersect) == 1):
    #                n_to_add=next(iter(this_intersect))
    #                this_data = self.refgraph.node[n_to_add]
    #                self.slabgraph.add_node(n_to_add, this_data)
    #                tmp_nodes.append(n_to_add)

    #        self.write_slabgraph_cif(self.cell,bond_block=False,descriptor="cutset%d"%i,relabel=False)
    #        # reset nodes
    #        for (u,v) in all_min_cutsets[i]:
    #            self.slabgraph.node[u]['element']='Si'
    #            self.slabgraph.node[v]['element']='Si'
    #        # reset edges
    #        for node in tmp_nodes:
    #            self.slabgraph.remove_node(node)
    #    
    #    # Debug vis all saturated edges' nodes
    #    tmp_nodes=[]
    #    for (u,v) in self.sat_edges1:
    #        # find common neighbor between u and v
    #        this_intersect = set(self.refgraph.neighbors(u)).intersection(self.refgraph.neighbors(v))     
    #        if(len(this_intersect) == 1):
    #            n_to_add=next(iter(this_intersect))
    #            this_data = self.refgraph.node[n_to_add]
    #            self.slabgraph.add_node(n_to_add, this_data)
    #            tmp_nodes.append(n_to_add)
    #    self.write_slabgraph_cif(self.cell,bond_block=False,descriptor="satedges",relabel=False)
    #    for node in tmp_nodes:
    #        self.slabgraph.remove_node(node)
    #        

    #    # wichever partition is the biggest is the one we keep
    #    # for now I am hoping that the algo always finds the symmetrically unique
    #    # cut CLOSEST to either the sink or source node
    #    if(len(self.partition1[0])>len(self.partition1[1])):
    #        self.keep_partition_1=self.partition1[0].copy()
    #        self.remove_partition_1=self.partition1[1].copy()
    #    else:
    #        self.keep_partition_1=self.partition1[1].copy()
    #        self.remove_partition_1=self.partition1[0].copy()

    #    # useful to also have a copy of the  partition sets with ciflabels
    #    self.partition1_labeled=[[],[]]
    #    for node in self.partition1[0]:
    #        self.partition1_labeled[0].append((node,self.refgraph.node[node]['ciflabel']))
    #    for node in self.partition1[1]:
    #        self.partition1_labeled[1].append((node,self.refgraph.node[node]['ciflabel']))


    #    print("\nForward s-t cut value:")
    #    print(self.cut_value1)        
    #    print("\nForward s-t cut edges:")
    #    self.forward_edges=self.get_edges_between_partitions(self.partition1)
    #    print(self.forward_edges)
    #    print("\nForward s-t cut labeled:")
    #    print([(self.slabgraph.node[n1]['ciflabel'],self.slabgraph.node[n2]['ciflabel']) for (n1, n2) in self.forward_edges])

    #    
    #    #print("Partition 1, set 0:")
    #    #print(self.partition1[0])
    #    #print("Partition 1, set 1:")
    #    #print(self.partition1[1])
    #    print("Forward cut (node, ciflabel) partition1:")
    #    print(self.partition1_labeled[0])
    #    print("Forward cut (node, ciflabel) partition2:")
    #    print(self.partition1_labeled[1])

    #    # remove the midpoint barrier
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='neutral')

    #    # now reverse the tree and reverse the source and target nodes
    #    # NOTE obsolete now that we have a directed graph and not a tree
    #    self.slabgraphdirecREV=self.slabgraphdirec.reverse(copy=True)
    #    #  create a barrier (aspect ratio of the slab is too large)
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='max')
    #    self.cut_value2, self.partition2, self.cutset2 = nxmfc.minimum_cut(
    #            self.slabgraphdirecREV,
    #            self.super_surface_node_max,
    #            self.super_surface_node_0,
    #            flow_func=nx.algorithms.flow.shortest_augmenting_path)

    #    # determine which are the removal/keep partitions
    #    if(len(self.partition2[0])>len(self.partition2[1])):
    #        self.keep_partition_2=self.partition2[0].copy()
    #        self.remove_partition_2=self.partition2[1].copy()
    #    else:
    #        self.keep_partition_2=self.partition2[1].copy()
    #        self.remove_partition_2=self.partition2[0].copy()

    #    # useful to also have a copy of the  partition sets with ciflabels
    #    self.partition2_labeled=[[],[]]
    #    for node in self.partition2[0]:
    #        self.partition2_labeled[0].append((node,self.refgraph.node[node]['ciflabel']))
    #    for node in self.partition2[1]:
    #        self.partition2_labeled[1].append((node,self.refgraph.node[node]['ciflabel']))


    #    print("\nReverse s-t cut value:")
    #    print(self.cut_value2)        
    #    print("\nReverse s-t cut edges:")
    #    print(self.get_edges_between_partitions(self.partition2))
    #    #print("Partition 2, set 0:")
    #    #print(self.partition2[0])
    #    #print("Partition 2, set 1:")
    #    #print(self.partition2[1])
    #    print("Reverse cut (node, ciflabel) partition1:")
    #    print(self.partition2_labeled[0])
    #    print("Reverse cut (node, ciflabel) partition2:")
    #    print(self.partition2_labeled[1])

    #    # remove the midpoint barrier
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='neutral')


    #def nx_min_cut_digraph(self,weight_barrier=False,layer_props=None):
    #    """
    #    Use the Nx minimum cut of a directed graph (forward and reverse) edges
    #    both exist with same weight, could equivalently be expressed as 
    #    an undirected graph and the stoer-wagner algorithm be used

    #    weight_barrier sets up a weight of inifinity for all edges whose nodes'
    #    vacuum coordinates are both < 0.5 to constrain the solution to one half 
    #    of the slab

    #    then the weight barrier reverses (coordiantes > 0.5) to find the solution
    #    on the other half of the slab
    #    """


    #    self.symmetric_cutsets=[]

    #    print("\n\nNx minimum_cut function on directed slab graph...")
    #    # Firt create a barrier (aspect ratio of the slab is too large)
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='min')
    #    # uses stoer-wagner to do max flow (and indirectly min cut)
    #    # given source and target node
    #    self.cut_value1, self.partition1 = nx.minimum_cut(
    #            self.slabgraphdirec,
    #            self.super_surface_node_0,
    #            self.super_surface_node_max,
    #            flow_func=nx.algorithms.flow.shortest_augmenting_path)

    #    # wichever partition is the biggest is the one we keep
    #    # for now I am hoping that the algo always finds the symmetrically unique
    #    # cut CLOSEST to either the sink or source node
    #    if(len(self.partition1[0])>len(self.partition1[1])):
    #        self.keep_partition_1=self.partition1[0].copy()
    #        self.remove_partition_1=self.partition1[1].copy()
    #    else:
    #        self.keep_partition_1=self.partition1[1].copy()
    #        self.remove_partition_1=self.partition1[0].copy()

    #    # useful to also have a copy of the  partition sets with ciflabels
    #    self.partition1_labeled=[[],[]]
    #    for node in self.partition1[0]:
    #        self.partition1_labeled[0].append((node,self.refgraph.node[node]['ciflabel']))
    #    for node in self.partition1[1]:
    #        self.partition1_labeled[1].append((node,self.refgraph.node[node]['ciflabel']))


    #    print("\nForward s-t cut value:")
    #    print(self.cut_value1)        
    #    #print("Partition 1, set 0:")
    #    #print(self.partition1[0])
    #    #print("Partition 1, set 1:")
    #    #print(self.partition1[1])
    #    #print("Forward cut (node, ciflabel) partition1:")
    #    #print(self.partition1_labeled[0])
    #    #print("Forward cut (node, ciflabel) partition2:")
    #    #print(self.partition1_labeled[1])
    #    print("\nForward s-t cut edges:")
    #    forward_edges=self.get_edges_between_partitions(self.partition1)
    #    print(forward_edges)
    #    print("\nSymmetric forward s-t cut edges")
    #    forward_edges_symm = self.id_symmetric_cut(forward_edges, layer_props)
    #    print(forward_edges_symm)

    #    self.symmetric_cutsets.append([forward_edges, forward_edges_symm])

    #    # remove the midpoint barrier
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='neutral')

    #    # now reverse the tree and reverse the source and target nodes
    #    self.slabgraphdirecREV=self.slabgraphdirec.reverse(copy=True)
    #    #  create a barrier (aspect ratio of the slab is too large)
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='max')
    #    self.cut_value2, self.partition2 = nx.minimum_cut(
    #            self.slabgraphdirecREV,
    #            self.super_surface_node_max,
    #            self.super_surface_node_0,
    #            flow_func=nx.algorithms.flow.shortest_augmenting_path)

    #    # determine which are the removal/keep partitions
    #    if(len(self.partition2[0])>len(self.partition2[1])):
    #        self.keep_partition_2=self.partition2[0].copy()
    #        self.remove_partition_2=self.partition2[1].copy()
    #    else:
    #        self.keep_partition_2=self.partition2[1].copy()
    #        self.remove_partition_2=self.partition2[0].copy()

    #    # useful to also have a copy of the  partition sets with ciflabels
    #    self.partition2_labeled=[[],[]]
    #    for node in self.partition2[0]:
    #        self.partition2_labeled[0].append((node,self.refgraph.node[node]['ciflabel']))
    #    for node in self.partition2[1]:
    #        self.partition2_labeled[1].append((node,self.refgraph.node[node]['ciflabel']))


    #    print("\nReverse s-t cut value:")
    #    print(self.cut_value2)        
    #    #print("Partition 2, set 0:")
    #    #print(self.partition2[0])
    #    #print("Partition 2, set 1:")
    #    #print(self.partition2[1])
    #    #print("Reverse cut (node, ciflabel) partition1:")
    #    #print(self.partition2_labeled[0])
    #    #print("Reverse cut (node, ciflabel) partition2:")
    #    #print(self.partition2_labeled[1])
    #    print("\nReverse s-t cut edges:")
    #    reverse_edges=self.get_edges_between_partitions(self.partition2)
    #    print(reverse_edges)
    #    print("\nSymmetric reverse s-t cut edges")
    #    reverse_edges_symm = self.id_symmetric_cut(reverse_edges, layer_props) 
    #    print(reverse_edges_symm)

    #    self.symmetric_cutsets.append([forward_edges, forward_edges_symm])

    #    # remove the midpoint barrier
    #    if(weight_barrier):
    #        self.create_weighted_barrier_on_opposite_half(start='neutral')

    ###########################################################################
    # START DEBUG visualization functions
    ###########################################################################
        
    def debug_min_cuts(self,cutset,step):
        """
        Add an O atom to slabgraph for every corresponding edge in the min cut
    
        """

        # reassign element type based on its slablayer
        for n,data in self.slabgraph.nodes_iter(data=True):
            slablayer=data['slablayer']
            # just to give it better coloring to start than an H atom in Mercury
            slablayer+=3
            data['element']=ATOMIC_NUMBER[slablayer]


        # if we have some other specialty debug features, add here
        # e.g. perhaps D4R units
        if(hasattr(self, 'all_D4R_nodes')):
            for n,data in self.slabgraph.nodes_iter(data=True):
                if(n in self.all_D4R_nodes):
                    data['element']=25


        tmp_nodes=[]
        for (u,v) in cutset:                                    
            # color nodes                                                   
            #self.slabgraph.node[u]['element']='Ge' 
            #self.slabgraph.node[v]['element']='Ge'                          
                                                                            
            # color edges                                                   
            this_intersect = set(self.refgraph.neighbors(u)).intersection(self.refgraph.neighbors(v))
            if(len(this_intersect) == 1):                                   
                n_to_add=next(iter(this_intersect))                         
                this_data = self.refgraph.node[n_to_add]                    
                self.slabgraph.add_node(n_to_add, this_data)                
                tmp_nodes.append(n_to_add)                                  

        
        # write a CIF file as an easy way to visualize the imoprtant info
        self.write_slabgraph_cif(self.slabgraph,self.cell,bond_block=False,\
                                 descriptor="cut%05d.debug"%step,relabel=False)

        # reset nodes                                                       
        for n,data in self.slabgraph.nodes_iter(data=True):
            data['element']=self.refgraph.node[n]['element']
        for node in tmp_nodes:                                              
            self.slabgraph.remove_node(node)


    def assign_slab_layers(self, layer_props):
        """
        Assing a new node attribute correspoding to its layer in the slab
        (the layer it belongs to must be calculated w/pymatgen
        """ 

        print("\nAssigning each atoms in slab to the correct slab layer")


        # NOTE important this is not rigorous since we are sorting possible identical c -values
        # need to get the layer data directly from pymatgen
        #tosort= [(n, data) for n, data in self.slabgraph.nodes_iter(data=True)]

        #sort_on_c=sorted(tosort, key=lambda tup: tup[1]['_atom_site_fract_z'])
   
        # 
        #for i in range(len(sort_on_c)):
        #    node=sort_on_c[i][0]
        #    slablayer=int(np.floor(i/layer_props['a_per_l']))
        #    self.slabgraph.node[node]['slablayer']=slablayer
        #    self.refgraph.node[node]['slablayer']=slablayer


        # NOTE a better way is to just make sure that the ordering is not sorted by Pymatgen
        # is every a_per_l atoms as we go from node 1 to N we know we have switched layers
        tosort= [(n, data) for n, data in self.slabgraph.nodes_iter(data=True)]

        sort_on_n=sorted(tosort, key=lambda tup: tup[0])
   
         
        for i in range(len(sort_on_n)):
            node=sort_on_n[i][0]
            slablayer=int(np.floor(i/layer_props['a_per_l']))
            #print(slablayer)
            self.slabgraph.node[node]['slablayer']=slablayer
            self.refgraph.node[node]['slablayer']=slablayer


        print("%d atoms in the slab"%len(sort_on_n))
        print("%d atoms per layer"%layer_props['a_per_l'])


    def debug_slab_layers(self):
        """
        Assign a false atom type based on the node's slablayer attribute
        so that we can vis what layer it's in
        """

        # reassign element type based on its slablayer
        for n,data in self.slabgraph.nodes_iter(data=True):
            slablayer=data['slablayer']
            #print(slablayer)
            # just to give it better coloring to start than an H atom in Mercury
            slablayer+=3
            data['element']=ATOMIC_NUMBER[slablayer]
            
        self.write_slabgraph_cif(self.slabgraph,self.cell,bond_block=False,descriptor="layer.debug",relabel=False)

        # reset element
        for n,data in self.slabgraph.nodes_iter(data=True):
            data['element']=self.refgraph.node[n]['element']

    ###########################################################################
    # END DEBUG visualization functions
    ###########################################################################

    def id_symmetric_cut(self, C, layer_props):
        """
        Find the identical cut based on inversion symmetry (inversion_mapping in layer_props)
        """

        C_symm = [(layer_props['inversion_mapping'][u],layer_props['inversion_mapping'][v]) for (u,v) in C]
        return C_symm

    def id_trans_layer_cut(self, C, layer_props):
        """
        Find the "opposite" cut based on the translational equivalent in the opposite layer
        """
        all_layers_in_cut = {}
        
        for (u,v) in C:
            all_layers_in_cut[self.slabgraph.node[u]['slablayer']]=None
            all_layers_in_cut[self.slabgraph.node[v]['slablayer']]=None

        # get a list of all layers spanned by cut
        all_layers_in_cut=sorted(all_layers_in_cut.keys())
        #print(all_layers_in_cut)

        # find the opposite layer for each layer of cut
        opp_layers=[(layer_props['slab_L']-1)-all_layers_in_cut[i] for i in range(len(all_layers_in_cut))]
        #print(opp_layers)        

        # now reverse the list of the opposite layers
        opp_layers=opp_layers[::-1]
        #print(opp_layers)        
        
        # the translational offset is the difference between the opposite and original cut layers for any index
        del_n = opp_layers[0]-all_layers_in_cut[0] 
        #print(del_n)

        # Tabulate the translational cut and return
        C_opp = [(u+layer_props['a_per_l']*del_n, v+layer_props['a_per_l']*del_n) for (u,v) in C]
        #C_opp = [(layer_props['translation_mapping'][u],layer_props['translation_mapping'][v]) for (u,v) in C]
        return C_opp

    ###########################################################################
    # START all functions for final manipulations to create valid slab
    ###########################################################################

    def generate_all_slabs_from_cutsets(self):
        """
        Now that all cutsets have been identified, we can create the final slabs
        """

        print("\nCreating a slab for each cutset")
        it=0
        for cutset in self.all_cut_C0:
            print("\nCutset %d:"%it)

            # duplicate copy that we can alter for each cutset
            tmpG = deepcopy(self.slabgraphdirec).to_undirected()
            print(type(tmpG))
            tmpG.name=str(self.slabgraphdirec.name)

            # delete both cuts of cutset, after which we remove the two components
            # containing the s an t (super surface nodes)
            tmpG = self.remove_surface_partitions_v2(tmpG, cutset)

            # for this temp graph, add back in all O's
            tmpG = self.add_all_connecting_nodes_v2(tmpG)

            # add final H caps
            tmpG = self.add_missing_hydrogens_v2(tmpG)

            # write this file
            thisid=self.all_cut_ids[it]
            self.write_slabgraph_cif(tmpG,self.cell,bond_block=False,descriptor="cut%s.addH"%thisid,relabel=False)
            #self.write_labgraph_cif(tmpG,self.cell,bond_block=False,descriptor="cut%s.addH"%thisid,relabel=False)
            self.write_silanol_surface_density(cutset,tmpG.name+".cut%s.addH"%thisid+".dat")
        
            it+=1
        

    def remove_surface_partitions_v2(self,G,onecutset):
        """
        disconnect a slabgraphdirec based on two cutsets, keep the con
        """

        print("\nAll nodes to keep after cutting (Si's)")
        cut0=onecutset[0]
        cut1=onecutset[1]

        if(nx.is_directed(G)):
            for (u,v) in cut0:
                G.remove_edge(u,v)
                G.remove_edge(v,u)
            for (u,v) in cut1:
                G.remove_edge(u,v)
                G.remove_edge(v,u)
        else:
            for (u,v) in cut0:
                G.remove_edge(u,v)
            for (u,v) in cut1:
                G.remove_edge(u,v)

        G=G.to_undirected()
        comps=nx.connected_components(G)
    
        nodes_to_remove=[]
        for comp in comps:
            if(self.super_surface_node_0 in comp or self.super_surface_node_max in comp):
                nodes_to_remove.extend(comp)

        for node in nodes_to_remove:
            G.remove_node(node)

        print("Total nodes kept: %d"%len(G.nodes()))

        return G

    def remove_surface_partitions(self):
        """
        Takes two partitions to remove from the slab graph
        one for each surface
        """
       
        # Note this will include the super surface nodes in the cut partition 
        self.all_removed_metal = self.remove_partition_1 | self.remove_partition_2

        print("\n\nAll Si/metal nodes to remove:")
        for node in self.all_removed_metal:
            print((node, self.refgraph.node[node]['ciflabel']))

        for node in self.all_removed_metal:
            self.slabgraph.remove_node(node)


    def add_all_connecting_nodes_v2(self, G):
        """
        Add all bridging nodes back into G that have been deleted in the original graph
        """
        
        print("\nAdd back in all connecting nodes (O's)")

        all_nodes=G.nodes()
        nodes_to_add=set()
        edges_to_add=[]
        for n1 in G.nodes_iter():
            for n2 in self.refgraph.neighbors(n1):
                # duplicate integers not created in a set
                nodes_to_add.add(n2)
                edges_to_add.append((n1,n2,self.refgraph.edge[n1][n2]))

        for n1 in nodes_to_add:
            G.add_node(n1,self.refgraph.node[n1])

        for n1, n2, data in edges_to_add:
            G.add_edge(n1,n2,data)

        print("Total nodes added: %d"%len(nodes_to_add))
        print("Total edges added: %d"%len(edges_to_add))

        return G

            
            

    def add_all_connecting_nodes(self):
       
        print("\n\nAdd back in the missing O's")

        # these are all the previously removed oxygens that need to be readded
        self.final_added_nodes = set()
        # these are all the edges that contained a removed oxygen that needs to be readded
        self.final_added_edges = []
        # these are all the edges that contained a removed oxygen that needs to be readded AND a reomoved Si 
        self.final_H_edges = []

        # self.removed_edges is any Si-O edge removed in intial graph condensation
        for i in range(len(self.removed_edges)):
            n1 = self.removed_edges[i][0]
            n2 = self.removed_edges[i][1]

    
            if n1 in self.slabgraph.nodes():
                if(n1 not in self.final_added_nodes):
                    # add the missing O node
                    #print("Adding an O (node %d, ciflabel %s)"%(n2,self.refgraph.node[n2]['ciflabel']))
                    self.slabgraph.add_node(n2, self.refgraph.node[n2])
                    # add its edge to Si
                    self.slabgraph.add_edge(n1,n2,self.removed_edges_data[i])
                    # keep track of its addition so we don't try it again
                    self.final_added_nodes.add(n2)
     
                    # check to see if the O we are adding is attached to a removed Si 
                    this_intersect = set(self.refgraph.neighbors(n2)).intersection(self.all_removed_metal)     
                    if(len(this_intersect)==1):
                        this_edge=(n2, next(iter(this_intersect)))
                        self.final_H_edges.append(this_edge)
                        print("Adding a forward edge for new O-H edge")
                        print(this_edge)
                    else:
                        pass
                        #print(this_intersect)
                        #print("Warning! Seems as though an Oxygen was identified for capping that was not attached to Si in the original graph... Something weird is probably going to happen...")
                    #print("Added!")
                    #print(self.slabgraph.node[n2])
                    #print(self.slabgraph.node[n2]['element'])
                    #print(self.final_added_nodes)
            elif n2 in self.slabgraph.nodes():
                if(n2 not in self.final_added_nodes):
                    # add the missing O node
                    #print("Adding an O (node %d, ciflabel %s)"%(n1, self.refgraph.node[n1]['ciflabel']))
                    self.slabgraph.add_node(n1, self.refgraph.node[n1])
                    # add its edge to Si
                    self.slabgraph.add_edge(n1,n2,self.removed_edges_data[i])
                    # keep track of its addition so we don't try it again
                    self.final_added_nodes.add(n1)

                    # check to see if the O we are adding is attached to a removed Si 
                    if(len(this_intersect)==1):
                        this_edge=((n1, next(iter(this_intersect))))
                        self.final_H_edges.append(this_edge)
                        print("Adding a reverse edge for new O-H edge")
                        print(this_edge)
                    else:
                        pass
                        #print(this_intersect)
                        #print("Warning! Seems as though an Oxygen was identified for capping that was not attached to Si in the original graph... Something weird is probably going to happen...")
                    #print("Added!")
                    #print(self.slabgraph.node[n1])
                    #print(self.slabgraph.node[n1]['element'])
                    #print(self.final_added_nodes)

        #print("Final added O's:" + str(self.final_added_nodes))
        print("%d O-Si bonds to convert to O-H:"%len(self.final_H_edges))
        print(self.final_H_edges)
        self.final_H_edges_ciflabel=[]
        for edge in self.final_H_edges:
            self.final_H_edges_ciflabel.append((self.refgraph.node[edge[0]]['ciflabel'],
                                                self.refgraph.node[edge[1]]['ciflabel']))
        print(self.final_H_edges_ciflabel)
                

        # NOTE doesn't work
        #for node in self.removed_nodes:
        #    # for each neighbor of node in the parent class Molecular Graph
        #    for neigh in self.neighbors(node):
        #        if neigh in self.slabgraph.nodes():
        #            self.final_added_nodes.append(self.graph.get_node(neigh))


        #for node in self.final_added_nodes:
        #    print(node)
        #    self.slabgraph.add_node(node)

    def add_missing_hydrogens_v2(self,G,debug=False):
        """
        Identify any undercoordinated sites in G wrt to the referenece graph

        Any missing nodes correspond to something that needs to be a hydrogen
        """

        all_nodes=set(G.nodes())
        final_H_edges=[]

        # all nodes in current graph
        for n1 in all_nodes:
            # all neightbors of this node in reference graph
            for n2 in self.refgraph.neighbors(n1):      
                # if this neighbor is missing in current graph
                if(n2 not in all_nodes):
                    # This node should be turned into an H
                    # Note this is stored as (0, Si) which is being transformed to (O,H)
                    final_H_edges.append((n1,n2))

        # the start index for each added H
        this_index=self.num_nodes+1
        # for each directed edge representing a converted O->Si to O->H
        for edge in final_H_edges:
            if(debug):
                print("\n")
                print(edge)
                print((self.refgraph.node[edge[0]]['ciflabel'],self.refgraph.node[edge[1]]['ciflabel']))
            # data of parent node (the surface O atom)
            parent_node_data=self.refgraph.node[edge[0]]
            # data of child node (the Si atom to be turned into H)
            old_child_node_data=self.refgraph.node[edge[1]]
            new_child_node_index = int(this_index)
            new_child_node_data = old_child_node_data.copy()
            new_child_node_data['element']="H"
            new_child_node_data['ciflabel']="H"+str(new_child_node_index)
   
            # get original edge data to check for periodicity
            old_edge_data=self.refgraph.edge[edge[0]][edge[1]]
            symflag=old_edge_data['symflag']

            # New/old coordinates of child node to write in file
            old_child_abc = [float(old_child_node_data['_atom_site_fract_x']),
                       float(old_child_node_data['_atom_site_fract_y']),
                       float(old_child_node_data['_atom_site_fract_z'])] 

            new_child_abc = deepcopy(old_child_abc)

            if(debug):
                print(symflag)
            if(symflag != '.'):
                # if periodic in x direction
                if(symflag[2]=='6' or symflag[2]=='4'):
                    if(new_child_node_data['_atom_site_fract_x'] <
                       parent_node_data['_atom_site_fract_x']):
                        new_child_abc[0]+=1.0
                    else:
                        new_child_abc[0]-=1.0

                # periodic in y direction
                elif(symflag[3]=='6' or symflag[3]=='4'):
                    if(new_child_node_data['_atom_site_fract_y'] <
                       parent_node_data['_atom_site_fract_y']):
                        new_child_abc[1]+=1.0
                    else:
                        new_child_abc[1]-=1.0
                
                # periodic in z direction
                elif(symflag[4]=='6' or symflag[4]=='4'):
                    if(new_child_node_data['_atom_site_fract_z'] <
                       parent_node_data['_atom_site_fract_z']):
                        new_child_abc[2]+=1.0
                    else:
                        new_child_abc[2]-=1.0

            if(debug):
                print("Old child abc:")
                print(old_child_abc)
                print("New child abc:")
                print(new_child_abc)

            new_child_xyz=self.to_cartesian(new_child_abc)
            if(debug):
                print("New child xyz:")
                print(new_child_xyz)
               
                print("Parent abc:")
            parent_abc=np.array([float(parent_node_data['_atom_site_fract_x']),
                                 float(parent_node_data['_atom_site_fract_y']),
                                 float(parent_node_data['_atom_site_fract_z'])])
            if(debug):
                print(parent_abc)
                print("Stored parent xyz:")
                print(parent_node_data['cartesian_coordinates'])
                print("Stored parent abc:")
                print(self.to_fractional(parent_node_data['cartesian_coordinates']))
                print("Calculated parent xyz:")
            updated_parent_xyz=self.to_cartesian(parent_abc)
            if(debug):
                print(updated_parent_xyz)
                print("Calculated parent abc:")
                print(self.to_fractional(updated_parent_xyz))
            #parent_node_data['cartesian_coordinates']=updated_parent_xyz.copy()

            bond_length=np.linalg.norm(new_child_xyz-parent_node_data['cartesian_coordinates'])
            if(debug):
                print("old bond length:")
                print(bond_length)
            h_bond_length=0.95
            scale_by=h_bond_length/bond_length
            if(debug):
                print("Scale by")
                print(scale_by)

            new_child_xyz=parent_node_data['cartesian_coordinates']+\
                    (new_child_xyz-parent_node_data['cartesian_coordinates'])*scale_by
            if(debug):
                print("New child xyz:")
                print(new_child_xyz)

            new_child_xyz=self.in_cell(new_child_xyz)
            if(debug):
                print("New child xyz in cell:")
                print(new_child_xyz)

            new_child_abc = self.to_fractional(new_child_xyz)
            if(debug):
                print("New child abc for printing:")
                print(new_child_abc)

            new_child_node_data['_atom_site_fract_x'] = str(new_child_abc[0])
            new_child_node_data['_atom_site_fract_y'] = str(new_child_abc[1])
            new_child_node_data['_atom_site_fract_z'] = str(new_child_abc[2])
            new_child_node_data['cartesian_coordinates']=new_child_xyz
            G.add_node(new_child_node_index, new_child_node_data)
            if(debug):
                print("Adding child H node w/index %d"%new_child_node_index)

            this_index+=1

        return G    

    def add_missing_hydrogens(self,debug=False):
        """
        Add back in the missing Hydrogens
        Need to be very careful because the Hydrogen to add could be attached
        to an oxygen across a periodic boundary
        """
        
        # check if an O was added on the surface

        print("\n\nAdding back in H's")
        print(self.final_H_edges)

        this_index = self.num_nodes+1
        # for each directed edge representing a converted O->Si to O->
        for edge in self.final_H_edges:
            if(debug):
                print("\n")
                print(edge)
                print((self.refgraph.node[edge[0]]['ciflabel'],self.refgraph.node[edge[1]]['ciflabel']))
            # data of parent node (the surface O atom)
            parent_node_data=self.refgraph.node[edge[0]]
            # data of child node (the Si atom to be turned into H)
            old_child_node_data=self.refgraph.node[edge[1]]
            new_child_node_index = int(this_index)
            new_child_node_data = old_child_node_data.copy()
            new_child_node_data['element']="H"
            new_child_node_data['ciflabel']="H"+str(new_child_node_index)
   
            # get original edge data to check for periodicity
            old_edge_data=self.refgraph.edge[edge[0]][edge[1]]
            symflag=old_edge_data['symflag']

            # New/old coordinates of child node to write in file
            old_child_abc = [float(old_child_node_data['_atom_site_fract_x']),
                       float(old_child_node_data['_atom_site_fract_y']),
                       float(old_child_node_data['_atom_site_fract_z'])] 

            new_child_abc = deepcopy(old_child_abc)

            if(debug):
                print(symflag)
            if(symflag != '.'):
                # if periodic in x direction
                if(symflag[2]=='6' or symflag[2]=='4'):
                    if(new_child_node_data['_atom_site_fract_x'] <
                       parent_node_data['_atom_site_fract_x']):
                        new_child_abc[0]+=1.0
                    else:
                        new_child_abc[0]-=1.0

                # periodic in y direction
                elif(symflag[3]=='6' or symflag[3]=='4'):
                    if(new_child_node_data['_atom_site_fract_y'] <
                       parent_node_data['_atom_site_fract_y']):
                        new_child_abc[1]+=1.0
                    else:
                        new_child_abc[1]-=1.0
                
                # periodic in z direction
                elif(symflag[4]=='6' or symflag[4]=='4'):
                    if(new_child_node_data['_atom_site_fract_z'] <
                       parent_node_data['_atom_site_fract_z']):
                        new_child_abc[2]+=1.0
                    else:
                        new_child_abc[2]-=1.0

            if(debug):
                print("Old child abc:")
                print(old_child_abc)
                print("New child abc:")
                print(new_child_abc)

            new_child_xyz=self.to_cartesian(new_child_abc)
            if(debug):
                print("New child xyz:")
                print(new_child_xyz)
               
                print("Parent abc:")
            parent_abc=np.array([float(parent_node_data['_atom_site_fract_x']),
                                 float(parent_node_data['_atom_site_fract_y']),
                                 float(parent_node_data['_atom_site_fract_z'])])
            if(debug):
                print(parent_abc)
                print("Stored parent xyz:")
                print(parent_node_data['cartesian_coordinates'])
                print("Stored parent abc:")
                print(self.to_fractional(parent_node_data['cartesian_coordinates']))
                print("Calculated parent xyz:")
            updated_parent_xyz=self.to_cartesian(parent_abc)
            if(debug):
                print(updated_parent_xyz)
                print("Calculated parent abc:")
                print(self.to_fractional(updated_parent_xyz))
            #parent_node_data['cartesian_coordinates']=updated_parent_xyz.copy()

            bond_length=np.linalg.norm(new_child_xyz-parent_node_data['cartesian_coordinates'])
            if(debug):
                print("old bond length:")
                print(bond_length)
            h_bond_length=0.95
            scale_by=h_bond_length/bond_length
            if(debug):
                print("Scale by")
                print(scale_by)

            new_child_xyz=parent_node_data['cartesian_coordinates']+\
                    (new_child_xyz-parent_node_data['cartesian_coordinates'])*scale_by
            if(debug):
                print("New child xyz:")
                print(new_child_xyz)

            new_child_xyz=self.in_cell(new_child_xyz)
            if(debug):
                print("New child xyz in cell:")
                print(new_child_xyz)

            new_child_abc = self.to_fractional(new_child_xyz)
            if(debug):
                print("New child abc for printing:")
                print(new_child_abc)

            new_child_node_data['_atom_site_fract_x'] = str(new_child_abc[0])
            new_child_node_data['_atom_site_fract_y'] = str(new_child_abc[1])
            new_child_node_data['_atom_site_fract_z'] = str(new_child_abc[2])
            new_child_node_data['cartesian_coordinates']=new_child_xyz
            self.slabgraph.add_node(new_child_node_index, new_child_node_data)
            if(debug):
                print("Adding child H node w/index %d"%new_child_node_index)

            this_index+=1
             
    def to_cartesian(self, coord):
        """
        return unwrapped cartesian coords from abc
        """
        return np.dot(coord,self.cell.cell)
                    
    def to_fractional(self, coord):
        f = np.dot(self.cell.inverse, coord) 
        return f 

    def write_silanol_surface_density(self, C, ofname):

        unit_area=self.cell.a*self.cell.b                                   
        # correct if this face is not orthogonal                                
        unit_area*=np.sin(np.deg2rad(self.cell.gamma))                          

        cut_weight=sum([self.slabgraph.edge[edg[0]][edg[1]]['weight'] for edg in C[0]])
        
        # NOTE changed to sum of edge weights instead of # of edges
        #per_surface_density=sum(C[0])/(unit_area)
        per_surface_density=cut_weight/(unit_area)

        f=open(ofname,"w")
        f.write("%d %.5f %.5f"%(len(C[0]), unit_area, per_surface_density))
        f.close()


    def write_slabgraph_cif(self, G, cell,bond_block=True,descriptor="debug",relabel=False):
        write_CIF(G,cell,bond_block,descriptor,relabel)

    def check_approximate_slab_thickness(self):
        """
        return the approximate slab thickness in the vacuum direction
        """
        min_coord=1.0
        max_coord=0.0

        for node, data in self.slabgraph.nodes_iter(data=True):
            if(self.vacuum_direc==0):
                if(float(data['_atom_site_fract_x'])>max_coord):
                    max_coord=float(data['_atom_site_fract_x'])
                if(float(data['_atom_site_fract_x'])<min_coord):
                    min_coord=float(data['_atom_site_fract_x'])
            elif(self.vacuum_direc==1):
                if(float(data['_atom_site_fract_y'])>max_coord):
                    max_coord=float(data['_atom_site_fract_y'])
                if(float(data['_atom_site_fract_y'])<min_coord):
                    min_coord=float(data['_atom_site_fract_y'])
            elif(self.vacuum_direc==2):
                if(float(data['_atom_site_fract_z'])>max_coord):
                    max_coord=float(data['_atom_site_fract_z'])
                if(float(data['_atom_site_fract_z'])<min_coord):
                    min_coord=float(data['_atom_site_fract_z'])

        if(self.vacuum_direc==0):
            approximate=(max_coord-min_coord)*self.cell.a
        elif(self.vacuum_direc==1):
            approximate=(max_coord-min_coord)*self.cell.b
        elif(self.vacuum_direc==2):
            approximate=(max_coord-min_coord)*self.cell.c
       
        print("Maximal slab distance ~ %.3f"%approximate) 
        return approximate

    def check_approximate_slab_thickness_v2(self):
        """
        return the approximate MINIMUM slab distance in the vacuum direction
        
        iterate over all set(surface_0_nodes) and set(surface_max_nodes) combinations
        and find the minimum euclidian distance
        """

        approx_min_dist = 10000.0
        for n1 in self.remove_partition_1:
            for n2 in self.remove_partition_2:
                #print(self.refgraph.node[n1]['cartesian_coordinates'])
                eucl_dist = np.linalg.norm(self.refgraph.node[n1]['cartesian_coordinates']-
                                           self.refgraph.node[n2]['cartesian_coordinates'])
                if(eucl_dist < approx_min_dist):
                    approx_min_dist = float(eucl_dist)

        print("Minimal slab distance ~ %.3f"%approx_min_dist)
        return approx_min_dist


    def check_slab_is_2D_periodic(self):
        """
        Check if slab is 2D periodic in the two dimensions other than the vacuum_direc
        """
        

        periodic_a=False
        periodic_b=False
        periodic_c=False
        periodic_2D=False

        for n1,n2,data in self.slabgraph.edges_iter(data=True):
            # having a O-H across a PBC doesn't count as being a periodic structure
            #print(n1,n2)
            if(self.slabgraph.node[n1]['element']!="H" and
               self.slabgraph.node[n2]['element']!="H"):
        
                if(data['symflag']=='1_455' or data['symflag']=='1_655'):
                    periodic_a=True
                elif(data['symflag']=='1_545' or data['symflag']=='1_565'):
                    periodic_b=True
                elif(data['symflag']=='1_554' or data['symflag']=='1_556'):
                    periodic_c=True

        if(self.vacuum_direc==0):
            if(periodic_b and periodic_c):
                periodic_2D=True
        elif(self.vacuum_direc==1):
            if(periodic_a and periodic_c):
                periodic_2D=True
        elif(self.vacuum_direc==2):
            if(periodic_a and periodic_b):
                periodic_2D=True

        return periodic_2D

    def write_average_silanol_density(self,ofname):
        """
        For now do a simple metric for the silanol density
        """
        print(self.cell.gamma)
        unit_area=0.0
        num_H_added = len(self.final_H_edges)
        if(self.vacuum_direc==0):
            unit_area=self.cell.b*self.cell.c
        elif(self.vacuum_direc==1):
            unit_area=self.cell.a*self.cell.c
        elif(self.vacuum_direc==2):
            unit_area=self.cell.a*self.cell.b

        # correct if this face is not orthogonal
        unit_area*=np.sin(np.deg2rad(self.cell.gamma))
        per_surface_density=(num_H_added/2)/(unit_area)

        print("Average surface density: %.7f"%(per_surface_density))
        f=open(ofname, "w")
        f.write("%.7f"%per_surface_density)
        

    def enumerate_all_primitive_rings(self):
        """
        Iterate through all minimal cycles and get statistics
        """

        max_cycle_length=0
        min_cycle_length=1000000
        total_rings=0
        for node, data in self.slabgraph.nodes_iter(data=True):
            print("Node %d is a part of all cycles:"%(node))
            for ring in data['rings']:
                print(ring)
                if(len(ring) > max_cycle_length):
                    max_cycle_length=len(ring)
                if(len(ring) < min_cycle_length):
                    min_cycle_length=len(ring)
                total_rings+=1

        print("Min cycle length: " + str(min_cycle_length))
        print("Max cycle length: " + str(max_cycle_length))
        print("Total rings: "      + str(total_rings))


# END SLAB GRAPH CLASS

def del_parenth(string):
    return re.sub(r'\([^)]*\)', '' , string)

def from_CIF(cifname):
    """Reads the structure data from the CIF
    - currently does not read the symmetry of the cell
    - does not unpack the assymetric unit (assumes P1)
    - assumes that the appropriate keys are in the cifobj (no error checking)
    """

    cifobj = CIF()
    cifobj.read(cifname)

    data = cifobj._data
    # obtain atoms and cell
    cell = Cell()
    # add data to molecular graph (to be parsed later..)
    mg = MolecularGraph(name=clean(cifname))

    # matches any integer values inside brackets
    cellparams = [float(del_parenth(i)) for i in [data['_cell_length_a'], 
                                     data['_cell_length_b'], 
                                     data['_cell_length_c'],
                                     data['_cell_angle_alpha'], 
                                     data['_cell_angle_beta'], 
                                     data['_cell_angle_gamma']]]
    cell.set_params(cellparams)
    
    #add atom nodes
    id = cifobj.block_order.index('atoms')
    atheads = cifobj._headings[id]
    for atom_data in zip(*[data[i] for i in atheads]):
        kwargs = {a:j.strip() for a, j in zip(atheads, atom_data)}
        mg.add_atomic_node(**kwargs)
        

    # add bond edges, if they exist
    try:
        id = cifobj.block_order.index('bonds')
        bondheads = cifobj._headings[id]
        for bond_data in zip(*[data[i] for i in bondheads]):
            kwargs = {a:j.strip() for a, j in zip(bondheads, bond_data)}
            mg.add_bond_edge(**kwargs)
    except:
        # catch no bonds
        print("No bonds reported in cif file - computing bonding..")
    mg.store_original_size()
    mg.cell = cell
    return cell, mg

def write_CIF(G, cell, bond_block=True,descriptor="debug",relabel=True):
    """Currently used for debugging purposes"""

    if(descriptor==None):
        c = CIF(name="%s"%(G.name))
    else:
        c = CIF(name="%s.%s"%(G.name,descriptor))

    # data block
    c.add_data("data", data_=G.name)
    c.add_data("data", _audit_creation_date=
                        CIF.label(c.get_time()))
    c.add_data("data", _audit_creation_method=
                        CIF.label("Lammps Interface v.%s"%(str(0))))

    # sym block
    c.add_data("sym", _symmetry_space_group_name_H_M=
                        CIF.label("P1"))
    c.add_data("sym", _symmetry_Int_Tables_number=
                        CIF.label("1"))
    c.add_data("sym", _symmetry_cell_setting=
                        CIF.label("triclinic"))

    # sym loop block
    c.add_data("sym_loop", _symmetry_equiv_pos_as_xyz=
                        CIF.label("'x, y, z'"))

    # cell block
    c.add_data("cell", _cell_length_a=CIF.cell_length_a(cell.a))
    c.add_data("cell", _cell_length_b=CIF.cell_length_b(cell.b))
    c.add_data("cell", _cell_length_c=CIF.cell_length_c(cell.c))
    c.add_data("cell", _cell_angle_alpha=CIF.cell_angle_alpha(cell.alpha))
    c.add_data("cell", _cell_angle_beta=CIF.cell_angle_beta(cell.beta))
    c.add_data("cell", _cell_angle_gamma=CIF.cell_angle_gamma(cell.gamma))
    # atom block
    element_counter = {}
    carts = []
    for node, data in G.nodes_iter(data=True):
        # can override write to try to preserve the already assigned ciflabel
        if(relabel==True):
            label = "%s%i"%(data['element'], node)
        else:
            try:
                label = "%s"%(data['ciflabel'])
            except:
                label = "%s%i"%(data['ciflabel'], node)


        c.add_data("atoms", _atom_site_label=
                                CIF.atom_site_label(label))
        c.add_data("atoms", _atom_site_type_symbol=
                                CIF.atom_site_type_symbol(data['element']))
        try:
            c.add_data("atoms", _atom_site_description=
                                CIF.atom_site_description(data['force_field_type']))
        except KeyError:
            c.add_data("atoms", _atom_site_description=
                                CIF.atom_site_description(data['element']))

        coords = data['cartesian_coordinates']
        carts.append(coords)
        fc = np.dot(cell.inverse, coords) 
        c.add_data("atoms", _atom_site_fract_x=
                                CIF.atom_site_fract_x(fc[0]))
        c.add_data("atoms", _atom_site_fract_y=
                                CIF.atom_site_fract_y(fc[1]))
        c.add_data("atoms", _atom_site_fract_z=
                                CIF.atom_site_fract_z(fc[2]))
        try:
            c.add_data("atoms", _atom_type_partial_charge=
                                CIF.atom_type_partial_charge(data['charge']))
        except KeyError:
            c.add_data("atoms", _atom_type_partial_charge="0.0")
    # bond block
    # must re-sort them based on bond type (Mat Sudio)
    if(bond_block):
        tosort = [(data['order'], (n1, n2, data)) for n1, n2, data in G.edges_iter2(data=True)]
        for ord, (n1, n2, data) in sorted(tosort, key=lambda tup: tup[0]):
            type = CCDC_BOND_ORDERS[data['order']]
            dist = data['length'] 
            sym = data['symflag']


            label1 = "%s%i"%(G.node[n1]['element'], n1)
            label2 = "%s%i"%(G.node[n2]['element'], n2) 
            c.add_data("bonds", _geom_bond_atom_site_label_1=
                                        CIF.geom_bond_atom_site_label_1(label1))
            c.add_data("bonds", _geom_bond_atom_site_label_2=
                                        CIF.geom_bond_atom_site_label_2(label2))
            c.add_data("bonds", _geom_bond_distance=
                                        CIF.geom_bond_distance(dist))
            c.add_data("bonds", _geom_bond_site_symmetry_2=
                                        CIF.geom_bond_site_symmetry_2(sym))
            c.add_data("bonds", _ccdc_geom_bond_type=
                                        CIF.ccdc_geom_bond_type(type))
    
    print('Output cif file written to %s.cif'%c.name)
    file = open("%s.cif"%c.name, "w")
    file.writelines(str(c))
    file.close()

def write_RASPA_CIF(graph, cell, classifier=0):
    """
    Same as debugging cif write routine
        except _atom_site_label is now equal to _atom_site_description
        b/c RASPA uses _atom_site_label as the type for assigning FF params
    """
    c = CIF(name="%s_raspa"%graph.name)
    # data block
    c.add_data("data", data_=graph.name)
    c.add_data("data", _audit_creation_date=
                        CIF.label(c.get_time()))
    c.add_data("data", _audit_creation_method=
                        CIF.label("Lammps Interface v.%s"%(str(0))))

    # sym block
    c.add_data("sym", _symmetry_space_group_name_H_M=
                        CIF.label("P1"))
    c.add_data("sym", _symmetry_Int_Tables_number=
                        CIF.label("1"))
    c.add_data("sym", _symmetry_cell_setting=
                        CIF.label("triclinic"))

    # sym loop block
    c.add_data("sym_loop", _symmetry_equiv_pos_as_xyz=
                        CIF.label("'x, y, z'"))

    # cell block
    c.add_data("cell", _cell_length_a=CIF.cell_length_a(cell.a))
    c.add_data("cell", _cell_length_b=CIF.cell_length_b(cell.b))
    c.add_data("cell", _cell_length_c=CIF.cell_length_c(cell.c))
    c.add_data("cell", _cell_angle_alpha=CIF.cell_angle_alpha(cell.alpha))
    c.add_data("cell", _cell_angle_beta=CIF.cell_angle_beta(cell.beta))
    c.add_data("cell", _cell_angle_gamma=CIF.cell_angle_gamma(cell.gamma))
    # atom block
    element_counter = {}
    carts = []
    
    # slight modification to make sure atoms printed out in same order as in data and original cif
    for node, data in sorted(graph.nodes_iter(data=True)):
        label = "%s%i"%(data['element'], node)

        # sometimes we need to keep the original CIF label in the case          
        # of slab geometries where charge symmetry is very reduced              
        if(classifier==0):                                                      
            c.add_data("atoms", _atom_site_label=                               
                              CIF.atom_site_label(data['force_field_type']))    
        elif(classifier==1):                                                    
            c.add_data("atoms", _atom_site_label=                               
                              CIF.atom_site_label(data['ciflabel']))        

        c.add_data("atoms", _atom_site_type_symbol=
                                CIF.atom_site_type_symbol(data['element']))
        #c.add_data("atoms", _atom_site_description=
        #                        CIF.atom_site_description(data['force_field_type']))
        coords = data['cartesian_coordinates']
        carts.append(coords)
        fc = np.dot(cell.inverse, coords) 
        c.add_data("atoms", _atom_site_fract_x=
                                CIF.atom_site_fract_x(fc[0]))
        c.add_data("atoms", _atom_site_fract_y=
                                CIF.atom_site_fract_y(fc[1]))
        c.add_data("atoms", _atom_site_fract_z=
                                CIF.atom_site_fract_z(fc[2]))
        c.add_data("atoms", _atom_site_charge=
                                CIF.atom_type_partial_charge(data['charge']))
    # bond block
    # must re-sort them based on bond type (Mat Sudio)
    #tosort = [(data['order'], (n1, n2, data)) for n1, n2, data in graph.edges_iter2(data=True)]
    #for ord, (n1, n2, data) in sorted(tosort, key=lambda tup: tup[0]):
    #    type = CCDC_BOND_ORDERS[data['order']]
    #    dist = data['length'] 
    #    sym = data['symflag']


    #    label1 = "%s%i"%(graph.node[n1]['element'], n1)
    #    label2 = "%s%i"%(graph.node[n2]['element'], n2) 
    #    c.add_data("bonds", _geom_bond_atom_site_label_1=
    #                                CIF.geom_bond_atom_site_label_1(label1))
    #    c.add_data("bonds", _geom_bond_atom_site_label_2=
    #                                CIF.geom_bond_atom_site_label_2(label2))
    #    c.add_data("bonds", _geom_bond_distance=
    #                                CIF.geom_bond_distance(dist))
    #    c.add_data("bonds", _geom_bond_site_symmetry_2=
    #                                CIF.geom_bond_site_symmetry_2(sym))
    #    c.add_data("bonds", _ccdc_geom_bond_type=
    #                                CIF.ccdc_geom_bond_type(type))
     
    print('Output cif file written to %s.cif'%c.name)
    file = open("%s.cif"%c.name, "w")
    file.writelines(str(c))
    file.close()

def write_RASPA_sim_files(lammps_sim, classifier=0):
    """
    Write the RASPA pseudo_atoms.def file for this MOF
    All generic adsorbates info automatically included
    Additional framework atoms taken from lammps_interface anaylsis
    """

    MOF_PSEUDO_ATOMS = []
    MOF_FF_MIXING = []

    data_list = []                                                              
    max_image = 0                                                               
    if(classifier == 0):                                                        
        for node, data in sorted(lammps_sim.unique_atom_types.items()):         
            #print(type(node))
            #print(type(data))                                                         
            #print(node)                                                          
            #print(data)                                                         
            # Note here that data is a tuple of (node, data)
            data[1]['node']=node                                                   
            data_list.append(data[1])                                              
                                                                                
            if(int(data[1]['image']) > max_image):                                 
                max_image = int(data[1]['image'])                                  
                                                                                
    elif(classifier == 1):                                                      
        for node, data in sorted(lammps_sim.graph.nodes_iter(data=True)):       
            data['node']=node                                                   
            #print(node)                                                         
            #print(data)                                                         
            data_list.append(data)                                              
                                                                                
            if(int(data['image']) > max_image):                                 
                max_image = int(data['image'])                                  
                                                                                
    print(max_image)                                                            
    for i in range(len(data_list)):                                             
                                                                                
        #ind = 0                                                                
        ##for char in key:                                                      
        #    # identify first non alphabetic character in string                
        #    if((ord(char) >= 65 and ord(char) <= 90) or (ord(char)>=97 and ord(char) <= 122)):
        #        ind += 1                                                       
        #    else:                                                              
        #        break                                                          
        #atmtype_ = key[:ind]                                                   
        #print(key,atmtype_)                                                    
        #print(atmtype_ in MASS)                                                
        #print(atmtype_ in COVALENT_RADII)                                      
        #print(key in UFF_DATA)                                                 
                                                                                
        if(data_list[i]['node']<max_image):                                     
            element_ = data_list[i]['element']                                  
            fftype_ = data_list[i]['force_field_type']                          
                                                                                
            try:                                                                
                if(classifier==0):                                              
                    type_spec_ = data_list[i]['force_field_type']               
                elif(classifier==1):                                            
                    type_spec_ = data_list[i]['ciflabel']                       
                                                                                
                print_ = 'yes'                                                  
                as_ = element_                                                  
                chem_ = element_                                                
                oxidation_ = '0'                                                
                mass_ = str(MASS[element_])                                     
                charge_ = str(data_list[i]['charge'])                           
                polarization_ = '0.0'                                           
                B_factor_ = '1.0'                                               
                radii_ = str(COVALENT_RADII[element_])                          
                connectivity_ = '0'                                             
                anisotropic_ = '0'                                              
                anisotropic_type_ = 'relative'                                  
                tinker_type_ = '0'                                              
                                                                                
                potential_ = 'lennard-jones'                                    
                eps_ = str(UFF_DATA[fftype_][3]*500)                            
                sig_ = str(UFF_DATA[fftype_][2]*(2**(-1./6.)))                  
            except:                                                             
                print("%s %s not found!\n"%(element_, fftype_))                 
                continue                                                        
                                                                                
            # finally, add strings to list for each RASPA file                  
            MOF_PSEUDO_ATOMS.append([type_spec_, print_, as_,chem_, oxidation_,mass_, charge_, \
                                     polarization_, B_factor_, radii_, connectivity_, \
                                     anisotropic_, anisotropic_type_, tinker_type_])
                                                                                
            MOF_FF_MIXING.append([type_spec_, potential_, eps_, sig_])     

    if(len(MOF_PSEUDO_ATOMS) == 0):
        print("ERROR: No MOF atoms found. Exiting...")
        sys.exit()

    # Determine final column widths
    col_widths = [0 for i in range(len(MOF_PSEUDO_ATOMS[0]))]
    for i in range(len(MOF_PSEUDO_ATOMS[0])):
        # num columns
        max_width = 0
        for j in range(len(MOF_PSEUDO_ATOMS)):
            # num rows
            if(len(MOF_PSEUDO_ATOMS[j][i]) > max_width):
                max_width = len(MOF_PSEUDO_ATOMS[j][i])
        col_widths[i] = max_width + 2

    col_widths1 = [0 for i in range(len(GENERIC_PSEUDO_ATOMS[0]))]
    for i in range(len(GENERIC_PSEUDO_ATOMS[0])):
        # num columns
        max_width = 0
        for j in range(len(GENERIC_PSEUDO_ATOMS)):
            # num rows
            if(len(GENERIC_PSEUDO_ATOMS[j][i]) > max_width):
                max_width = len(GENERIC_PSEUDO_ATOMS[j][i])
        col_widths1[i] = max_width + 2

    col_widths_final = [max(col_widths[i], col_widths1[i]) for i in range(len(col_widths))]


    # Begin file writing
    f = open('pseudo_atoms.def','w')

    # write header 
    num_psuedo_atoms =len(GENERIC_PSEUDO_ATOMS) + len(MOF_PSEUDO_ATOMS)
    GENERIC_PSEUDO_ATOMS_HEADER[1] = str(num_psuedo_atoms)
    for line in GENERIC_PSEUDO_ATOMS_HEADER:
        f.write("".join(word for word in line) +'\n')

    # write this MOFs pseudo atoms
    for i in range(len(MOF_PSEUDO_ATOMS)):
        base_string = ""
        for j in range(len(MOF_PSEUDO_ATOMS[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(MOF_PSEUDO_ATOMS[i][j])))

            base_string += MOF_PSEUDO_ATOMS[i][j]
            base_string += buff
        f.write(base_string + '\n')

    # write the generic adsorbates
    for i in range(len(GENERIC_PSEUDO_ATOMS)):
        base_string = ""
        for j in range(len(GENERIC_PSEUDO_ATOMS[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(GENERIC_PSEUDO_ATOMS[i][j])))

            base_string += GENERIC_PSEUDO_ATOMS[i][j]
            base_string += buff
        f.write(base_string + '\n')

    f.close()


    # Determine column widths for FF MIXING
    col_widths = [0 for i in range(len(MOF_FF_MIXING[0]))]
    for i in range(len(MOF_FF_MIXING[0])):
        # num columns
        max_width = 0
        for j in range(len(MOF_FF_MIXING)):
            # num rows
            if(len(MOF_FF_MIXING[j][i]) > max_width):
                max_width = len(MOF_FF_MIXING[j][i])
        col_widths[i] = max_width + 2

    col_widths1 = [0 for i in range(len(GENERIC_FF_MIXING[0]))]
    for i in range(len(GENERIC_FF_MIXING[0])):
        # num columns
        max_width = 0
        for j in range(len(GENERIC_FF_MIXING)):
            # num rows
            if(len(GENERIC_FF_MIXING[j][i]) > max_width):
                max_width = len(GENERIC_FF_MIXING[j][i])
        col_widths1[i] = max_width + 2

    col_widths_final = [max(col_widths[i], col_widths1[i]) for i in range(len(col_widths))]

    # write ff mixing file
    f = open('force_field_mixing_rules.def','w')

    # write header
    num_interactions = len(MOF_FF_MIXING) + len(GENERIC_FF_MIXING)
    GENERIC_FF_MIXING_HEADER[5] = str(num_interactions)
    for line in GENERIC_FF_MIXING_HEADER:
        f.write("".join(word for word in line) +'\n')


    # write this MOFs pseudo atoms
    for i in range(len(MOF_FF_MIXING)):
        base_string = ""
        for j in range(len(MOF_FF_MIXING[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(MOF_FF_MIXING[i][j])))

            base_string += MOF_FF_MIXING[i][j]
            base_string += buff
        f.write(base_string + '\n')

    # write the generic adsorbates
    for i in range(len(GENERIC_FF_MIXING)):
        base_string = ""
        for j in range(len(GENERIC_FF_MIXING[0])):
            buff = "".join(" " for i in range(col_widths_final[j] - len(GENERIC_FF_MIXING[i][j])))

            base_string += GENERIC_FF_MIXING[i][j]
            base_string += buff
        f.write(base_string + '\n')

    for line in GENERIC_FF_MIXING_FOOTER:
        f.write("".join(word for word in line) +'\n')

    f.close()


class MDMC_config(object):
    """
    Very sloppy for now but just doing the bare minimum to get this up and running
    for methane in flexible materials
    """

    def __init__(self, lammps_sim):
        
        try:
            f = open("MDMC.config","r")
        except:
            self.initialized = False
            print("Warning! No MDMC.config file found.  LAMMPS sim files will not have guest molecule info")
            return

        lines = f.readlines()

        outlines = ""
        for line in lines:
            parsed = line.strip().split()

            if(parsed[0] == "num_framework"):
                outlines += "num_framework\t%d\n"%(nx.number_of_nodes(lammps_sim.graph))
            if(parsed[0] == "type_framework"):
                outlines += "type_framework\t%d\n"%(len(lammps_sim.unique_atom_types.keys()))
            if(parsed[0] == "type_guest"):
                self.type_guest = int(parsed[1])
                outlines += "type_guest\t%d\n"%(self.type_guest)
            if(parsed[0] == "pair_coeff"):
                parsed[1] = str(len(lammps_sim.unique_atom_types.keys()) + self.type_guest)
                for word in parsed:
                    outlines += word + " "
                outlines += "\n"
            if(parsed[0] == "mass_guest"):
                parsed[1] = str(len(lammps_sim.unique_atom_types.keys()) + self.type_guest)
                for word in parsed:
                    outlines += word + " "
                outlines += "\n"

        f.close()
        
        f = open("MDMC.config", "w")
        f.write(outlines)
        f.close()

        return


class Cell(object):
    def __init__(self):
        self._cell = np.identity(3, dtype=np.float64)
        # cell parameters (a, b, c, alpha, beta, gamma)
        self._params = (1., 1., 1., 90., 90., 90.)
        self._inverse = None

    @property
    def volume(self):
        """Calculate cell volume a.bxc."""
        b_cross_c = cross(self.cell[1], self.cell[2])
        return dot(self.cell[0], b_cross_c)

    def get_cell(self):
        """Get the 3x3 vector cell representation."""
        return self._cell

    def get_cell_inverse(self):
        """Get the 3x3 vector cell representation."""
        return self._inverse

    def mod_to_UC(self, num):
        """
        Retrun any fractional coordinate back into the unit cell
        """
        if(hasattr(num,'__iter__')):
            for i in range(len(num)):
                if(num[i] < 0.0):
                    num[i] = 1+math.fmod(num[i], 1.0)
                else:
                    num[i] = math.fmod(num[i], 1.0)

            return num
        else:
            if(num < 0.0):
                num = math.fmod((num*(-1)), 1.0)
            else:
                num = math.fmod(num, 1.0)

    def set_cell(self, value):
        """Set cell and params from the cell representation."""
        # Class internally expects an array
        self._cell = np.array(value).reshape((3,3))
        self.__mkparam()
        self.__mklammps()
        # remake cell so a in x, b in xy and c in xyz
        self.__mkcell()
        self._inverse = np.linalg.inv(self.cell.T)

    # Property so that params are updated when cell is set
    cell = property(get_cell, set_cell)

    def get_params(self):
        """Get the six parameter cell representation as a tuple."""
        return tuple(self._params)

    def set_params(self, value):
        """Set cell and params from the cell parameters."""
        self._params = value
        self.__mkcell()
        self.__mklammps()
        self._inverse = np.linalg.inv(self.cell.T)

    params = property(get_params, set_params)

    def minimum_supercell(self, cutoff):
        """Calculate the smallest supercell with a half-cell width cutoff.
        
        Increment from smallest cell vector to largest. So the supercell
        is not considering the 'unit cell' for each cell dimension.

        """
        a_cross_b = np.cross(self.cell[0], self.cell[1])
        b_cross_c = np.cross(self.cell[1], self.cell[2])
        c_cross_a = np.cross(self.cell[2], self.cell[0])

        #volume = np.dot(self.cell[0], b_cross_c)

        widths = [np.dot(self.cell[0], b_cross_c) / np.linalg.norm(b_cross_c),
                  np.dot(self.cell[1], c_cross_a) / np.linalg.norm(c_cross_a),
                  np.dot(self.cell[2], a_cross_b) / np.linalg.norm(a_cross_b)]

        return tuple(int(math.ceil(2*cutoff/x)) for x in widths)

    def orthogonal_transformation(self):
        """Compute the transformation from the original unit cell to a supercell which
        has 90 degree angles between it's basis vectors. This is somewhat approximate,
        and the angles will not be EXACTLY 90 deg.

        """
        zero_itol = 0.002 # tolerance for zero in the inverse matrix 
        M = self._inverse.T.copy()
        M[np.where(np.abs(M) < zero_itol)] = 0.
        absmat = np.abs(M)
        # round all near - zero values to zero
        absmat[np.where(absmat < zero_itol)] = 0.
        divs = np.array([np.min(absmat[i, np.nonzero(absmat[i])]) for i in range(3)])

        # This is a way to round very small entries in the inverse matrix, so that
        # supercells are not unwieldly
        MN = np.around(M / divs[:,None])
        return MN

    def update_supercell(self, tuple):
        self._cell = np.multiply(self._cell.T, tuple).T
        self.__mkparam()
        self.__mklammps()
        self._inverse = np.linalg.inv(self._cell.T)

    @property
    def minimum_width(self):
        """The shortest perpendicular distance within the cell."""
        a_cross_b = cross(self.cell[0], self.cell[1])
        b_cross_c = cross(self.cell[1], self.cell[2])
        c_cross_a = cross(self.cell[2], self.cell[0])

        volume = dot(self.cell[0], b_cross_c)

        return volume / min(np.linalg.norm(b_cross_c), np.linalg.norm(c_cross_a), np.linalg.norm(a_cross_b))

    @property
    def inverse(self):
        """Inverted cell matrix for converting to fractional coordinates."""
        try:
            if self._inverse is None:
                self._inverse = np.linalg.inv(self.cell.T)
        except AttributeError:
            self._inverse = np.linalg.inv(self.cell.T)
        return self._inverse

    @property
    def crystal_system(self):
        """Return the IUCr designation for the crystal system."""
        #FIXME(tdaff): must be aligned with x to work
        if self.alpha == self.beta == self.gamma == 90:
            if self.a == self.b == self.c:
                return 'cubic'
            elif self.a == self.b or self.a == self.c or self.b == self.c:
                return 'tetragonal'
            else:
                return 'orthorhombic'
        elif self.alpha == self.beta == 90:
            if self.a == self.b and self.gamma == 120:
                return 'hexagonal'
            else:
                return 'monoclinic'
        elif self.alpha == self.gamma == 90:
            if self.a == self.c and self.beta == 120:
                return 'hexagonal'
            else:
                return 'monoclinic'
        elif self.beta == self.gamma == 90:
            if self.b == self.c and self.alpha == 120:
                return 'hexagonal'
            else:
                return 'monoclinic'
        elif self.a == self.b == self.c and self.alpha == self.beta == self.gamma:
            return 'trigonal'
        else:
            return 'triclinic'

    def __mkcell(self):
        """Update the cell representation to match the parameters."""
        a_mag, b_mag, c_mag = self.params[:3]
        alpha, beta, gamma = [x * DEG2RAD for x in self.params[3:]]
        a_vec = np.array([a_mag, 0.0, 0.0])
        b_vec = np.array([b_mag * np.cos(gamma), b_mag * np.sin(gamma), 0.0])
        c_x = c_mag * np.cos(beta)
        c_y = c_mag * (np.cos(alpha) - np.cos(gamma) * np.cos(beta)) / np.sin(gamma)
        c_vec = np.array([c_x, c_y, (c_mag**2 - c_x**2 - c_y**2)**0.5])
        self._cell = np.array([a_vec, b_vec, c_vec])

    def __mkparam(self):
        """Update the parameters to match the cell."""
        cell_a = np.sqrt(sum(x**2 for x in self.cell[0]))
        cell_b = np.sqrt(sum(x**2 for x in self.cell[1]))
        cell_c = np.sqrt(sum(x**2 for x in self.cell[2]))
        alpha = np.arccos(sum(self.cell[1, :] * self.cell[2, :]) /
                       (cell_b * cell_c)) * 180 / np.pi
        beta = np.arccos(sum(self.cell[0, :] * self.cell[2, :]) /
                      (cell_a * cell_c)) * 180 / np.pi
        gamma = np.arccos(sum(self.cell[0, :] * self.cell[1, :]) /
                       (cell_a * cell_b)) * 180 / np.pi
        self._params = (cell_a, cell_b, cell_c, alpha, beta, gamma)

    def __mklammps(self):
        a, b, c, alpha, beta, gamma = self._params
        lx = a
        xy = b*math.cos(gamma*DEG2RAD)
        xz = c*math.cos(beta*DEG2RAD)
        ly = math.sqrt(b**2 - xy**2)
        yz = (b*c*math.cos(alpha*DEG2RAD) - xy*xz)/ly
        lz = math.sqrt(c**2 - xz**2 - yz**2)
        self._lammps = (lx, ly, lz, xy, xz, yz)

    @property
    def lx(self):
        return self._lammps[0]
    @property
    def ly(self):
        return self._lammps[1]
    @property
    def lz(self):
        return self._lammps[2]
    @property
    def xy(self):
        return self._lammps[3]
    @property
    def xz(self):
        return self._lammps[4]
    @property
    def yz(self):
        return self._lammps[5]

    @property
    def a(self):
        """Magnitude of cell a vector."""
        return self.params[0]

    @property
    def b(self):
        """Magnitude of cell b vector."""
        return self.params[1]

    @property
    def c(self):
        """Magnitude of cell c vector."""
        return self.params[2]

    @property
    def alpha(self):
        """Cell angle alpha."""
        return self.params[3]

    @property
    def beta(self):
        """Cell angle beta."""
        return self.params[4]

    @property
    def gamma(self):
        """Cell angle gamma."""
        return self.params[5]

def clean(name):
    name = os.path.split(name)[-1]
    if name.endswith('.cif'):
        name = name[:-4]
    return name
