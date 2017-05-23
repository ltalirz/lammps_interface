#!/usr/bin/env python

from argparse import ArgumentParser
import os
import subprocess
def git_revision_hash():
    src_dir = os.path.dirname(os.path.abspath(__file__))
    wrk_dir = os.getcwd()
    os.chdir(src_dir)
    rev_no = len(subprocess.check_output(['git', 'rev-list', 'HEAD'], universal_newlines=True).strip().split("\n")) 
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], universal_newlines=True).strip()
    os.chdir(wrk_dir)
    return (rev_no, commit)
rev_no, commit = git_revision_hash()
__version_info__ = (0, 0, rev_no, "%s"%commit)
__version__ = "%i.%i.%i.%s"%__version_info__

class Options(object):

    def __init__(self):
        #print("Lammps_interface version: %s"%__version__)
        self.run_command_line_options()

    def run_command_line_options(self):
        parser = ArgumentParser(description="LAMMPS interface :D", prog="lammps_interface")
        parser.add_argument("-V", "--version",
                            action="version",
                            version="%(prog)s version "+__version__)
        parser.add_argument("-o", "--outputcif",
                            action="store_true",
                            dest="output_cif",
                            help="Write a .cif file for visualization."+
                                 " Necessary for debugging purposes, this"+
                                 " file can show the user how the structure "+
                                 "has been interpreted by the program.")

        parser.add_argument("-or", "--outputraspa",
                            action="store_true",
                            dest="output_raspa",
                            help="Write a .cif file for RASPA (FF types in _atom_site_label)"+
                                 " Write pseudo_atoms.def file for this MOF"+
                                 " Write force_field_mixing_rules.def file for this MOF"+
                                 " Write force_field.def file for this MOF")

        #split the command line options into separate groups for nicer
        #visualization.
        force_field_group = parser.add_argument_group("Force Field options")
        force_field_group.add_argument("-ff", "--force_field", action="store", 
                                       type=str, dest="force_field",
                                       default="UFF",
                                       help="Enter the requested force "+
                                          "field to describe the system. Current "+
                                          "options are 'BTW_FF', 'Dreiding', 'UFF', "+
                                          "'UFF4MOF', and 'Dubbeldam'."+
                                          " The default is set to the Universal "+
                                          "Force Field [UFF].")
        force_field_group.add_argument("--molecule-ff", action="store", 
                                       dest="mol_ff",
                                       default=None,
                                       help="Chose a force field for any molecules "+
                                          "found in the structure. This is applies a 'blanket' "+
                                          "to all found molecules, so exercise with caution. Future "+
                                          "iterations will consider an input file to differentiate "+
                                          "force fields between different molecules. Default is the "+
                                          "same force field requested for the framework (assumes some "+
                                          "generalized FF like UFF or Dreiding).")
        force_field_group.add_argument("--h-bonding", action="store_true", 
                                       dest="h_bonding",
                                       default=False,
                                       help="Add hydrogen bonding potentials "+
                                          "to the force field characterization."+
                                          " Currently only applies to Dreiding. "+
                                          "Default is off.")
        force_field_group.add_argument("--dreid-bond-type", action="store", 
                                       dest="dreid_bond_type",
                                       type=str,
                                       default="harmonic",
                                       help="Request the type of bonding potential "+
                                          "for the Dreiding force field. Options are 'morse' or "+
                                          "'harmonic'. Default is harmonic.")
        force_field_group.add_argument("--fix-metal", action="store_true", 
                                       dest="fix_metal",
                                       default=False,
                                       help="Fix the metal geometries with "+
                                          "modified potentials to match their "+
                                          "input geometries. The potential isn't set "+ 
                                          "to be overly rigid so that the material "+
                                          "will behave physically in finite temperature "+
                                          "calculations, however it may introduce some "+
                                          "unintended artifacts so exercise with caution. "+
                                          "Useful for structure minimizations. Currently only "+
                                          "applies to UFF and Dreiding Force Fields. Default is "+
                                          "off.")
        force_field_group.add_argument("--eps-scale", action="store", 
                                       dest="eps_scale_factor",
                                       type=float,
                                       default=1.0,
                                       help="Scale the epsilon Lennard-Jones "+
                                          "parameters by a constant factor. This "+
                                          "can be used to generate the FlexZIF force field, for "+
                                          "example, by setting UFF epsilon to 0.635. Default is 1.0 (standard epsilon).") 
        simulation_group = parser.add_argument_group("Simulation options")
        simulation_group.add_argument("--minimize", action="store_true",
                                      dest="minimize",
                                      default=False,
                                      help="Request input files necessary for"
                                      + " a geometry optimization. Default off")
        simulation_group.add_argument("--bulk-moduli", action="store_true",
                                      dest="bulk_moduli",
                                      default=False,
                                      help="Request input files necessary for"
                                      + " an energy vs volume calculation. This will use "+
                                      "values from ITER_COUNT and MAX_DEV to create "+
                                      "the volume range")
        simulation_group.add_argument("--thermal-scaling", action="store_true",
                                      dest="thermal_scaling",
                                      default=False,
                                      help="Request input files necessary for"
                                      + " a temperature scaling calculation. This will use "+
                                      "values from ITER_COUNT and MAX_DEV to create "+
                                      "the temperature range")
        simulation_group.add_argument("--npt", action="store_true",
                                      dest="npt",
                                      default=False,
                                      help="Request input files necessary for"
                                      + " an isothermal-isobaric simulation. This will use "+
                                      "values from TEMP and PRESSURE, NEQSTP, and NPRODSTP to "+
                                      "produce the input file.")
        simulation_group.add_argument("--gcmc", action="store_true",
                                      dest="gcmc",
                                      default=False,
                                      help="Request a gcmc simulation for a molecule defined by "
                                      + "INSERT_MOLECULE. This will use "+
                                      "values from TEMP and PRESSURE, NEQSTP, and NPRODSTP to "+
                                      "produce the input file. Default is 'off'")
        simulation_group.add_argument("--gcmc-every", action="store",
                                      dest="gcmc_every",
                                      default=10,
                                      type=int,
                                      help="Invoke the GCMC fix every this many MD timesteps. "+
                                      "Default is 10 (ignored if GCMC=False).")
        simulation_group.add_argument("--gcmc-exch", action="store",
                                      dest="gcmc_exch",
                                      default=100,
                                      type=int,
                                      help="Perform this many insertion/deletions on average "+
                                      "each time the GCMC fix is invoked (which is controlled by "+
                                      "the GCMC_EVERY variable). Default is 100.")
        simulation_group.add_argument("--gcmc-mc", action="store",
                                      dest="gcmc_mc",
                                      default=100,
                                      type=int,
                                      help="Perform this many Monte Carlo moves on average "+
                                      "each time the GCMC fix is invoked (which is controlled by "+
                                      "the GCMC_EVERY variable). Default is 100.")
        simulation_group.add_argument("--gcmc-disp", action="store",
                                      dest="gcmc_disp",
                                      default=1.0,
                                      type=float,
                                      help="Maximum distance (in Angstroms) a particle will move "+
                                      "during a Monte Carlo displacement. Default is 1.0 Angstrom. "+
                                      "Only relevant if the GCMC fix is invoked.") 

        simulation_group.add_argument("--pxrd", action="store_true",
                                      dest="pxrd",
                                      default=False,
                                      help="Request that a powder pattern be "
                                      + "computed for the simulation. Due to limitations "+
                                      "in the package in LAMMPS, only orthogonal "+
                                      "cells can be used. A warning will be printed "+
                                      "when this is requested, and the input file will "+
                                      "assume the simulation will proceed as orthogonal." + 
                                      " One can use the '-O' option to try and orthogonalize their "+
                                      "unit cell if desired. (may not work)")
        simulation_group.add_argument("--nvt", action="store_true",
                                      dest="nvt",
                                      default=False,
                                      help="Request input files necessary for"
                                      + " an canonical simulation. This will use "+
                                      "values from TEMP, NEQSTP, and NPRODSTP to "+
                                      "produce the input file. Equilibration with a "+
                                      "Langevin thermostat, Production with Nose-Hoover.")
        simulation_group.add_argument("--cutoff", action="store",
                                      type=float, dest="cutoff",
                                      default=12.5,
                                      help="Set the long-range cutoff "+
                                      "to this value in Angstroms." + 
                                      " This will determine the size of "+
                                      "the supercell computed for the simulation. "+
                                      "Default is 12.5 angstroms.")
        simulation_group.add_argument("--replication", action="store",
                                      type=str, dest="replication",
                                      default=None,
                                      help="Manually specify the replications to form the supercell "+
                                      "Use comma, space or 'x' delimited values for the a,b,c directions." + 
                                      " This is useful when dealing with flexible materials " +
                                      "where you know that structural collapse will result in " +
                                      "the box decreasing past 2*rcut")
        simulation_group.add_argument("-O","--orthogonalize", action="store_true",
                                      default=False,
                                      dest="orthogonalize",
                                      help="Makes a supercell of the simulation box with more-or-less "+
                                           "orthogonal supercell vectors. This is an approximation, but is "+
                                           "useful for certain calculations. Default is FALSE.")
        simulation_group.add_argument("--randomize-velocities",
                                      action="store_true",
                                      default=False,
                                      dest="random_vel",
                                      help="Adds a velocity randomization of the atoms "+
                                           "prior to finite temperature simulation. The velocities are "+
                                           "randomized to TEMP.")
        simulation_group.add_argument("--dcd",
                                      action="store",
                                      default=0,
                                      type=int,
                                      dest="dump_dcd",
                                      help="Store trajectory of simulation in a "+
                                           "dcd format every DUMP_DCD steps. Default is no trajectory "+
                                           "file will be written.")
        simulation_group.add_argument("--xyz",
                                      action="store",
                                      default=0,
                                      type=int,
                                      dest="dump_xyz",
                                      help="Store trajectory of simulation in a "+
                                           "xyz format every DUMP_XYZ steps. If not requested, then no trajectory "+
                                           "file will be written.")
        simulation_group.add_argument("--lammpstrj",
                                      action="store",
                                      default=0,
                                      type=int,
                                      dest="dump_lammpstrj",
                                      help="Store trajectory of simulation in a "+
                                           "lammpstrj format every DUMP_LAMMPSTRJ" +
                                           " steps. If not requested, then no trajectory "+
                                           "file will be written.")
        simulation_group.add_argument("--restart",
                                      action="store_true",
                                      default=False,
                                      dest="restart",
                                      help="Store last snapshot of trajectory of simulation in "+
                                           "lammps traj file format. index of last step RESTART = NEQSTP + NPRODSTP. "+
                                           "If NEQSTP and NPRODSTP are not specified, then RESTART=1")
        simulation_group.add_argument("--msd",
                                      action="store",
                                      default=0,
                                      type=int,
                                      dest="msd",
                                      help="Request a mean squared displacement for a requested MOLECULE "+
                                           "every MSD steps. NOTE: this requires the user to request a molecule on the command line! "+
                                           "Default is no msd file will be written.")

        parameter_group = parser.add_argument_group("Parameter options")
        parameter_group.add_argument("-t", "--tolerance",
                                     action="store",
                                     type=float,
                                     default=0.4,
                                     dest="tol",
                                     help="Tolerance in angstroms to determine "+
                                          "detection of inorganic clusters. " +
                                          "Default is 0.4 angstroms.")
        parameter_group.add_argument("--neighbour-size", "--neighbor-size",
                                     action="store",
                                     type=int,
                                     default=5,
                                     dest="neighbour_size",
                                     help="To find SBUs in the framework via "+
                                          "pattern recognition. This parameter "+
                                          "determines how large a subset of atoms "+
                                          "to search around each central atom in the framework. "+
                                          "Central atoms are typically considered the "+
                                          "metal species for inorganic SBUs or carbon/nitrogen "+
                                          "for organic SBUs. This parameter will collect all "+
                                          "atoms within NEIGHBOUR_SIZE bonds from the central atom. "+
                                          "Default is 5.")
        parameter_group.add_argument("--iter-count",
                                     action="store",
                                     type=int,
                                     default=10,
                                     dest="iter_count",
                                     help="Number of iteration steps to "+
                                          "change a variable of interest (temperature, volume). "+
                                          "Default is 10 steps.")
        parameter_group.add_argument("--max-deviation",
                                     action="store",
                                     type=float,
                                     default=0.01,
                                     dest="max_dev",
                                     help="Max deviation of adjusted variable "+
                                          "at each step is scaled by MAX_DEV/ITER_COUNT. "+
                                          "Default is 0.01 (ideal for volume).")
        parameter_group.add_argument("--temperature",
                                     action="store",
                                     type=float,
                                     default=298.0,
                                     dest="temp",
                                     help="Simulation temperature. This parameter is used "+
                                          "only if NPT or NVT are True. "+
                                          "Default is 298.0 Kelvin.")
        parameter_group.add_argument("--pressure",
                                     action="store",
                                     type=float,
                                     default=1.0,
                                     dest="pressure",
                                     help="Simulation pressure. This parameter is used "+
                                          "only if NPT is True. "+
                                          "Default is 1.0 atmosphere.")
        parameter_group.add_argument("--production-steps",
                                     action="store",
                                     type=int,
                                     default=200000,
                                     dest="nprodstp",
                                     help="Number of production steps in the simulation. "+
                                          "Applies to NPT and THERMAL_SCALING simulations. " +
                                          "Default is 200,000 steps. (corresponding to "+
                                          "200 ps if the timestep is 1 fs)")
        parameter_group.add_argument("--equilibration-steps",
                                     action="store",
                                     type=int,
                                     default=200000,
                                     dest="neqstp",
                                     help="Number of equilibration steps in the simulation. "+
                                          "Applies to NPT and THERMAL_SCALING simulations. " +
                                          "Default is 200,000 steps. (corresponding to "+
                                          "200 ps if the timestep is 1 fs)")

        # parameters for automated slab generation
        parameter_group.add_argument("--slab-face",
                                     action="store",
                                     type=str,
                                     default="1x0x0",
                                     dest="slab_face",
                                     help="The filler face of the slab you would like to generate"+
                                          "Must have the format ixjxk (default 1x0x0)")
        parameter_group.add_argument("--slab-target-thickness",
                                     action="store",
                                     type=float,
                                     default=30.0,
                                     dest="slab_target_thick",
                                     help="Target thickness for slab after min cut of the initial ASE slab"+
                                          "If 0.0, there is no attempt to have the final slab be of a certain thickness"+
                                          "Default is 30.0 ")
        parameter_group.add_argument("--slab-L",
                                     action="store",
                                     type=int,
                                     default=0,
                                     dest="slab_L",
                                     help="L (thickness) pararameter for naiive slab generaation by ASE"+
                                          "If > 0, we will use this value as the starting L for ASE"+
                                          "Successively larger ASE slabs will be built if min cut slab is not periodic"+
                                          "But we won't build successively larger ASE slabs based on length of the min cut slab"+  
                                          "If = 0, we will find the L that gives us an initial ASE slab closes to slab-target-thickness"+ 
                                          "Then we will grow the initial ASE slab until the min cut slab is periodic AND thicker than slab-target-thickness"+
                                          "Default is 0 ")
        parameter_group.add_argument("--slab-vacuum",
                                     action="store",
                                     type=float,
                                     default=12.5,
                                     dest="slab_vacuum",
                                     help="Approximate vacuum on either side of the slab"+
                                          "Final vacuum will be larger because the min cut slab"+
                                          "will always be smaller than the initial ASE slab"+
                                          "Default is 12.5")
        parameter_group.add_argument("--slab-verbose",
                                     action="store",
                                     type=bool,
                                     default=False,
                                     dest="slab_verbose",
                                     help="Output intermediate and failed min cut attempts during slab generation")

        molecule_insertion_group = parser.add_argument_group("Molecule insertion options")
        molecule_insertion_group.add_argument("--insert-molecule",
                                              action="store",
                                              type=str,
                                              default="",
                                              dest="insert_molecule",
                                              help="Prepeare an insertion of this molecule type."+
                                                   " Default is no molecule insertion. Current options are "+
                                                   "TIP5P_Water, TIP4P_Water, SPC_E, TIP3P, EPM2_CO2, N2_TraPPE."+
                                                   " More to come ;)")
        
        molecule_insertion_group.add_argument("--deposit",
                                              action="store",
                                              type=int,
                                              default=0,
                                              dest="deposit",
                                              help="Create commands to place DEPOSIT particles "+
                                                   "in the region of the unit cell. The particle "+
                                                   "types are provided by the INSERT_MOLECULE value."+
                                                   " This will also depend on the number of "+
                                                   "equilibrium steps NEQSTP requested, and "+
                                                   "currently only works with an NVT simulation."+
                                                   " NOTE: The number of particles to deposit "+
                                                   "is considered in the unit cell! The program "+
                                                   "will multipy this value by the number of unit "+
                                                   "cells that were needed to produce the simulation "+
                                                   "supercell.")

        parser.add_argument(metavar="CIF", dest="cif_file",
                            help="path to cif file to interpret")

        args = vars(parser.parse_args())
        self._set_attr(args)

    def _set_attr(self, args):
        for key, value in args.items():
            setattr(self, key, value)
