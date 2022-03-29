from openmm.app import *
from openmm import *
from openmm.unit import *
from simtk import openmm, unit
from openmmtools import testsystems

water_box = testsystems.WaterBox(box_edge=20.0*unit.nanometer)
positions = water_box.positions
atoms = list(water_box.topology.atoms())
elements = [a.element.symbol for a in atoms]
PDBFile.writeFile(water_box.topology, positions, open('water-20.pdb', 'w'))
