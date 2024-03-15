#!/usr/bin/env python
from __future__ import print_function
from openmm.app import *
from openmm import *
from openmmtorch import TorchForce
from simtk.unit import *
from sys import stdout
from time import gmtime, strftime
from datetime import datetime
import sys, os
from nn_classes import NNAtoms
from schnet_force import SchNetForce, SchNetForcePBC, SchNet, Atomwise
import torch
import schnetpack as sch
import time 

nn_resids = [0]
#Previously trained model
prev_model = "best_model"

temperature = 300.0*kelvin
pressure = 1.0*atmosphere
cutoff = 1.0*nanometer

barofreq = 100
Topology.loadBondDefinitions("pb_residues.xml")
strdir = 'nn_test/'
if not os.path.isdir(strdir): os.mkdir(strdir)
pdb = PDBFile("nhc_aci_220.pdb")

integ_md = DrudeLangevinIntegrator(temperature, 1/picosecond, 1*kelvin, 1/picosecond, 0.001*picoseconds)
integ_md.setMaxDrudeDistance(0.02)

modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField('pb.xml')
modeller.addExtraParticles(forcefield)

system = forcefield.createSystem(modeller.topology, nonbondedCutoff=cutoff, constraints=None, rigidWater=True)
nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == NonbondedForce][0]
customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == CustomNonbondedForce][0]
if pdb.topology.getPeriodicBoxVectors():
    nbondedForce.setNonbondedMethod(NonbondedForce.PME)
drudeForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if type(f) == DrudeForce][0]
customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(),NonbondedForce.CutoffPeriodic))
print('nbMethod : ', customNonbondedForce.getNonbondedMethod())

for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)

#barostat = MonteCarloBarostat(pressure, temperature, barofreq)
#system.addForce(barostat)

totmass = 0.*dalton
for i in range(system.getNumParticles()):
    totmass += system.getParticleMass(i)

platform = Platform.getPlatformByName('CUDA')
#platform = Platform.getPlatformByName('CPU')
#platform = Platform.getPlatformByName('OpenCL')
#properties = {'OpenCLPrecision': 'mixed','OpenCLDeviceIndex':'0'}
properties = {'CudaPrecision': 'mixed'}

simmd = Simulation(modeller.topology, system, integ_md, platform, properties)
#simmd = Simulation(modeller.topology, system, integ_md, platform)
simmd.context.setPositions(modeller.positions)

platform = simmd.context.getPlatform()
platformname = platform.getName();
print(platformname)

#NNAtoms class collects indices of atoms that the neural network will be applied to
nn_atoms = NNAtoms(simmd, system, nn_resids)

#Exclude bonded and intra nonbonded interactions for the neural network residues
nn_atoms.generateNNExclusions()

#Defines the class used to generate the representation for the neural network
schnet = SchNet(
    n_atom_basis=128,
    n_filters=128,
    n_gaussians=30,
    n_interactions=3,
    cutoff=8., cutoff_network=sch.nn.cutoff.CosineCutoff
)

#Since a new neural network class is being initialized, we have to set the parameters
#to those of a previously trained neural network
best_model = torch.load(prev_model)

#Cutoff used in previous neural network
cutoff = best_model.representation.interactions[0].cutoff_network.cutoff

#Collects output from neural network 
output = Atomwise(n_in=128, property='energy')

#Since the function call for a TorchForce that uses PBCs is different than from one that doesn't use
#PBCs, there are two separate SchNetForce classes
schnet_force = SchNetForcePBC(nn_atoms.atomicNumbers, nn_atoms.realNNAtomIndex, schnet, output, cutoff)
#schnet_force = SchNetForce(nn_atoms.atomicNumbers, nn_atoms.realNNAtomIndex, schnet, output)

#Set the parameters of the newly defined SchNetForce to those of a previous model
best_model_state_dict = best_model.state_dict()
schnet_force_state_dict = schnet_force.state_dict()
for key in schnet_force_state_dict.keys():
    if key in best_model_state_dict.keys():
        schnet_force_state_dict[key] = best_model_state_dict[key]
schnet_force.load_state_dict(schnet_force_state_dict)

#Create TorchScript and save model
schnet_force = torch.jit.script(schnet_force)
schnet_force.save('model.pt')

# Create the TorchForce from the serialized compute graph
torch_force = TorchForce('model.pt')
torch_force.setUsesPeriodicBoundaryConditions(True)
system.addForce(torch_force)

# now reinitialize to make sure changes are stored in context
state = simmd.context.getState(getEnergy=False,getForces=False,getVelocities=False,getPositions=True)
positions = state.getPositions()
simmd.context.reinitialize()
simmd.context.setPositions(positions)

for i in range(system.getNumForces()):
    f = system.getForce(i)
    type(f)
    f.setForceGroup(i)

state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
position = state.getPositions()
simmd.context.setPositions(position)
print(str(state.getKineticEnergy()))
print(str(state.getPotentialEnergy()))
for i in range(system.getNumForces()):
    f = system.getForce(i)
    print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**i).getPotentialEnergy()))

# write initial pdb file with drude oscillators
PDBFile.writeFile(simmd.topology, position, open(strdir+'start_drudes.pdb', 'w'))
simmd.reporters = []
simmd.reporters.append(DCDReporter(strdir+f'md_nvt.dcd', 1000))
simmd.reporters.append(CheckpointReporter(strdir+f'md_nvt.chk', 1000))
simmd.reporters[1].report(simmd,state)

print('Starting NVT Simulation...')
start = time.time()
for i in range(10):
    simmd.step(1000)
    print(i,strftime("%Y-%m-%d %H:%M:%S", gmtime()))
    print(i,datetime.now())
    state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
    print(str(state.getKineticEnergy()))
    print(str(state.getPotentialEnergy()))
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        print(type(f), str(simmd.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()))
print(time.time() - start)
print('Finished NVT Simulation')

# print equilibrated pdb file
state = simmd.context.getState(getEnergy=True,getForces=True,getPositions=True)
position = state.getPositions()
simmd.topology.setPeriodicBoxVectors(state.getPeriodicBoxVectors())
PDBFile.writeFile(simmd.topology, position, open(strdir+f'equilibrated.pdb', 'w'))

exit()
