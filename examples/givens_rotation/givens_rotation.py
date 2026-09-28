import sunrise as sn
import tequila as tq
from numpy import pi
from copy import deepcopy

geometry = 'H 0. 0. 0. \n H 0. 0. 1. \n H 0. 0. 2.\n H 0. 0. 3.'
mol = sn.Molecule(geometry=geometry, basis_set='sto-3g',nature='f',backend='pyscf').use_native_orbitals()
ref = mol.compute_energy('fci')
mol, edges = mol.use_CLPO_orbitals_and_edges() # Since H4 at equal separated distance, edges will be [(0,1),(2,3)]
U = sn.FCircuit.from_edges(n_orb=mol.n_orbitals, edges=edges)

# First we optimize the CLPO orbitals
opt = sn.optimize_orbitals(circuit=U, molecule=mol,silent=True,backend='tcc')
sn.plot_MO(opt.molecule,filename='first_graph_orbitals')
omol = opt.molecule
a = tq.Variable('UR(0,1)')
b = tq.Variable('UR(2,3)')
c = tq.Variable('UR(1,2)')
d = tq.Variable('UR(0,3)')
e = tq.Variable('UC(1,2)')
f = tq.Variable('UC(0,3)')
# We want to start the Orbital Rotators from pi/2
UR = omol.UR(0,1,(a+0.5)*pi) + omol.UR(2,3,(b+0.5)*pi)
UR1 = omol.UR(1,2,(c+0.5)*pi) + omol.UR(0,3,(d+0.5)*pi)
UC = omol.UC(1,2,e) + omol.UC(0,3,f)
# SPA+ = First graph + basis change back to native base + basis change to 2nd graph basis + rotation back to the original first graph basis
SPAplus:sn.FCircuit = U + UR + UR1 + UC + UR1.dagger() + UR.dagger()
BK = sn.Braket(backend='tcc',circuit=SPAplus, molecule=omol)
E_spa = sn.minimize(BK, silent=True)
print("SPA Error H4:", (opt.energy-ref)*1000)
print('SPA+ Error H4:', (E_spa.energy-ref)*1000)

# Given that one can do orbital rotation matrix <=> UR + Phase gates, we can use the UR minimized variables to show on which basis we are at each point
## First UR layer
r0mol = deepcopy(omol)
mUR = deepcopy(UR)
mUR = mUR.map_variables({d:E_spa.variables[d] for d in mUR.extract_variables()})
r0mol = r0mol.transform_orbitals(mUR)
sn.plot_MO(r0mol,filename='back_to_native')
## First + Second UR layer
r1mol = deepcopy(omol)
mUR1 = deepcopy(UR+UR1)
mUR1 = mUR1.map_variables({d:E_spa.variables[d] for d in mUR1.extract_variables()})
r1mol = r1mol.transform_orbitals(mUR1)
sn.plot_MO(r1mol,filename='second_graph_orbitals')
