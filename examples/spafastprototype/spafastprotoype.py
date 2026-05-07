import sunrise as sn
from numpy import eye,pi
import tequila as tq
import networkx as nx
from copy import deepcopy

#Step 1: Molecule intialization
n_atoms = 8
geo = "".join([f'H 0. 0. {i}\n'.format(i) for i in range(n_atoms)])
mol = tq.Molecule(geometry=geo,basis_set='sto-3g',backend='pyscf',units='a').use_native_orbitals()

# The first graph correspond to: H-H H-H H-H ...
edges = [(2*i,2*i+1) for i in range(mol.n_orbitals//2)]

#Step 2: The SPA fast prototype already contains an orbital optimizer which takes
# advantage of this decomposition to compute the 1-RDM and 2-RDM more efficiently:
# Each RDM element is expanded in Pauli strings: <O>_U = sum_k c_k <P_k>_U
# Since the circuit factorizes over clusters (defined by `grouping`, see below), 
# the expectation values also factorize:  <P_k>_U = <P_k,0>_U0 · <P_k,1>_U1 · <P_k,2>_U2 · ...
# Thus only local expectation values are computed and combined, instead of simulating the full system.

initial_guess = eye(mol.n_orbitals) # See explanation DOI: 10.22331/q-2023-08-03-1073
for edge in edges:
    initial_guess[edge[0],edge[1]] = 1
    initial_guess[edge[1],edge[0]] = -1

# For further molecules, we recomend the combination with sn.CLPO.generate_CLPO_molecule_edges
# see Sunrise/examples/CLPO_orbitals for an example

grouping = None # Grouping keyword maybe None,number, or list of numbers
#If none, the circuit conectivity is employed. 
#If Number, the circuit conectivity is grouped in blocks of your number +-1
#If list of numbers, you decide it manually

opt = sn.SPAFP.run_spa(mol,edges,initial_guess=initial_guess.T,grouping=grouping)
mol = opt.molecule
sn.plot_MO(molecule=mol,filename='SPA-first_graph')

# Step 3: You may extend this circuit and still taking advantage of this decomposition as:

# Here we will add the graph H* H-H *H H* H-H *H even if it not the more chemically relevant but to show the decomposition 
U = mol.make_ansatz("SPA",edges=edges) 
a0 = tq.Variable('UR(0,1)')
a1 = tq.Variable('UR(4,5)')
b0 = tq.Variable('UR(2,3)')
b1 = tq.Variable('UR(6,7)')
c0 = tq.Variable('UR(1,2)')
c1 = tq.Variable('UR(5,6)')
d0 = tq.Variable('UR(0,3)')
d1 = tq.Variable('UR(4,7)')
e = tq.Variable('UC(1,2)')
f = tq.Variable('UC(6,7)')
# We want to start the Orbital Rotators from pi/2
UR = mol.UR(0,1,(a0+0.5)*pi) + mol.UR(2,3,(b0+0.5)*pi) + mol.UR(4,5,(a0+0.5)*pi) + mol.UR(6,7,(b0+0.5)*pi)
UR1 = mol.UR(1,2,(c0+0.5)*pi) + mol.UR(0,3,(d0+0.5)*pi) + mol.UR(5,6,(c1+0.5)*pi) + mol.UR(4,7,(d1+0.5)*pi)
UC = mol.UC(1,2,e) + mol.UC(6,7,f)

Uspaplus = U + UR + UR1 + UC + UR1.dagger() + UR.dagger()
# Now qubits 0-7 and 8-15 are entangled, they can't decompose

# If you want to check the circuit separability:
print('Circuit Decompostion: ',list(nx.connected_components(Uspaplus.to_networkx())))
E = sn.SPAFP.decompose(H=mol.make_hamiltonian(),U=Uspaplus,grouping=[[*range(0,8)],[*range(8,16)]])
res = sn.minimize(E,silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-5})
print('Extended SPA Energy: ',res.energy)

# As already shown on the givens rotation example, we can plot the intermediate "graph two" orbitals
r1mol = deepcopy(mol)
mUR1 = deepcopy(UR+UR1)
mUR1 = mUR1.map_variables({d:res.variables[d] for d in mUR1.extract_variables()})
r1mol = r1mol.transform_orbitals(mUR1)
sn.plot_MO(r1mol,filename='SPA-second_graph')
