import tequila as tq
import pytest
import sunrise as sn
from numpy import isclose

@pytest.mark.parametrize("geom",["H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2","O 0.00000 0.00000 0.11779\nH 0.00000 0.75545 -0.47116\nH 0.00000 -0.75545 -0.47116"])
@pytest.mark.parametrize('nature',['tequila','hybrid'])
def test_fast_spa_opt(geom, nature):
    tqmol = tq.Molecule(geometry=geom, basis_set='sto-3g', units='a')
    snmol = tq.Molecule(geometry=geom, basis_set='sto-3g', units='a', nature=nature)

    tqmol,tqedges = sn.CLPO.generate_CLPO_molecule_edges(mol=tqmol)
    snmol,snedges = sn.CLPO.generate_CLPO_molecule_edges(mol=snmol)

    tqU = tqmol.make_ansatz('SPA',edges=tqedges) + tqmol.UC(1,2,"a")
    snU = snmol.make_ansatz('SPA',edges=tqedges) + snmol.UC(1,2,"a")

    tqres = tq.minimize(tq.ExpectationValue(H=tqmol.make_hamiltonian(),U=tqU),silent=True)
    snres = tq.minimize(sn.SPAFP.decompose(H=tqmol.make_hamiltonian(),U=tqU,grouping=4),silent=True, gradient="2-point",method_options={"finite_diff_rel_step":1.e-4})

    assert isclose(tqres.energy, snres.energy)

@pytest.mark.parametrize("geom",["H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2","O 0.00000 0.00000 0.11779\nH 0.00000 0.75545 -0.47116\nH 0.00000 -0.75545 -0.47116"])
@pytest.mark.parametrize('nature',['tequila','hybrid'])
def test_fast_spa_opt(geom, nature):
    tqmol = tq.Molecule(geometry=geom, basis_set='sto-3g', units='a')
    snmol = tq.Molecule(geometry=geom, basis_set='sto-3g', units='a', nature=nature)

    tqmol,tqedges = sn.CLPO.generate_CLPO_molecule_edges(mol=tqmol)
    snmol,snedges = sn.CLPO.generate_CLPO_molecule_edges(mol=snmol)

    tqopt = tq.chemistry.optimize_orbitals(molecule=tqmol, circuit=tqmol.make_ansatz("HCB-SPA",edges=tqedges), silent=True, use_hcb=True)
    snopt = sn.SPAFP.run_spa(mol=snmol, edges=snedges, silent=True, initial_guess=False, grouping=4)

    assert isclose(tqopt.energy, snopt.energy)


