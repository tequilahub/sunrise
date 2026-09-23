from tequila import TequilaException
from sunrise.molecules import qubit_base
from sunrise.molecules.qubit_base import Molecule as QubitMolecule, MoleculeFromOpenFermion, MoleculeFromTequila
from sunrise.molecules.qubit_base.qc_base import QuantumChemistryBase
from .fermionic_base import FerMolecule
from .hybrid_base import HyMolecule

def Molecule(geometry: str = None,basis_set: str = None,nature: str = 'qubit',orbital_type: str = None,backend: str = None,guess_wfn=None,name: str = None,*args,**kwargs)->QuantumChemistryBase:
    """

    Parameters
    ----------
    geometry
        molecular geometry as string or as filename (needs to be in xyz format with .xyz ending)
    basis_set
        quantum chemistry basis set (sto-3g, cc-pvdz, etc)
    nature
        Molecule type (qubit, hybrid, fermionic); 'tequila' is kept as an alias for 'qubit'
    backend
        quantum chemistry backend (psi4, pyscf)
    guess_wfn
        pass down a psi4 guess wavefunction to start the scf cycle from
        can also be a filename leading to a stored wavefunction
    name
        name of the molecule, if not given it's auto-deduced from the geometry
        can also be done vice versa (i.e. geometry is then auto-deduced to name.xyz)
    args
    kwargs

    Returns
    -------
        Molecule object
    """
    if 'units' not in kwargs:
        kwargs['units'] = 'angstrom'
    if nature.lower() in ('qubit','q','tequila','t'):
        return QubitMolecule(geometry=geometry,basis_set=basis_set,orbital_type=orbital_type,backend=backend,guess_wfn=guess_wfn,name=name,*args,**kwargs)
    elif nature.lower()=='hybrid' or nature.lower()=='h':
        if 'select' in kwargs:
            select = kwargs['select']
            kwargs.pop('select')
        else: select = {}
        return HyMolecule(geometry=geometry,select=select,basis_set=basis_set,orbital_type=orbital_type,backend=backend,guess_wfn=guess_wfn,name=name,*args,**kwargs)
    elif nature.lower()=='fermionic' or nature.lower()=='f':
        return FerMolecule(geometry=geometry,basis_set=basis_set,orbital_type=orbital_type,backend=backend,guess_wfn=guess_wfn,name=name,*args,**kwargs)
    else:
        raise TequilaException(f"Molecule Nature not identified: {nature}. Only qubit, hybrid and fermionic allowed.")

