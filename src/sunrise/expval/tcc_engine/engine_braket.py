#  Copyright (c) 2023. The TenCirChem Developers. All Rights Reserved.
#
#  This file is distributed under ACADEMIC PUBLIC LICENSE
#  and WITHOUT ANY WARRANTY. See the LICENSE file for details.


from functools import partial
from typing import Tuple
import logging

import tensorcircuit as tc
from tencirchem.utils.backend import jit, value_and_grad
from tencirchem.static.hamiltonian import apply_op
from tencirchem.static.ci_utils import get_ci_strings
from tencirchem.static.evolve_civector import get_civector_nocache
from tencirchem.static.ucc import translate_init_state
from tencirchem.static.evolve_civector import get_civector as get_civector_
from tencirchem.static.evolve_statevector import get_statevector as get_statevector_
from tencirchem.static.evolve_tensornetwork import get_statevector_tensornetwork
from tencirchem.static.evolve_pyscf import get_civector_pyscf

from sunrise.expval.tcc_engine.evolve_pyscf import get_expval_and_grad_pyscf,get_energy_and_grad_pyscf
from sunrise.expval.tcc_engine.evolve_civector import get_energy_and_grad_civector,get_energy_and_grad_civector_nocache
from sunrise.expval.tcc_engine.evolve_civector import get_expval_and_grad_civector,get_expval_and_grad_civector_nocache
from tequila import Variable,Objective,simulate
from tencirchem.static.engine_ucc import get_ket,civector_to_statevector



logger = logging.getLogger(__name__)


GETVECTOR_MAP = {
    "tensornetwork": get_statevector_tensornetwork,
    "statevector": get_statevector_,
    "civector": get_civector_,
    "civector-large": get_civector_nocache,
    "pyscf": get_civector_pyscf,
}

def get_expval(angles, hamiltonian, n_qubits, n_elec_s,total_variables, 
    params,ex_ops: Tuple,engine ,mode: str = "fermion", init_state=None,
    params_bra=None,ex_ops_bra:Tuple=None,init_state_bra=None):
    'Now retuns also the overlap betwen states'
    if ex_ops_bra is None:
        ex_ops_bra = ex_ops
    if init_state_bra is None:
        init_state_bra = init_state
    if params_bra is None:
        params_bra = params
    assert len(angles)==len(total_variables)
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    pa = []
    for p in params:
        pa.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    pa = []
    for p in params_bra:
        pa.append(map_variables(p,dangles))
    map_params_bra = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    logger.info(f"Entering `get_energy`")
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
    init_state_ket = translate_init_state(init_state, n_qubits, ci_strings)
    init_state_bra = translate_init_state(init_state_bra, n_qubits, ci_strings)
    ket = GETVECTOR_MAP[engine](
        map_params, n_qubits, n_elec_s, tuple(ex_ops),  tuple([*range(len(ex_ops))]), mode=mode, init_state=init_state_ket
    )
    bra = GETVECTOR_MAP[engine](
        map_params_bra, n_qubits, n_elec_s, tuple(ex_ops_bra),  tuple([*range(len(ex_ops_bra))]), mode=mode, init_state=init_state_bra
    )
    hket = apply_op(hamiltonian, ket)
    return bra @ hket,bra @ ket

def get_energy(angles, hamiltonian, n_qubits, n_elec_s, total_variables,params,ex_ops, mode, init_state, engine):
    assert len(angles)==len(total_variables)
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    pa = []
    for p in params:
        pa.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    logger.info(f"Entering `get_energy`")
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
    init_state = translate_init_state(init_state, n_qubits, ci_strings)
    ket = GETVECTOR_MAP[engine](
        map_params, n_qubits, n_elec_s, tuple(ex_ops), tuple([*range(len(ex_ops))]), mode=mode, init_state=init_state
    )
    hket = apply_op(hamiltonian, ket)
    return ket @ hket

def get_statevector(angles, total_variables,params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state, engine):
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
    assert len(angles)==len(total_variables)
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    pa = []
    for p in params:
        pa.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    ket = get_ket(map_params, n_qubits, n_elec_s, ex_ops, param_ids, mode, init_state, ci_strings, engine)
    if engine.startswith("civector") or engine == "pyscf":
        statevector = civector_to_statevector(ket, n_qubits, ci_strings)
    else:
        statevector = ket
    return statevector

get_expval_statevector = partial(get_expval, engine="statevector")
try:
    get_expval_and_grad_statevector = jit(value_and_grad(get_expval_statevector), static_argnums=[2, 3, 4, 5, 6])
except NotImplementedError:

    def get_expval_and_grad_statevector(*args, **kwargs):
        raise NotImplementedError("Non JAX-backend for statevector engine")


get_expval_tensornetwork = partial(get_expval, engine="tensornetwork")
try:
    get_expval_and_grad_tensornetwork = jit(value_and_grad(get_expval_tensornetwork), static_argnums=[2, 3, 4, 5, 6])
except NotImplementedError:

    def get_expval_and_grad_tensornetwork(*args, **kwargs):
        raise NotImplementedError("Non JAX-backend for tensornetwork engine")


EXPVAL_AND_GRAD_MAP = {
    "tensornetwork": get_expval_and_grad_tensornetwork,
    "statevector": get_expval_and_grad_statevector,
    "civector": get_expval_and_grad_civector,
    "civector-large": get_expval_and_grad_civector_nocache,
    "pyscf": get_expval_and_grad_pyscf,
}

get_energy_statevector = partial(get_energy, engine="statevector")
try:
    get_energy_and_grad_statevector = jit(value_and_grad(get_energy_statevector), static_argnums=[2, 3, 4, 5, 6])
except NotImplementedError:

    def get_energy_and_grad_statevector(*args, **kwargs):
        raise NotImplementedError("Non JAX-backend for statevector engine")


get_energy_tensornetwork = partial(get_energy, engine="tensornetwork")
try:
    get_energy_and_grad_tensornetwork = jit(value_and_grad(get_energy_tensornetwork), static_argnums=[2, 3, 4, 5, 6])
except NotImplementedError:

    def get_energy_and_grad_tensornetwork(*args, **kwargs):
        raise NotImplementedError("Non JAX-backend for tensornetwork engine")


ENERGY_AND_GRAD_MAP = {
    "tensornetwork": get_energy_and_grad_tensornetwork,
    "statevector": get_energy_and_grad_statevector,
    "civector": get_energy_and_grad_civector,
    "civector-large": get_energy_and_grad_civector_nocache,
    "pyscf": get_energy_and_grad_pyscf,
}

def get_expval_and_grad(angles:list,hamiltonian, n_qubits, n_elec_s, engine, mode: str,total_variables, ex_ops: Tuple, params,params_bra,  init_state, ex_ops_bra: Tuple = None,  init_state_bra= None ):
    if engine not in EXPVAL_AND_GRAD_MAP:
        raise ValueError(f"Engine '{engine}' not supported")

    func = EXPVAL_AND_GRAD_MAP[engine]
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
    init_state = translate_init_state(init_state, n_qubits, ci_strings)
    init_state_bra = translate_init_state(init_state_bra, n_qubits, ci_strings)
    return func(angles, hamiltonian, n_qubits, n_elec_s, total_variables, params,tuple(ex_ops), mode, init_state,
    params_bra,tuple(ex_ops_bra),init_state_bra)

def get_energy_and_grad(angles, hamiltonian, n_qubits, n_elec_s, total_variables,params,ex_ops, mode, init_state, engine):
    if engine not in ENERGY_AND_GRAD_MAP:
        raise ValueError(f"Engine '{engine}' not supported")

    func = ENERGY_AND_GRAD_MAP[engine]
    ci_strings = get_ci_strings(n_qubits, n_elec_s, mode)
    init_state = translate_init_state(init_state, n_qubits, ci_strings)
    return func(angles, hamiltonian, n_qubits, n_elec_s, total_variables,params, tuple(ex_ops), mode, init_state)

def map_variables(x:list[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x