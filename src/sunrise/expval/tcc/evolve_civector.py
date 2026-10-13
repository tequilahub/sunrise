from tencirchem.static.evolve_civector import *
from tencirchem.static.evolve_civector import _get_gradients_civector,_get_gradients_civector_nocache
from tequila import simulate,grad,QTensor
from tequila.objective.objective import Objective,Variable,FixedVariable
import tensorcircuit as tc
from typing import  Union
from numpy import zeros

def get_expval_and_grad_civector(
    angles, hamiltonian, n_qubits, n_elec_s,total_variables,
    params,ex_ops: Tuple, mode: str = "fermion", init_state=None,
    params_bra=None,ex_ops_bra:Union[Tuple,None]=None,init_state_bra=None):
    #Fist check bra
    if ex_ops_bra is None: ex_ops_bra = ex_ops
    if params_bra is None: params_bra = params
    if init_state_bra is None: init_state_bra = init_state
    #We receive list [x0,x1,...] refering to the total variables
    assert len(angles)==len(total_variables)
    #Then we create {v0:x0,v1:x1,...}
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    #To build the CIVector we need the already maped params
    pa = []
    for p in params:
        pa.append(map_variables(p,dangles))
    pa_bra = []
    for p in params:
        pa_bra.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    map_params_bra = tc.backend.numpy(tc.backend.convert_to_tensor(pa_bra).astype(tc.rdtypestr))
    ket = get_civector(map_params    ,  n_qubits, n_elec_s, ex_ops    , [*range(len(ex_ops))]    , mode, init_state)
    bra = get_civector(map_params_bra, n_qubits , n_elec_s, ex_ops_bra, [*range(len(ex_ops_bra))], mode, init_state_bra)
    
    #Compute Hvects and energy 
    hbra = tc.backend.numpy(apply_op(hamiltonian, bra))
    hket = tc.backend.numpy(apply_op(hamiltonian, ket))
    energy = hbra @ ket
    #We got the gradient with respect to each ex_op, since assumed each ex_ops has different theta 
    # already cached
    op_tensors = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode=mode)
    op_tensors_bra = get_operator_tensors(n_qubits, n_elec_s, ex_ops_bra, mode=mode)

    theta_tensors = get_theta_tensors(map_params, [*range(len(ex_ops))])
    theta_tensors_bra = get_theta_tensors(map_params_bra, [*range(len(ex_ops_bra))])
    op_tensors = list(op_tensors) + list(theta_tensors)
    op_tensors_bra = list(op_tensors_bra) + list(theta_tensors_bra)
    
    
    gradients_beforesum = _get_gradients_civector(hbra, ket, *op_tensors[1:])
    gradients_beforesum_bra = _get_gradients_civector(hket, bra, *op_tensors_bra[1:])

    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    gradients_beforesum_bra = tc.backend.numpy(gradients_beforesum_bra)
    #Now we have to compute the gradient of all parameters with respect to all possible variables
    ang_grad = np.zeros((len(params),len(total_variables)))
    ang_grad_bra = np.zeros((len(params_bra),len(total_variables)))
    #TODO: Improve this using tequila Objective, and copy paste in the other functions
    for i,pa in enumerate(params):
        if isinstance(pa,FixedVariable):
            continue
        for j,an in enumerate(total_variables):
            ang_grad[i,j]=simulate(grad(1*pa,an),variables=dangles) 
    for i,pa in enumerate(params_bra):
        if isinstance(pa,FixedVariable):
            continue
        for j,an in enumerate(total_variables):
            ang_grad_bra[i,j]=simulate(grad(1*pa,an),variables=dangles)
    gradients = np.add(gradients_beforesum.dot(ang_grad),gradients_beforesum_bra.dot(ang_grad_bra))

    return energy,  gradients, bra @ ket

def get_energy_and_grad_civector(
    angles, hamiltonian, n_qubits, n_elec_s, total_variables,
    params, ex_ops: Tuple,  mode: str = "fermion", init_state=None):
    
    assert len(angles)==len(total_variables)
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    pa = []
    for p in params: 
        pa.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    ket = get_civector(map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket
    # already cached
    op_tensors = get_operator_tensors(n_qubits, n_elec_s, ex_ops, mode=mode)
    theta_tensors = get_theta_tensors(map_params, [*range(len(ex_ops))])
    op_tensors = list(op_tensors) + list(theta_tensors)
    gradients_beforesum = _get_gradients_civector(bra, ket, *op_tensors[1:])
    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    ang_grad = np.zeros((len(params),len(total_variables)))
    for i,pa in enumerate(params):
        if isinstance(pa,FixedVariable):
            continue
        for j,an in enumerate(total_variables):
            ang_grad[i,j]=simulate(grad(1*pa,an),variables=dangles)
    gradients = gradients_beforesum.dot(ang_grad)
    return energy, 2 * gradients

def get_expval_and_grad_civector_nocache(
    angles, hamiltonian, n_qubits, n_elec_s,total_variables,
    params,ex_ops: Tuple, mode: str = "fermion", init_state=None,
    params_bra=None,ex_ops_bra:Tuple=None,init_state_bra=None):
    
    if ex_ops_bra is None: ex_ops_bra = ex_ops
    if params_bra is None: params_bra = params
    if init_state_bra is None: init_state_bra = init_state
    assert len(angles)==len(total_variables)
    #Then we create {v0:x0,v1:x1,...}
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    #To build the CIVector we need the already maped params
    pa = []
    for p in params:
        pa.append(map_variables(p,dangles))
    pa_bra = []
    for p in params:
        pa_bra.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))
    map_params_bra = tc.backend.numpy(tc.backend.convert_to_tensor(pa_bra).astype(tc.rdtypestr))
    ket = get_civector_nocache(map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode, init_state)
    bra = get_civector_nocache(map_params_bra, n_qubits, n_elec_s, ex_ops_bra, [*range(len(ex_ops_bra))], mode, init_state_bra)
    hbra = apply_op(hamiltonian, bra)
    hket = apply_op(hamiltonian, ket)
    energy = hbra @ ket

    gradients_beforesum = _get_gradients_civector_nocache(hbra, ket, map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode)
    gradients_beforesum = tc.backend.numpy(gradients_beforesum)
    gradients_beforesum_bra = _get_gradients_civector_nocache(hket, bra, map_params_bra, n_qubits, n_elec_s, ex_ops_bra, [*range(len(ex_ops_bra))], mode)
    gradients_beforesum_bra = tc.backend.numpy(gradients_beforesum_bra)

    ang_grad = np.zeros((len(params),len(total_variables)))
    ang_grad_bra = np.zeros((len(params_bra),len(total_variables)))
    for i,pa in enumerate(params):
        if isinstance(pa,FixedVariable):
            continue
        for j,an in enumerate(total_variables):
            ang_grad[i,j]=simulate(grad(1*pa,an),variables=dangles) 
    for i,pa in enumerate(params_bra):
        if isinstance(pa,FixedVariable):
            continue
        for j,an in enumerate(total_variables):
            ang_grad_bra[i,j]=simulate(grad(1*pa,an),variables=dangles)
    gradients = np.add(gradients_beforesum.dot(ang_grad),gradients_beforesum_bra.dot(ang_grad_bra))

    return energy, gradients, bra @ ket

def get_energy_and_grad_civector_nocache(
    angles, hamiltonian, n_qubits, n_elec_s, total_variables,
    params, ex_ops: Tuple,  mode: str = "fermion", init_state=None):

    assert len(angles)==len(total_variables)
    dangles = {total_variables[i].name:angles[i] for i in range(len(angles))}
    pa = []
    for p in params:
        pa.append(map_variables(p,dangles))
    map_params = tc.backend.numpy(tc.backend.convert_to_tensor(pa).astype(tc.rdtypestr))

    ket = get_civector_nocache(map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode, init_state)
    bra = apply_op(hamiltonian, ket)
    energy = bra @ ket

    gradients_beforesum = _get_gradients_civector_nocache(bra, ket, map_params, n_qubits, n_elec_s, ex_ops, [*range(len(ex_ops))], mode)
    gradients_beforesum = tc.backend.numpy(gradients_beforesum)

    ang_grad = np.zeros((len(params),len(total_variables)))
    for i,pa in enumerate(params):
        if isinstance(pa,FixedVariable):
            continue
        for j,an in enumerate(total_variables):
            ang_grad[i,j]=simulate(grad(1*pa,an),variables=dangles)
    gradients = gradients_beforesum.dot(ang_grad)
    
    return energy, 2 * gradients

def map_variables(x:Union[Variable,Objective],dvariables:dict):
    if isinstance(x,Variable):
        x = x.map_variables(dvariables)
    elif isinstance(x,Objective):
        x=simulate(x,dvariables)
    return x