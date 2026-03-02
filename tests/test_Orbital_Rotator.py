import numpy as np
import tequila as tq
from sunrise import OrbitalRotation as OR


def test_Orbital_Rotation():
    idx = [0, 1]
    a = tq.QTensor(shape=(2, 2))
    a[0, 0] = tq.Variable("a00")
    a[0, 1] = tq.Variable('a01')
    a[1, 0] = tq.Variable('a10')
    a[1, 1] = tq.Variable('a11')
    RA = OR(orbitals=idx, matrix=a)
    variables = {'a00': 0, 'a01': 1, 'a10': 1, 'a11': 0}
    wfn_a = tq.wavefunction.qubit_wavefunction.QubitWaveFunction.from_string('1*|0011>')
    assert wfn_a.isclose(tq.simulate(tq.gates.X([0,1]) + RA, variables=variables))

    jdx = [1, 2]
    b = tq.QTensor(shape=(2, 2))
    b[0, 0] = tq.Variable("b00")
    b[0, 1] = tq.Variable("b00")
    b[1, 0] = tq.Variable("b00")
    b[1, 1] = tq.Variable("b00")
    variables.update({"b00": 1, "b01": -1, "b10": 1, "b11": 1})
    b = (1 / (np.sqrt(2))) * b
    RB = OR(orbitals=jdx,matrix=b)
    wfn_b = tq.wavefunction.qubit_wavefunction.QubitWaveFunction.from_string('0.5*|001100> - 0.5*|000110> + 0.5*|001001> + 0.5|000011>')
    assert wfn_b.isclose(tq.simulate(tq.gates.X([2,3]) + RB, variables=variables))

    RC = RA + RB
    RD = RB + RA
    wfn_c = tq.wavefunction.qubit_wavefunction.QubitWaveFunction.from_string('- 0.5*|001100> + 0.5*|000110> - 0.5*|001001> - 0.5*|000011>')
    wfn_d = tq.wavefunction.qubit_wavefunction.QubitWaveFunction.from_string('-1*|110000>')
    wfn_e = tq.wavefunction.qubit_wavefunction.QubitWaveFunction.from_string(' 1*|001100>')
    wfn_f = tq.wavefunction.qubit_wavefunction.QubitWaveFunction.from_string(' 0.5*|110000> - 0.5*|010010> + 0.5*|100001> + 0.5*|000011>')

    assert wfn_c.isclose(tq.simulate(objective=tq.gates.X([0,1])+RC, variables=variables))
    assert wfn_d.isclose(tq.simulate(objective=tq.gates.X([2,3])+RC, variables=variables))
    assert wfn_e.isclose(tq.simulate(objective=tq.gates.X([0,1])+RD, variables=variables))
    assert wfn_f.isclose(tq.simulate(objective=tq.gates.X([2,3])+RD, variables=variables))
