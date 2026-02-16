import numpy
import time
import tequila as tq
from .decompose import decompose

def tqcomp(*args, **kwargs):
    # print("\nHERE WE ARE\n")
    return tq.compile(*args, **kwargs)

def fast_qtensor(qtensor, variables, do_decompose=False, evaluate=True,grouping:int=None,backend='qulacs'):
    simple = True
    shape = qtensor.shape
    n = 1
    for x in shape:
        n *= x
    flat = qtensor.reshape([n])
    ulist = {}
    circuit_ids = {}
    start = time.time()
    for i in range(n):
        E = flat[i]
        if len(E.get_expectationvalues()) > 1:
            simple=False
            break
        E = E.get_expectationvalues()[0]
        if len(E.H)>1:
            simple = False
            break

        H = E.H[0].simplify(threshold=1.e-6)
        U = E.U
        idx = hash(str(U))
        circuit_ids[idx]=U
        if idx in ulist:
            ulist[idx][i]=H
        else:
            ulist[idx] = {i:H}

    #print("first half: {}s".format(time.time()-start))
    start = time.time()
    if not simple:
        # print("are we here?")
        return tq.simulate(qtensor, variables)

    values = [0]*n

    #print(len(ulist))
    for idx,Hlist in ulist.items():
        U = circuit_ids[idx]
        H = list(Hlist.values())
        if do_decompose:
            tmp = decompose(H, U, nolist=False,grouping=grouping)
        else:
            tmp = [tq.ExpectationValue(H=h, U=U) for h in H]#, shape=[len(Hlist)])
            tmp = tq.QTensor(tmp, shape=[len(tmp)])

        if evaluate:
            # print("we evaluate?")
            vars = {x:variables[x] for x in variables.keys() if x in tmp.extract_variables()}
            tmp = tq.simulate(tmp, vars,backend=backend)
        else:
            #xstart = time.time()
            tmp = tqcomp(tmp,backend=backend)
            #print("compiling: {}s".format(time.time() - xstart))

        if len(Hlist) == 1:
            tmp = [tmp]

        keys = list(Hlist.keys())
        for i in range(len(H)):
            values[keys[i]] = tmp[i]

    #print("second half: {}s".format(time.time()-start))

    if evaluate:
        values = numpy.asarray(values)
        return values.reshape(shape),shape
    else:
        return values, shape




if __name__ == "__main__":

    mol = tq.Molecule(geometry="Be 0.0 0.0 0.0", basis_set="sto-3g")
    U = mol.make_ansatz(name="UpCCGSD")
    H = mol.make_hamiltonian()
    E = tq.ExpectationValue(H=H, U=U)
    CE = tq.compile(E)

    variables = {k:1.0 for k in U.extract_variables()}
    rdm1, rdm2 = mol.compute_rdms(U=U, evaluate=False, spin_free=True)
    print(rdm1)
    print(rdm2)
    start = time.time()
    yrdm1 = tq.simulate(rdm1, variables=variables)
    print(time.time() - start)
    start = time.time()
    zrdm1 = fast_qtensor(rdm1, variables=variables)
    print(time.time() - start)

    start = time.time()
    yrdm2 = tq.simulate(rdm2, variables=variables)
    print(time.time() - start)

    start = time.time()
    zrdm2 = fast_qtensor(rdm2, variables=variables)
    print(time.time() - start)
    xrdm1, xrdm2 = mol.compute_rdms(U=U, variables=variables, evaluate=True, spin_free=True)

    print(sum((xrdm1 - zrdm1).flatten()))
    print(sum((xrdm1 - yrdm1).flatten()))
    print(sum((xrdm2 - yrdm2).flatten()))


