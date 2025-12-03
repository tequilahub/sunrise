import tequila as tq
import sunrise as sn
import random
import numpy as np
from datetime import datetime

geom = "H 0. 0. 0.\n Be 0. 0. 1.6\n H 0. 0. 3.2"
backend='tcc'
snmol = sn.Molecule(geometry=geom,basis_set='sto-3g',nature='f',units='a')
random.seed(datetime.now().timestamp())
snU = snmol.make_ansatz('UpCCSD')
snU.export_to('custom_gate.pdf',style={'single':'red','generic':"#AC61AC"}) 
v = {d:random.random()*np.pi for d in snU.extract_variables()}
snU = snU.map_variables(v)
snU.export_to('custom_range.pdf',style={'color_range':["#00FF6A",'red']})
