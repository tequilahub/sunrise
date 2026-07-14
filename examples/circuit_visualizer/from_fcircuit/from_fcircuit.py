import tequila as tq
import sunrise as sun

U = sun.FCircuit()
U.initial_state = tq.gates.X([0,1,3,4])
U += sun.gates.UR(0,1,angle="a") + sun.gates.UR(2,3,angle="a") + sun.gates.UR(1,2,angle="a")
U += sun.gates.UC(1,2,angle="a")
U += sun.gates.FermionicExcitation(indices=[(0,2),(3,5)],angle="a",reordered=False)

U.export_to("from_fcircuit_example.qpic") # Create qpic file
U.export_to("from_fcircuit_example.pdf") # Create pdf file
# visual_circuit.export_to("from_fcircuit_example.png") # Create png file