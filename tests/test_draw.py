import tequila as tq

import sunrise as sn
from sunrise.graphical.qpic_visualization import qpic_to_matplotlib


def test_draw_without_qpic(tmp_path):
    U = sn.gates.FermionicExcitation([(0, 2), (1, 3)], "a")
    U.initial_state = tq.gates.X([0, 1])
    U += sn.gates.UC(0, 1, "b")
    U += sn.gates.UR(1, 2, 0.5)
    U += sn.gates.FermionicExcitation([(1, 2)], "d", reordered=True)
    U.export_to("circuit.qpic", filepath=str(tmp_path))
    for ftype in ("png", "pdf"):
        qpic_to_matplotlib("circuit", str(tmp_path), ftype)
        assert (tmp_path / f"circuit.{ftype}").stat().st_size > 0
