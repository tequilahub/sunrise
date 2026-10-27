# graphical – circuit drawing

Draws tequila and sunrise circuits as publication-quality figures using [qpic](https://github.com/qpic/qpic) (LaTeX/TikZ under the hood).

- `draw(circuit, ...)` – render a circuit; it can show it in **spin** or **spatial-orbital** view, with symbolic or numeric angles, and group gates into named blocks.
- `qpic_to_pdf`, `qpic_to_png` – convert an existing `.qpic` file.

<p align="center"><img src="../fermionic_operations/example_circuit.png" width="320" alt="SPA circuit"/></p>

```python
import sunrise as sn

U = sn.FCircuit.from_edges(n_orb=4, edges=[(0, 1), (2, 3)])
sn.draw(U, filename="spa")  # -> spa.png (also shown inline in Jupyter)
```

Requires `qpic`, LaTeX and ImageMagick (`convert`) for PDF/PNG output.

**Gallery:** [spin view](../../../examples/circuit_visualizer/basic_features/basic_spin_view.pdf) · [spatial view](../../../examples/circuit_visualizer/basic_features/basic_spatial_view.pdf) · [numeric angles](../../../examples/circuit_visualizer/basic_features/basic_numeric.pdf) · [more examples](../../../examples/circuit_visualizer)
