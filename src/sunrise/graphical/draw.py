"""Top-level ``draw`` function for Sunrise and tequila circuits."""

import hashlib
import os


def draw(circuit, filename=None, width=None, height=300, *args, **kwargs):
    """Draw an FCircuit, QCircuit or tequila Objective as a PNG image.

    FCircuits are rendered with the Sunrise engine (via ``FCircuit.export_to``),
    plain tequila QCircuits fall back to ``tq.draw``. For an Objective, every
    distinct expectation value circuit is drawn to ``<filename>_<i>.png``. In
    Jupyter the image is displayed inline, otherwise its path is printed.

    Parameters
    ----------
    circuit : FCircuit, tq.QCircuit or tq.Objective
        The circuit or objective to draw. ExpectationValues are unwrapped to their circuit.
    filename : str, optional
        Output file name; the extension is forced to ``.png``. Default is a
        unique name ``sunrise_draw_<hash>.png``.
    width : int, optional
        Width of the image displayed in Jupyter.
    height : int, optional
        Height of the image displayed in Jupyter. Default is 300.
    *args, **kwargs
        Passed to ``FCircuit.export_to`` (e.g. ``style``). A ``backend`` keyword is ignored.

    Examples
    --------
    >>> import sunrise as sun
    >>> sun.draw(circuit, filename="my_circuit.png")
    """
    from tequila.objective import Objective

    # We always use the Sunrise engine for FCircuits, so ignore tequila's backend kwarg
    kwargs.pop("backend", None)

    # If it's an Objective, draw all unique circuits contained within it
    if isinstance(circuit, Objective):
        drawn = {}
        for i, E in enumerate(circuit.get_expectationvalues()):
            if E in drawn:
                print(f"\nExpectation Value {i} is the same as {drawn[E]}")
            else:
                print(f"\nExpectation Value {i}:")
                print(f"total measurements = {E.count_measurements()}")
                variables = E.U.extract_variables()
                print(f"variables          = {len(variables)}")
                fname = f"{filename or 'obj'}_{i}.png"
                print(f"circuit            = {fname}")
                draw(E.U, *args, filename=fname, width=width, height=height, **kwargs)
            drawn[E] = i
        return

    # Unwrap ExpectationValues or Compiled Backend Circuits
    if hasattr(circuit, "U"):
        circuit = circuit.U
    if hasattr(circuit, "abstract_circuit"):
        circuit = circuit.abstract_circuit

    # Generate a unique temporary filename if none is provided
    if filename is None:
        h = hashlib.md5(str(circuit).encode()).hexdigest()[:8]
        filename = f"sunrise_draw_{h}.png"

    # Ensure we are exporting to PNG for display
    if not filename.endswith(".png"):
        filename = filename.split(".")[0] + ".png"

    # Route to Sunrise FCircuit engine, or fallback to Tequila for standard QCircuits
    if hasattr(circuit, "export_to"):
        circuit.export_to(filename, *args, **kwargs)
    else:
        import tequila as tq

        tq.draw(circuit, *args, filename=filename, **kwargs)
        return

    # Display inline if running in a Jupyter Notebook / IPython environment
    try:
        from IPython.display import Image, display

        display(Image(filename=filename, width=width, height=height))
    except ImportError:
        print(f"Circuit drawn and saved to: {os.path.abspath(filename)}")
