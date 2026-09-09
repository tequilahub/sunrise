import os
import hashlib

def draw(circuit, filename=None, width=None, height=300, *args, **kwargs):
    """
    Draw a Sunrise FCircuit or Tequila Objective using the Sunrise graphical engine.
    
    Parameters
    ----------
    circuit : FCircuit, QCircuit, or Objective
        The circuit or objective to draw.
    filename : str, optional
        Name of the output file. Defaults to a unique temporary name.
    width : int, optional
        Width of the displayed image in Jupyter.
    height : int, optional
        Height of the displayed image in Jupyter.
    *args, **kwargs :
        Additional arguments passed to FCircuit.export_to (e.g., show_spatial_orbitals, style).
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
                draw(E.U, filename=fname, width=width, height=height, *args, **kwargs)
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
        tq.draw(circuit, filename=filename, *args, **kwargs)
        return
        
    # Display inline if running in a Jupyter Notebook / IPython environment
    try:
        from IPython.display import Image, display
        display(Image(filename=filename, width=width, height=height))
    except ImportError:
        print(f"Circuit drawn and saved to: {os.path.abspath(filename)}")