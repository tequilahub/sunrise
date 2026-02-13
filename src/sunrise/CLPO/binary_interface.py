import subprocess
import sys
import os
import shlex
from importlib import resources
from numpy import array

def call_janpa(command:str='',output_dir=None,silent=False):
    """
    Wraper for the janpa executable files. See CLPO.README.md for link.
    command:str Command string as it would be introduced on the terminal. See sunrise.call_janpa() for options
    silent:bool Whether to silece janpa output 
    output_dir: default None = working file.
    """
    if sys.platform == "darwin":
        filename = 'JANPA_macos'
    elif sys.platform == "linux" or sys.platform == "linux2":
        filename = 'JANPA_linux'
    elif sys.platform == "win32":
        raise NotImplementedError('Windows not implemented (yet?)')
    else: raise Exception('Is this code being run inside Doom?')
    
    if output_dir is None:
        output_dir = os.getcwd()
    with resources.as_file(resources.files('sunrise.CLPO.bin').joinpath(filename)) as binary_path:
        
        # 2. Ensure executable permissions (Linux/Mac)
        if sys.platform != "win32":
            st = os.stat(binary_path)
            os.chmod(binary_path, st.st_mode | 0o111)

        cmd = [os.path.abspath(binary_path)]
        if len(command):
            extra_args = shlex.split(command)
            cmd.extend(extra_args)
        # 4. Run the subprocess
        try:
            res = subprocess.run(
                cmd, 
                capture_output=silent, 
                text=True, 
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error executing binary: {e.stderr}")
            raise

def call_molden2aim(moldenfile:str,input_text:str="y\nn\nn\nn\n",output_dir=None):
    """
    Wraper for the molden2aim executable. See CLPO.README.md for link.
    moldenfile:str Modeln file such as 'mymolecule.molden'
    input_text:str String answering the input required by molden2aim
    output_dir: default None = working file.
    """
    if sys.platform == "darwin":
        filename = 'molden2aim_mc.exe'
    elif sys.platform == "linux" or sys.platform == "linux2":
        filename = 'molden2aim_lnx.exe'
    elif sys.platform == "win32":
        raise NotImplementedError('Windows not implemented (yet?)')
    else: raise Exception('Is this code being run inside Doom?')

    if output_dir is None:
        output_dir = os.getcwd()
    with resources.as_file(resources.files('sunrise.CLPO.bin').joinpath(filename)) as binary_path:
        
        # 2. Ensure executable permissions (Linux/Mac)
        if sys.platform != "win32":
            st = os.stat(binary_path)
            os.chmod(binary_path, st.st_mode | 0o111)
        cmd = [os.path.abspath(binary_path)]
        cmd.extend(['-i']) #{output_dir}/
        cmd.extend([f'{output_dir}/{moldenfile}'])
        # 4. Run the subprocess
        try:
            result = subprocess.run(
                cmd,
                input=input_text,
                capture_output=True, 
                text=True, 
                check=True,
                cwd=output_dir
            )
        except subprocess.CalledProcessError as e:
            print(f"Error executing binary: {e.stderr}")
            raise

def extract_clpo_graph(graph_file:str):
    """
    Reads a CLPO 'graph' file and extracts a graph representation.

    Returns:
        List of tuples:
        - (i,)      for lone pairs
        - (i, i+1)  for BD/NB pairs
    """
    nodes = []

    with open(graph_file, "r") as f:
        lines = f.readlines()

    # Skip header
    data_lines = [line.rstrip() for line in lines if line.strip()][1:]

    i = 0
    while i < len(data_lines):
        line = data_lines[i]

        # Lone pair
        if "(LP)" in line:
            nodes.append((i,))
            i += 1
            continue

        # Bonding orbital → must pair with next line
        if "(BD)" in line:
            if i + 1 >= len(data_lines):
                raise ValueError("BD entry without following NB line")

            nodes.append((i, i + 1))
            i += 2
            continue

        i += 1

    return nodes

def read_molden_mo_matrix(filename:str):
    """
    Reads a Molden file and returns a NumPy array where
    each column is an MO coefficient vector.
    """
    mo_vectors = []
    current_mo = []
    in_mo_section = False

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()

            # Detect MO section
            if line == "[MO]":
                in_mo_section = True
                continue

            if not in_mo_section:
                continue

            if line.startswith("Sym="):
                if current_mo:
                    mo_vectors.append(current_mo)
                    current_mo = []
                continue

            # Skip metadata lines
            if (
                line.startswith("Ene=") or
                line.startswith("Spin=") or
                line.startswith("Occup=") or
                not line
            ):
                continue

            parts = line.split()
            if len(parts) == 2:
                try:
                    coeff = float(parts[1])
                    current_mo.append(coeff)
                except ValueError:
                    pass

        if current_mo:
            mo_vectors.append(current_mo)

    matrix = array(mo_vectors).T
    return matrix
