import subprocess
import sys
import os
import shlex
from importlib import resources


def call_janpa(command:str,output_dir=None):
    """
    Wraps the binary executable.
    Usage: mypackage.call_binary("positional", option="value")
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
    print(output_dir)
    with resources.as_file(resources.files('sunrise.CLPO.bin').joinpath(filename)) as binary_path:
        
        # 2. Ensure executable permissions (Linux/Mac)
        if sys.platform != "win32":
            st = os.stat(binary_path)
            os.chmod(binary_path, st.st_mode | 0o111)

        cmd = [os.path.abspath(binary_path)]
        extra_args = shlex.split(command)
        cmd.extend(extra_args)
        # 4. Run the subprocess
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error executing binary: {e.stderr}")
            raise

def call_molden2aim(moldenfile:str,input_text:str="y\nn\nn\nn\n",output_dir=None):
    """
    Wraps the binary executable.
    Usage: mypackage.call_binary("positional", option="value")
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
            return result.stdout
        except subprocess.CalledProcessError as e:
            print(f"Error executing binary: {e.stderr}")
            raise