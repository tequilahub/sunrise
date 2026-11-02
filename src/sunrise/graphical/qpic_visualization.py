"""Wrappers around the ``qpic`` command line tool."""

import os
import subprocess


def qpic_to_pdf(filename, filepath=None):
    """Run ``qpic`` to convert ``<filename>.qpic`` into a PDF.

    Parameters
    ----------
    filename : str
        Name of the qpic file without the ``.qpic`` extension.
    filepath : str, optional
        Directory containing the file. Default is the current working directory.
    """
    if filepath:
        subprocess.call(["qpic", f"{filename}.qpic", "-f", "pdf"], cwd=filepath)
    else:
        subprocess.call(["qpic", f"{filename}.qpic", "-f", "pdf"], cwd=os.getcwd())


def qpic_to_png(filename, filepath=None):
    """Run ``qpic`` to convert ``<filename>.qpic`` into a PNG.

    Parameters
    ----------
    filename : str
        Name of the qpic file without the ``.qpic`` extension.
    filepath : str, optional
        Directory containing the file. Default is the current working directory.
    """
    if filepath:
        subprocess.call(["qpic", f"{filename}.qpic", "-f", "png"], cwd=filepath)
    else:
        subprocess.call(["qpic", f"{filename}.qpic", "-f", "png"], cwd=os.getcwd())
