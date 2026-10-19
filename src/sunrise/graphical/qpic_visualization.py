import subprocess
import os


def qpic_to_pdf(filename, filepath=None):
    """Calls qpic to generate a pdf from the given apic file."""
    if filepath:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "pdf"], cwd=filepath)
    else:
        subprocess.call(
            ["qpic", "{}.qpic".format(filename), "-f", "pdf"], cwd=os.getcwd()
        )


def qpic_to_png(filename, filepath=None):
    """Calls qpic to generate a png from the given apic file."""

    if filepath:
        subprocess.call(["qpic", "{}.qpic".format(filename), "-f", "png"], cwd=filepath)
    else:
        subprocess.call(
            ["qpic", "{}.qpic".format(filename), "-f", "png"], cwd=os.getcwd()
        )
