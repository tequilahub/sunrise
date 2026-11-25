"""Small console decorations: an ASCII-art fish and a fish progress bar."""

import sys


def giuseppe():
    """Print an ASCII-art fish."""
    giuseppe = r'''
                                                        .
            ___---""""""""""""""---____                / |
       _--""   \)))))))))))))))))))))))"""""___       /  |
    _-" _       \))))))_-"|))))))))))))))))))))"""---' __|
 _-"   / \       |))))|   |)))))))))))))))))))))))))/--  |
&lt;___.  \_/       |))))|   |))))))))))))))))))))))))&lt;-    |
 "-_             |))))|   |)))))))))))))))))))))))))\--__|
    "-_         /)))))) -_|))))))))))))))))))))___---.   |
       "--__   /)))))))))))))))))))))))_____"""       \  |
            """---______________---""""        mga     \ |
                                                        '
    '''
    print(giuseppe)


def giuseppe_bar(step: int, total_steps: int, toolbar_width: int = 50):
    """Draw a fish progress bar on stdout, overwriting the current line.

    Parameters
    ----------
    step : int
        Number of completed steps.
    total_steps : int
        Total number of steps.
    toolbar_width : int, optional
        Approximate bar width in characters. Rounded down to a multiple of ``total_steps``.
    """
    toolbar_width = total_steps * (toolbar_width // total_steps)
    step_width = toolbar_width // total_steps
    i = step * step_width
    sys.stdout.write(f"\r\b[{i * '~'}><(((º>{(toolbar_width - i) * ' '}]")
    sys.stdout.flush()
