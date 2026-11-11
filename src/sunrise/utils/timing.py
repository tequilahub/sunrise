"""Minimal timing decorator that records the run time of decorated functions."""

import time

_silent = True
_all_timinings = {}


def reset_timings():
    """Reset the recorded timings."""
    _all_timinings = {}


def analyse(data=None):
    """Print max, min and average run time per timed function.

    Parameters
    ----------
    data : dict, optional
        Mapping ``function name -> list of run times in seconds``. Default is the
        timings recorded by ``timing``.

    Returns
    -------
    tuple of float
        ``(average, max, min)`` of the first entry in ``data``.
    """
    if data is None:
        data = _all_timinings
    for k, v in data.items():
        average = sum(v) / len(v)
        vmax = max(v)
        vmin = min(v)

        print(f"{k:30}")
        print(f"max : {vmax:2.5f}s")
        print(f"min : {vmin:2.5f}s")
        print(f"av  : {average:2.5f}s")

        return average, vmax, vmin


def timing(f, *args, **kwargs):
    """Wrap ``f`` so that each call records its wall-clock time.

    Parameters
    ----------
    f : callable
        Function to time. Times are stored per ``f.__name__`` and printed
        after each call only if the module flag ``_silent`` is False.

    Returns
    -------
    callable
        The timed version of ``f``.
    """

    def timed_function(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        if not _silent:
            print(f"{f.__name__:30} took {end - start}s")
        if f.__name__ in _all_timinings:
            _all_timinings[f.__name__] += [end - start]
        else:
            _all_timinings[f.__name__] = [end - start]
        return result

    return timed_function
