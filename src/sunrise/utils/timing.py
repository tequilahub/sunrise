import time

_silent = True
_all_timinings = {}


def reset_timings():
    _all_timinings = {}


def analyse(data=None):
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
