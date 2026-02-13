import time

_silent = True
_all_timinings = {}

def reset_timings():
    _all_timinings = {}

def analyse(data=None):
    if data is None:
        data = _all_timinings
    for k,v in data.items():
        average = sum(v)/len(v)
        vmax = max(v)
        vmin = min(v)

        print("{:30}".format(k))
        print("max : {:2.5f}s".format(vmax))
        print("min : {:2.5f}s".format(vmin))
        print("av  : {:2.5f}s".format(average))
        
        return average, vmax, vmin

def timing(f, *args, **kwargs):
    def timed_function(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        if not _silent: print("{:30} took {}s".format(f.__name__, end - start))
        if f.__name__ in _all_timinings:
            _all_timinings[f.__name__] += [end - start]
        else:
            _all_timinings[f.__name__] = [end-start]
        return result
    return timed_function
