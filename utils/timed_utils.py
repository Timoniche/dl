import time


def timed(f):
    def wrapper(*args, **kwds):
        start = time.time_ns()
        result = f(*args, **kwds)
        elapsed = (time.time_ns() - start) // 1e6
        print("%s took %d ms to finish" % (f.__name__, elapsed))
        return result

    return wrapper
