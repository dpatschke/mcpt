import numpy as np
import numba

@numba.jit(nopython=True)
def np_unique(a):
    b = np.sort(a.flatten())
    unique = list(b[:1])
    for x in b[1:]:
        if x != unique[-1]:
            unique.append(x)
    return unique
