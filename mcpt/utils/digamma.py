import numpy as np
import numba

@numba.vectorize(target='cpu')
def digamma_cpu(x):

    value = 0.0
    #  Use approximation for small argument.
    if ( x <= 0.000001 ):
        euler_mascheroni = 0.57721566490153286060
        value = -euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x
        return value

    #  Reduce to DIGAMA(X + N).
    while (x < 8.5):
        value = value - 1.0 / x
        x =  x + 1.0

    #  Use Stirling's (actually de Moivre's) expansion.
    r = 1.0 / x
    value = value + np.log(x) - 0.5 * r
    r = r * r
    value = value \
        - r * ( 1.0 / 12.0 \
        - r * ( 1.0 / 120.0 \
        - r * ( 1.0 / 252.0 \
        - r * ( 1.0 / 240.0 \
        - r * ( 1.0 / 132.0 ) ) ) ) )

    return value
