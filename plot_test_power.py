import numpy as np
import pylab as pl

import bayespf


def accumulate_power_results(N=100,M=100):

    rr = []
    for i in range(M):
        if np.random.rand()>0.5:
            f, p = np.random.rand(2)
            pulsed = True
        else:
            f, p = 0., 0.
            pulsed = False

        events = bayespf.generate(f, p, N)

        phases, fractions, r, P = bayespf.infer(events, n_phase=100, n_frac=101)
        
        rr.append((P,pulsed))

    return rr

def O(p):
    return p/(1-p)
def plot_power_results(r):
    r = sorted(r)

    tot = 0
    rr = []
    for (P,pulsed) in r:
        if pulsed:
            tot += 1
        rr.append((P,tot))

    pl.loglog([O(p) for (p,t) in rr], [O(float(t)/(1+rr[-1][1])) for (p,t) in rr])

if __name__=='__main__':
    plot_power_results(accumulate_power_results())
    pl.show()
