import numpy as np
import scipy.stats

import bayespf

def test_credible_interval():

    M = 50
    N = 100
    p_edge = 0.1


    in_interval_f = 0
    for i in range(M):
        f, p = np.random.rand(2)
        
        events = bayespf.generate(f, p, N)

        phases, fractions, r, P = bayespf.infer(events, n_phase=200, n_frac=401)

        frac_pdf = np.average(r,axis=0)
        lfi, ufi = np.searchsorted(np.cumsum(frac_pdf)/np.sum(frac_pdf), 
                [p_edge, 1-p_edge])

        if fractions[lfi]<=f and (ufi==len(fractions) or f<fractions[ufi]):
            in_interval_f += 1

    assert 0.01<scipy.stats.binom(M,1-2*p_edge).cdf(in_interval_f)<0.99

if __name__=='__main__':
    test_credible_interval()
