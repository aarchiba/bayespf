import numpy as np
import scipy.stats

import bayespf

def test_credible_interval():

    M = 200
    N = 30

    ps = []
    for i in range(M):
        f, p = np.random.rand(2)
        
        events = bayespf.generate(f, p, N)

        phases, fractions, r, P = bayespf.infer(events)

        frac_pdf = np.average(r,axis=0)
        fi = np.searchsorted(fractions, f/2)
        p = np.sum(frac_pdf[:fi])/np.sum(frac_pdf)
        p = 2*min(p, 1-p)

        ps.append(p)


    assert scipy.stats.kstest(ps,lambda x: x)[1] > 0.01

def test_pulsed_probability():
    np.random.seed(0)
    p_accept = 0.75
    M = 50
    N = 30

    accepted = 0
    correct = 0
    total = 0
    while accepted<M:
        if np.random.rand()>0.5:
            f, p = np.random.rand(2)
            pulsed = True
        else:
            f, p = 0., 0.
            pulsed = False

        events = bayespf.generate(f, p, N)

        phases, fractions, r, P = bayespf.infer(events, n_phase=100, n_frac=101)

        if P>=p_accept:
            accepted += 1
            if pulsed:
                correct += 1
        if P<1-p_accept:
            accepted += 1
            if not pulsed:
                correct += 1
        total += 1
        if accepted!=0:
            fac = correct/float(accepted)
        else:
            fac = "NaN"

        print "P = %f\tfraction correct\t%s\t(fraction %g accepted)" % (P, fac, accepted/float(total))

    print "cdf value %f" % (scipy.stats.binom(M,p_accept).cdf(correct),)
    assert 0.01<scipy.stats.binom(M,p_accept).cdf(correct)

if __name__=='__main__':
    test_credible_interval()
    test_pulsed_probability()
