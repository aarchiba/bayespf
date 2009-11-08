import numpy as np
import scipy.stats

def pdf_data_given_model(fraction, phase, x):
    return fraction*(1+np.cos(2*np.pi*(x-phase)))+(1-fraction)

def generate(fraction, phase, n):
    m = np.random.binomial(n, fraction)
    pulsed = np.random.rand(m)
    c = np.sin(2*np.pi*pulsed)>np.random.rand(m)
    pulsed[c] *= -1
    pulsed += 0.25+phase
    pulsed %= 1
    
    r = np.concatenate((pulsed, np.random.rand(n-m)))
    np.random.shuffle(r)
    return r


def infer(events, n_phase=200, n_frac=201):

    events = np.asarray(events)
    phases = np.linspace(0,1,n_phase,endpoint=False)
    fractions = np.linspace(0,1,n_frac)

    lpdf = np.zeros((n_phase, n_frac))
    for e in events:
        lpdf += np.log(pdf_data_given_model(fractions, phases[:,np.newaxis], e))

    mx = np.amax(lpdf)
    p = np.exp(lpdf - mx)/np.average(np.exp(lpdf-mx))
    
    S = np.average(np.exp(lpdf))

    return phases, fractions, p, S/(S+1)


if __name__=='__main__':
    import pylab as pl
    
    events = generate(0.05,0.5,8000)
    phases, fractions, r, P = infer(events)
    print "Probability the signal is pulsed: %f" % P

    pl.subplot(211)
    pl.contourf(fractions, phases, r)
    pl.xlabel("Pulsed fraction")
    pl.ylabel("Phase")

    pl.subplot(212)
    p = np.average(r,axis=0)
    li, mi, ui = np.searchsorted(np.cumsum(p)/np.sum(p), [scipy.stats.norm.cdf(-1), 0.5, scipy.stats.norm.cdf(1)])
    pl.plot(fractions, p)
    pl.xlabel("Pulsed fraction")
    pl.ylabel("Probability")
    pl.axvline(fractions[li])
    pl.axvline(fractions[mi])
    pl.axvline(fractions[ui])
    print "Pulsed fraction: %f [%f, %f]" % (fractions[mi], fractions[li], fractions[ui])

    pl.show()
