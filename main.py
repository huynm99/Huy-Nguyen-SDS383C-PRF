# FULL NAME: HUY NGUYEN
# UT EID: hmn633
# FINAL PROJECT - STATISTICAL MODELING I


# ----- gess.py -------------

import numpy
from IPython import parallel

import fit_mvstud


def ess_update(x0, mu, cholSigma, logl, logf, tparams, cur_log_lklhd=None):
    calls = 0
    x0 = x0 - mu
    dim = x0.size
    assert cholSigma.shape == (dim, dim), "cholSigma is wrong size"

    if cur_log_lklhd is None:
        cur_log_lklhd = logl(x0 + mu, logf, tparams)
        calls += 1

    # Set up the ellipse and the slice threshold
    nu = numpy.dot(cholSigma, numpy.random.normal(size=dim))
    u = numpy.log(numpy.random.random())
    h = u + cur_log_lklhd

    # Bracket whole ellipse with both edges at first proposed point
    phi = numpy.random.random() * 2 * numpy.pi
    phi_min = phi - 2 * numpy.pi
    phi_max = phi

    # Slice sample the loop
    while True:
        # Compute x_prop for proposed angle difference and check if it
        # is on the slice
        x_prop = x0 * numpy.cos(phi) + nu * numpy.sin(phi)
        cur_log_lklhd = logl(x_prop + mu, logf, tparams)
        calls += 1
        if cur_log_lklhd > h:
            break
        # Shrink slice to rejected point
        if phi > 0:
            phi_max = phi
        elif phi < 0:
            phi_min = phi
        else:
            raise Exception("Shrunk to current position and still not acceptable.")
        # Propose new angle difference
        phi = numpy.random.random() * (phi_max - phi_min) + phi_min

    x1 = x_prop + mu
    return x1, calls, cur_log_lklhd, nu


def tess_update(x0, info_for_engines):
    (mu, Sigma, invSigma, cholSigma, nu, logf, logl, thinning, repeats) = info_for_engines
    dim = x0.size
    tparams = (dim, mu, invSigma, nu)

    x1 = x0
    cur_log_lklhd = logl(x1, logf, tparams)
    xs = numpy.zeros((repeats / thinning, dim))
    calls = 1

    for i in xrange(repeats):
        # Sample p(s|x)
        alpha = (dim + nu) / 2
        beta = (nu + numpy.dot(x1 - mu, numpy.dot(invSigma, x1 - mu))) / 2
        s = 1. / numpy.random.gamma(alpha, 1. / beta)

        # Sample p(x|s)
        chol_Sigma = numpy.sqrt(s) * cholSigma  # this equals chol(s*Sigma)
        x1, new_calls, cur_log_lklhd, ess_nu = ess_update(x1, mu, chol_Sigma, logl, logf, tparams, cur_log_lklhd)

        if (i + 1) % thinning == 0:
            xs[(i + 1) / thinning - 1] = x1
        calls += new_calls
    return xs, calls


def parallel_tess_update(group1, group2, logf, thinning, repeats, dview=None):
    (n, dim) = group1.shape
    mu, Sigma, nu = fit_mvstud.fit_mvstud(group2)
    if nu == numpy.Inf:
        nu = 1e6
    invSigma = numpy.linalg.inv(Sigma)
    try:
        cholSigma = numpy.linalg.cholesky(Sigma)
    except:
        print
        "Error: fit_mvstud failed to converge. This may mean that you are not using enough Markov chains."
        raise

    def logl(x, logf, tparams):
        def logt(x, tparams):
            (dim, mu, invSigma, nu) = tparams
            return -(dim + nu) / 2 * numpy.log(1 + numpy.dot(x - mu, numpy.dot(invSigma, x - mu)) / nu)

        return logf(x) - logt(x, tparams)

    info_for_engines = (mu, Sigma, invSigma, cholSigma, nu, logf, logl, thinning, repeats)

    if dview is None:
        print
        "Note that parallelism is not being used."
        results = map(tess_update, group1, n * [info_for_engines])
    else:
        results = dview.map_sync(tess_update, group1, n * [info_for_engines])

    samples = numpy.zeros((n, repeats / thinning, dim))
    calls = 0
    for i in xrange(n):
        samples[i, :, :] = results[i][0]
        calls += results[i][1]

    return samples, calls


def parallel_gess(chains, iters, burnin, thinning, starts, logf, repeats, dview=None):
    dim = starts.shape[1]
    assert starts.shape[0] == 2 * chains, "starts is wrong shape"
    assert iters % repeats == 0, "iters must be divisible by repeats"
    assert burnin % repeats == 0, "burnin must be divisible by repeats"
    assert repeats % thinning == 0, "repeats must be divisible by thinning"

    group1 = starts[:chains, :]
    group2 = starts[chains:, :]
    samples = numpy.zeros((2 * chains, iters / thinning, dim))
    calls = 0

    # do the sampling
    for i in xrange(-burnin / repeats, iters / repeats):
        samples1, calls1 = parallel_tess_update(group1, group2, logf, thinning, repeats, dview)
        group1 = samples1[:, repeats / thinning - 1, :]
        samples2, calls2 = parallel_tess_update(group2, group1, logf, thinning, repeats, dview)
        group2 = samples2[:, repeats / thinning - 1, :]

        if i >= 0:
            # update samples
            for j in xrange(chains):
                samples[j, i * repeats / thinning:(i + 1) * repeats / thinning, :] = samples1[j]
                samples[chains + j, i * repeats / thinning:(i + 1) * repeats / thinning, :] = samples2[j]
            calls += (calls1 + calls2)

    return samples, calls


#---------------- fit_mvstud.py -------
# import numpy
# from scipy import optimize
# from scipy import special
#
#
# def fit_mvstud(data, tolerance=1e-6):
#     def opt_nu(delta_iobs, nu):
#         def func0(nu):
#             w_iobs = (nu + dim) / (nu + delta_iobs)
#             f = -special.psi(nu / 2) + numpy.log(nu / 2) + numpy.sum(numpy.log(w_iobs)) / n - numpy.sum(
#                 w_iobs) / n + 1 + special.psi((nu + dim) / 2) - numpy.log((nu + dim) / 2)
#             return f
#
#         if func0(1e6) >= 0:
#             nu = numpy.inf
#         else:
#             nu = optimize.brentq(func0, 1e-6, 1e6)
#         return nu
#
#     data = data.T
#     (dim, n) = data.shape
#     mu = numpy.array([numpy.median(data, 1)]).T
#     Sigma = numpy.cov(data) * (n - 1) / n + 1e-1 * numpy.eye(dim)
#     nu = 20
#
#     last_nu = 0
#     i = 0
#     while numpy.abs(last_nu - nu) > tolerance:
#         i += 1
#         if i >= 500:
#             import cPickle
#             cPickle.dump(data, open('output/fit_mvstud_data.pickle', 'wb'))
#             break
#         diffs = data - mu
#         delta_iobs = numpy.sum(diffs * numpy.linalg.solve(Sigma, diffs), 0)
#
#         # update nu
#         last_nu = nu
#         nu = opt_nu(delta_iobs, nu)
#         if nu == numpy.inf:
#             return mu.T[0], Sigma, nu
#
#         w_iobs = (nu + dim) / (nu + delta_iobs)
#
#         # update Sigma
#         Sigma = numpy.dot(w_iobs * diffs, diffs.T) / n
#
#         # update mu
#         mu = numpy.sum(w_iobs * data, 1) / sum(w_iobs)
#         mu = numpy.array([mu]).T
#
#     return mu.T[0], Sigma, nu

#------------- example.py ------------
# from IPython import parallel
# import numpy
#
# import gess
#
# rc = parallel.Client(packer='pickle') # for running on starcluster on
#                                       # EC2. This line may have to be
#                                       # modified when running on other
#                                       # clusters
# dview = rc[:]
# print("using " + str(len(rc.ids)) + " engines")
#
# with dview.sync_imports():
#     import numpy
#     import os
#
# working_dir = os.getcwd()
# dview.execute("os.chdir('" + working_dir + "')")
# dview.execute("os.environ['MKL_NUM_THREADS']='1'") # prevent numpy from multithreading
#
# dim = 10
# num_cores = len(rc.ids)
# iters = 1000
# burnin = 1000
# thinning = 1
# repeats = 100
# chains = num_cores
#
# # a simple pdf
# def logf(x):
#     return -.5 * numpy.dot(x, x)
#
# starts = numpy.random.normal(loc=numpy.zeros(dim), scale=1, size=(2*chains,dim))
# samples, calls = gess.parallel_gess(chains, iters, burnin, thinning, starts, logf, repeats, dview)
